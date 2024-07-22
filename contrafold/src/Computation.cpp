/////////////////////////////////////////////////////////////////
// Computation.cpp
//
// This class provides a wrapper around a generic
// DistributedComputation object that provides specific
// functionality such as
// 
//   - forming output filenames
//   - predicting structures for all input sequences
//   - performing computations for the optimizer.
/////////////////////////////////////////////////////////////////

#include "Computation.hpp"

/////////////////////////////////////////////////////////////////
// Computation::FormatOutputFilename()
//
// Decide on output filename, if any.  The arguments to this
// function consist of (1) a boolean variable indicating whether
// the output destination should be treated as the name of an
// output directory (and the output filename is chosen to match
// the input file) or whether the output destination should be
// interpreted as the output filename; (2) the name of the input 
// file to be processed; and (3) the supplied output destination. 
/////////////////////////////////////////////////////////////////

std::string Computation::FormatOutputFilename(bool output_to_directory, 
                                              const std::string &input_filename, 
                                              const std::string &output_destination) const 
{
    if (output_destination == "") return "";
    if (!output_to_directory) return output_destination;
    
    std::string::size_type separator_pos = input_filename.find_last_of(DIR_SEPARATOR_CHAR);
    std::string base_filename = (separator_pos == std::string::npos ? input_filename : input_filename.substr(separator_pos+1));
    return output_destination + DIR_SEPARATOR_CHAR + base_filename;
}

/////////////////////////////////////////////////////////////////
// Computation::Computation()
// Computation::~Computation()
//
// Constructor and destructor.
/////////////////////////////////////////////////////////////////

Computation::Computation(const std::vector<std::string> &filenames,
                         const std::string &output_parens_destination,
                         const std::string &output_bpseq_destination,
                         const std::string &output_posteriors_destination,
                         bool toggle_complementary_only,
                         bool toggle_use_constraints,
                         bool toggle_partition,
                         bool toggle_viterbi,
                         bool toggle_verbose,
                         double posterior_cutoff) : 
    DistributedComputation(toggle_verbose), toggle_complementary_only(toggle_complementary_only),
    toggle_use_constraints(toggle_use_constraints), toggle_partition(toggle_partition), toggle_viterbi(toggle_viterbi),
    posterior_cutoff(posterior_cutoff) 
{
    // build file descriptions
    
    FileDescription f; 
    for (size_t i = 0; i < filenames.size(); i++)
    {
        SStruct sstruct(filenames[i]);
        f.input_filename = filenames[i];
        f.output_parens_filename = FormatOutputFilename(filenames.size() > 1, filenames[i], output_parens_destination);
        f.output_bpseq_filename = FormatOutputFilename(filenames.size() > 1, filenames[i], output_bpseq_destination);
        f.output_posteriors_filename = FormatOutputFilename(filenames.size() > 1, filenames[i], output_posteriors_destination);
        f.size = int(Pow(double(sstruct.GetLength()), 3.0));
        f.weight = 1.0;
        file_descriptions.push_back(f);
    }
}

Computation::~Computation()
{}

/////////////////////////////////////////////////////////////////
// Computation::GetAllUnits()
//
// Return a vector containing the index of every input file.
/////////////////////////////////////////////////////////////////

std::vector<int> Computation::GetAllUnits() const 
{
    std::vector<int> ret;
    for (size_t i = 0; i < file_descriptions.size(); i++)
        ret.push_back(i);
    return ret;
}

/////////////////////////////////////////////////////////////////
// Computation::FilterNonparsable()
//
// Filter a vector of units, removing any units whose supplied
// structures are not parsable.
/////////////////////////////////////////////////////////////////

std::vector<int> Computation::FilterNonparsable(const std::vector<int> &units,
						const std::vector<double> &values)
{
    Assert(IsMasterNode(), "Routine should only be called by master process.");
    
    std::vector<WorkUnit *> work_units;
    for (size_t i = 0; i < units.size(); i++)
        work_units.push_back(new ProcessingUnit(CHECK_PARSABILITY, 
                                                units[i], file_descriptions[units[i]].size, 0, toggle_complementary_only, 
                                                toggle_use_constraints, toggle_partition, 
                                                toggle_viterbi, false, posterior_cutoff));
    
    std::vector<double> parsable;
    RunMasterNode(parsable, work_units, values);
    
    for (size_t i = 0; i < work_units.size(); i++)
        delete work_units[i];
    
    std::vector<int> ret;  
    for (size_t i = 0; i < units.size(); i++)
    {
        Assert(units[i] >= 0 && units[i] < int(parsable.size()), "Out-of-bounds index.");
        if (parsable[units[i]])
        {
            ret.push_back(units[i]);
        }
        else
        {
            std::cerr << "No valid parse for file: " << file_descriptions[units[i]].input_filename << std::endl;
        }      
    }
    
    return ret;
}

/////////////////////////////////////////////////////////////////
// Computation::PredictStructure()
//
// Run prediction algorithm on each of the work units.
/////////////////////////////////////////////////////////////////

void Computation::PredictStructure(const std::vector<int> &units,
                                   const std::vector<double> &values, 
                                   double gammar,
                                   bool toggle_exact)
{
    Assert(IsMasterNode(), "Routine should only be called by master process.");
    
    std::vector<WorkUnit *> work_units;
    for (size_t i = 0; i < units.size(); i++)
        work_units.push_back(new ProcessingUnit(PREDICT_STRUCTURE,
                                                units[i], file_descriptions[units[i]].size, gammar, toggle_complementary_only, 
                                                toggle_use_constraints, toggle_partition, 
                                                toggle_viterbi, toggle_exact, posterior_cutoff));
    
    std::vector<double> unused;
    RunMasterNode(unused, work_units, values);
    
    for (size_t i = 0; i < work_units.size(); i++)
        delete work_units[i];
}

/////////////////////////////////////////////////////////////////
// Computation::ComputeFunction()
//
// Compute negative log-likelihood of the model over a fixed set
// of work units using a particular setting of the parameters.
/////////////////////////////////////////////////////////////////

double Computation::ComputeFunction(const std::vector<int> &units,
                                    const std::vector<double> &values,
                                    bool toggle_exact)
{
    Assert(IsMasterNode(), "Routine should only be called by master process.");
    
    if (cached_units != units ||
        cached_values != values ||
        cached_function.size() == 0)
    {
        
        std::vector<WorkUnit *> work_units;
        for (size_t i = 0; i < units.size(); i++)
            work_units.push_back(new ProcessingUnit(COMPUTE_FUNCTION,
                                                    units[i], file_descriptions[units[i]].size, 0, toggle_complementary_only, 
                                                    toggle_use_constraints, toggle_partition, 
                                                    toggle_viterbi, toggle_exact, posterior_cutoff));
        
        RunMasterNode(cached_function, work_units, values);
        
        cached_units = units;
        cached_values = values;
        cached_v.clear();
        cached_gradient.clear();
        cached_Hv.clear();    
        
        for (size_t i = 0; i < work_units.size(); i++)
            delete work_units[i];
    }
    
    return cached_function[0];
}

/////////////////////////////////////////////////////////////////
// Computation::ComputeGradient()
//
// Compute gradient of the negative log-likelihood of the model 
// over a fixed set of work units using a particular setting of 
// the parameters.
/////////////////////////////////////////////////////////////////

std::vector<double> Computation::ComputeGradient(const std::vector<int> &units,
                                                 const std::vector<double> &values,
                                                 bool toggle_exact)
{
    Assert(IsMasterNode(), "Routine should only be called by master process.");
    
    if (cached_units != units ||
        cached_values != values ||
        cached_function.size() == 0 ||
        cached_gradient.size() == 0)
    {
        
        std::vector<WorkUnit *> work_units;
        for (size_t i = 0; i < units.size(); i++)
            work_units.push_back(new ProcessingUnit(COMPUTE_GRADIENT,
                                                    units[i], file_descriptions[units[i]].size, 0, toggle_complementary_only, 
                                                    toggle_use_constraints, toggle_partition, 
                                                    toggle_viterbi, toggle_exact, posterior_cutoff));
        
        RunMasterNode(cached_gradient, work_units, values);
        
        cached_units = units;
        cached_values = values;
        cached_v.clear();
        cached_function.clear();
        cached_function.push_back(cached_gradient.back());
        cached_gradient.pop_back();
        cached_Hv.clear();    
        
        for (size_t i = 0; i < work_units.size(); i++)
            delete work_units[i];
    }
    
    return cached_gradient;
}

/////////////////////////////////////////////////////////////////
// Computation::ComputeHv()
//
// Compute product of the Hessian with an arbitrary vector v.
/////////////////////////////////////////////////////////////////

std::vector<double> Computation::ComputeHv(const std::vector<int> &units,
                                           const std::vector<double> &values,
                                           const std::vector<double> &v)
{
    Assert(IsMasterNode(), "Routine should only be called by master process.");
    
    if (cached_units != units ||
        cached_values != values ||
        cached_v != v ||
        cached_Hv.size() == 0)
    {
        
        std::vector<WorkUnit *> work_units;
        for (size_t i = 0; i < units.size(); i++)
            work_units.push_back(new ProcessingUnit(COMPUTE_HV,
                                                    units[i], file_descriptions[units[i]].size, 0, toggle_complementary_only, 
                                                    toggle_use_constraints, toggle_partition, 
                                                    toggle_viterbi, true, posterior_cutoff));
        
        std::vector<double> new_values = values;
        new_values.insert(new_values.end(), v.begin(), v.end());
        RunMasterNode(cached_Hv, work_units, new_values);
        
        cached_units = units;
        cached_values = values;
        cached_v = v;
        cached_function.clear();
        cached_gradient.clear();
        
        for (size_t i = 0; i < work_units.size(); i++)
            delete work_units[i];
    }
    
    return cached_Hv;
}

/////////////////////////////////////////////////////////////////
// Computation::SanityCheckGradient()
//
// Perform sanity check for the gradient computation.
/////////////////////////////////////////////////////////////////

void Computation::SanityCheckGradient(const std::vector<int> &units,
                                      const std::vector<double> &x)
{
    const int NUM_PER_GROUP = 5;
    const int ATTEMPTS = 8;
    
    std::cerr << "Starting gradient sanity check..." << std::endl;
    
    std::vector<double> g = ComputeGradient(units, x, true);
    double f = ComputeFunction(units, x, true);
    std::vector<double> xp = x;
    
    const std::vector<HyperparameterGroup> &groups = temp_params.GetHyperparameterGroups();
    for (size_t k = 0; k < groups.size(); k++)
    {
        int num_left = NUM_PER_GROUP;
        
        // perform sanity check for a group of parameters
        
        std::cerr << "Performing sanity check for parameter group: " << groups[k].name 
                  << " (indices " << groups[k].begin << " to " << groups[k].end << ", limit " << num_left << ")" << std::endl;
        
        for (int i = groups[k].begin; num_left && i <= groups[k].end; i++)
        {
            // perform sanity check for a single parameter
            
            std::vector<double> gp(ATTEMPTS);
            for (int j = 0; j < ATTEMPTS; j++)
            {
                double EPSILON = Pow(10.0, double(-j));
                xp[i] += EPSILON;
                gp[j] = (ComputeFunction(units, xp, true) - f) / EPSILON;
                xp[i] = x[i];
                
                if (g[i] == gp[j]) break;
                if (Abs(g[i] - gp[j]) / (Abs(g[i]) + Abs(gp[j])) < 1e-5) break;
            }
            
            // print results of sanity check
            
            if (g[i] != 0 || g[i] != gp[0])
            {
                std::cerr << std::setw(13) << i << std::setw(13) << g[i];
                for (int j = 0; j < ATTEMPTS; j++)
                    std::cerr << std::setw(13) << gp[j];
                std::cerr << std::endl;
                num_left--;
            }
        }
    }
    
    std::cerr << "Gradient sanity check complete." << std::endl;
}

/////////////////////////////////////////////////////////////////
// UnitComparator::UnitComparator()
// UnitComparator::operator()
//
// Sort work units in order of decreasing size.
/////////////////////////////////////////////////////////////////

UnitComparator::UnitComparator(Computation &computation) :
    computation(computation)
{}

bool UnitComparator::operator()(int i, int j) const 
{
    Assert(0 <= i && i < int(computation.file_descriptions.size()), "Out-of-bounds.");
    Assert(0 <= j && j < int(computation.file_descriptions.size()), "Out-of-bounds.");
    return (computation.file_descriptions[i].size > computation.file_descriptions[j].size);
}

/////////////////////////////////////////////////////////////////
// Computation::DoComputation()
//
// Decide what type of computation needs to be done and then
// pass the work on to the appropriate routine.
/////////////////////////////////////////////////////////////////

void Computation::DoComputation(std::vector<double> &ret, 
                                const WorkUnit *unit,
                                const std::vector<double> &values)
{
    const ProcessingUnit &_unit = *reinterpret_cast<const ProcessingUnit *>(unit);
    
    switch (_unit.command)
    {
        case CHECK_PARSABILITY:
            CheckParsability(_unit, ret, values);
            break;
        case COMPUTE_FUNCTION:
            if (_unit.toggle_exact)
                ComputeFunction<double>(_unit, ret, values);
            else
                ComputeFunction<Real>(_unit, ret, values);
            break;    
        case COMPUTE_GRADIENT:
            if (_unit.toggle_exact)
                ComputeGradient<double>(_unit, ret, values);
            else
                ComputeGradient<Real>(_unit, ret, values);
            break;    
        case COMPUTE_HV:
            ComputeHv(_unit, ret, values);
            break;    
        case PREDICT_STRUCTURE:
            if (_unit.toggle_exact)
                PredictStructure<double>(_unit, values);
            else
                PredictStructure<Real>(_unit, values);
            break;
        default: 
            Assert(false, "Unknown command type.");
    }
    
    if (_unit.command == COMPUTE_FUNCTION && ret[0] < 0)
    {
        std::cerr << "Bad function value (" << ret[0] << ") detected: ";
        std::cerr << _unit.GetSummary() << std::endl;
        
        Parameters params;
        params.WriteToFile("badparams", values);
        std::cerr << "BREAKING" << std::endl;
        exit(0);
    }
}

/////////////////////////////////////////////////////////////////
// Computation::CheckParsability()
//
// Check to see if a sequence is parsable or not.  Return a
// vector with a "1" in the appropriate spot indicating that a
// file is not parsable.
/////////////////////////////////////////////////////////////////

void Computation::CheckParsability(const ProcessingUnit &unit,
                                   std::vector<double> &ret, 
                                   const std::vector<double> &values)
{
    SStruct sstruct(file_descriptions[unit.index].input_filename);
    
    // conditional score
    
    InferenceEngine<Real> engine(sstruct.GetSequence(), unit.toggle_complementary_only);
    engine.LoadParameters(ConvertVector<Real>(values));
    engine.UseMapping(sstruct.GetMapping());
    engine.ComputeViterbi();
    Real conditional_score = engine.GetViterbiScore();
    
    // check for bad parse
    
    ret.clear();
    ret.resize(file_descriptions.size());
    ret[unit.index] = 1;
    
    if (conditional_score < Real(NEG_INF/2))
        ret[unit.index] = 0;
}

/////////////////////////////////////////////////////////////////
// Computation::ComputeFunction()
//
// Return a vector containing a single entry with the function
// value.
/////////////////////////////////////////////////////////////////

template<class T>
void Computation::ComputeFunction(const ProcessingUnit &unit,
                                  std::vector<double> &ret, 
                                  const std::vector<double> &values)
{
    SStruct sstruct(file_descriptions[unit.index].input_filename);
    std::vector<T> loss_weights(sstruct.GetLength() + 1, T(HAMMING_LOSS));
    
    if (unit.toggle_viterbi)
    {
        // unconditional score
        
        InferenceEngine<T> engine(sstruct.GetSequence(), unit.toggle_complementary_only);
        engine.LoadParameters(ConvertVector<T>(values));
        engine.UseLoss(sstruct.GetMapping(), loss_weights);
        engine.ComputeViterbi();
        T unconditional_score = engine.GetViterbiScore();
        
        // conditional score
        
        engine.UseMapping(sstruct.GetMapping());
        engine.ComputeViterbi();
        T conditional_score = engine.GetViterbiScore();
        
        // compute function value
        
        Assert(conditional_score <= unconditional_score, "Conditional score cannot exceed unconditional score.");
        ret.clear();
        ret.push_back(double(unconditional_score - conditional_score) * file_descriptions[unit.index].weight);
        
        // check for bad parse
        
        if (conditional_score < T(NEG_INF/2))
        {
            std::cerr << "Unexpected bad parse for file: " << file_descriptions[unit.index].input_filename << std::endl;
            ret.back() = 0;
            return;
        }
        
    }
    else
    {
        
        // unconditional score
        
        InferenceEngine<T> engine(sstruct.GetSequence(), unit.toggle_complementary_only);
        engine.LoadParameters(ConvertVector<T>(values));
        engine.UseLoss(sstruct.GetMapping(), loss_weights);
        engine.ComputeInside();
        T unconditional_score = engine.ComputePartitionCoefficient();
        
        // conditional score
        
        engine.UseMapping(sstruct.GetMapping());
        engine.ComputeInside();
        T conditional_score = engine.ComputePartitionCoefficient();
        
        // compute function value
        
        Assert(conditional_score <= unconditional_score, "Conditional score cannot exceed unconditional score.");
        ret.clear();
        ret.push_back(double(unconditional_score - conditional_score) * file_descriptions[unit.index].weight);
        
        // check for bad parse
        
        if (conditional_score < T(NEG_INF/2))
        {
            std::cerr << "Unexpected bad parse for file: " << file_descriptions[unit.index].input_filename << std::endl;
            fill(ret.begin(), ret.end(), 0);
            return;
        }
        
        // avoid precision problems
        
        if (ret.back() < 0) ret.back() = 0;
    }
}

/////////////////////////////////////////////////////////////////
// Computation::ComputeGradient()
//
// Return a vector containing the gradient and function value.
/////////////////////////////////////////////////////////////////

template<class T>
void Computation::ComputeGradient(const ProcessingUnit &unit,
                                  std::vector<double> &ret, 
                                  const std::vector<double> &values)
{
    SStruct sstruct(file_descriptions[unit.index].input_filename);
    std::vector<T> loss_weights(sstruct.GetLength() + 1, T(HAMMING_LOSS));
    
    if (unit.toggle_viterbi)
    {
        // unconditional score
        
        InferenceEngine<T> engine(sstruct.GetSequence(), unit.toggle_complementary_only);
        engine.LoadParameters(ConvertVector<T>(values));
        engine.UseLoss(sstruct.GetMapping(), loss_weights);
        engine.ComputeViterbi();
        T unconditional_score = engine.GetViterbiScore();
        std::vector<T> unconditional_counts = engine.ComputeViterbiFeatureCounts();
        std::vector<int> unconditional_mapping = engine.PredictPairingsViterbi();
        
        // conditional score
        
        engine.UseMapping(sstruct.GetMapping());
        engine.ComputeViterbi();
        T conditional_score = engine.GetViterbiScore();
        std::vector<T> conditional_counts = engine.ComputeViterbiFeatureCounts();
        std::vector<int> conditional_mapping = engine.PredictPairingsViterbi();
        
        // compute subgradient
        
        ret = ConvertVector<double>(unconditional_counts - conditional_counts) * file_descriptions[unit.index].weight;
        
        // compute function value
        
        Assert(conditional_score <= unconditional_score, "Conditional score cannot exceed unconditional score.");
        ret.push_back(double(unconditional_score - conditional_score) * file_descriptions[unit.index].weight);
        
        // check for bad parse
        
        if (conditional_score < T(NEG_INF/2))
        {
            std::cerr << "Unexpected bad parse for file: " << file_descriptions[unit.index].input_filename << std::endl;
            fill(ret.begin(), ret.end(), 0);
            return;
        }
        
        /*
        // compute weighted Hamming loss
        
        ret.push_back (0);
        for (size_t i = 1; i < conditional_mapping.size(); i++)
        if (conditional_mapping[i] != unconditional_mapping[i])
        ret.back() += loss_weights[i];
        ret.back() *= work_unit->weight;
        */
        
    }
    else
    {
        // unconditional score
        
        InferenceEngine<T> engine(sstruct.GetSequence(), unit.toggle_complementary_only);
        engine.LoadParameters(ConvertVector<T>(values));
        engine.UseLoss(sstruct.GetMapping(), loss_weights);
        engine.ComputeInside();
        engine.ComputeOutside();
        T unconditional_score = engine.ComputePartitionCoefficient();
        std::vector<T> unconditional_counts = engine.ComputeFeatureCountExpectations();
        
        // conditional score
        
        engine.UseMapping(sstruct.GetMapping());
        engine.ComputeInside();
        engine.ComputeOutside();
        T conditional_score = engine.ComputePartitionCoefficient();
        std::vector<T> conditional_counts = engine.ComputeFeatureCountExpectations();
        
        /*
          {
          Parameters params;
          std::vector<std::string> names = params.GetNames();
          std::vector<double> u = ConvertVector<double>(unconditional_counts);
          std::vector<double> c = ConvertVector<double>(conditional_counts);
          for (size_t i = 0; i < names.size(); i++) {
          std::cout << names[i] << "\t" << u[i] << "\t" << c[i] << std::endl;
          }
          exit(0);
          }
        */
        
        // compute gradient
        
        ret = ConvertVector<double>(unconditional_counts - conditional_counts) * file_descriptions[unit.index].weight;
        
        // compute function value
        
        Assert(conditional_score <= unconditional_score, "Conditional score cannot exceed unconditional score.");
        ret.push_back(double(unconditional_score - conditional_score) * file_descriptions[unit.index].weight);
        
        // check for bad parse
        
        if (conditional_score < T(NEG_INF/2))
        {
            std::cerr << "Unexpected bad parse for file: " << file_descriptions[unit.index].input_filename << std::endl;
            fill(ret.begin(), ret.end(), 0);
            return;
        }
    }
}

/////////////////////////////////////////////////////////////////
// Computation::ComputeHv()
//
// Return a vector containing Hv.
/////////////////////////////////////////////////////////////////

void Computation::ComputeHv(const ProcessingUnit &unit,
                            std::vector<double> &ret, 
                            const std::vector<double> &raw_values)
{
    std::vector<double> values(raw_values.begin(), raw_values.begin() + raw_values.size()/2);
    std::vector<double> v(raw_values.begin() + raw_values.size()/2, raw_values.end());
    
    if (unit.toggle_viterbi)
    {
        Error ("No generalization of Hv for Viterbi parsing exists yet.");
    }
    else
    {
        const double EPSILON = 1e-8;
        std::vector<double> ret2;
        ComputeGradient<double>(unit, ret, values + EPSILON * v);
        ComputeGradient<double>(unit, ret2, values - EPSILON * v);
        ret = (ret - ret2) / (2.0 * EPSILON);
    }
}

/////////////////////////////////////////////////////////////////
// Computation::PredictStructure()
//
// Predict structure of a single sequence.
/////////////////////////////////////////////////////////////////

template<class T>
void Computation::PredictStructure(const ProcessingUnit &unit,
                                   const std::vector<double> &values)
{
    // load sequence, with constraints if necessary
    
    SStruct sstruct(file_descriptions[unit.index].input_filename);
    InferenceEngine<T> engine(sstruct.GetSequence(), unit.toggle_complementary_only);
    if (unit.toggle_use_constraints) engine.UseMapping(sstruct.GetMapping());
    engine.LoadParameters(ConvertVector<T>(values));
    
    if (unit.toggle_partition)
    {
        engine.ComputeInside();
        T part_coeff = engine.ComputePartitionCoefficient();
        std::cout << "Log partition coefficient for \"" << file_descriptions[unit.index].input_filename << "\": " 
                  << part_coeff << std::endl;
        return;
    }
    
    SStruct *solution = NULL;
    
    // perform computation
    
    if (unit.toggle_viterbi)
    {
        engine.ComputeViterbi();
        solution = new SStruct(sstruct.GetSequence(), 
                               engine.PredictPairingsViterbi(), 
                               sstruct.GetName());
    }
    else
    {
        engine.ComputeInside();
        engine.ComputeOutside();
        engine.ComputePosterior();
        solution = new SStruct(sstruct.GetSequence(), 
                               engine.PredictPairingsPosterior(unit.gammar),
                               sstruct.GetName());
    }
    
    // write output

    if (file_descriptions[unit.index].output_parens_filename != "")
    {
        std::ofstream outfile(file_descriptions[unit.index].output_parens_filename.c_str());
        if (outfile.fail()) Error("Unable to open output parens file for writing.");
        solution->WriteParens(outfile, unit.toggle_complementary_only);
        outfile.close();
    }
  
    if (file_descriptions[unit.index].output_bpseq_filename != "")
    {
        std::ofstream outfile(file_descriptions[unit.index].output_bpseq_filename.c_str());
        if (outfile.fail()) Error("Unable to open output bpseq file for writing.");
        solution->WriteBPSEQ(outfile, unit.toggle_complementary_only);
        outfile.close();
    }
    
    if (file_descriptions[unit.index].output_posteriors_filename != "")
    {
        Assert(false, "Not yet implemented.");
        T *posterior = engine.GetPosterior(unit.posterior_cutoff);
        SparseMatrix<T> sparse(posterior, sstruct.GetLength()+1, T(0));
        delete [] posterior;
        std::ofstream outfile(file_descriptions[unit.index].output_posteriors_filename.c_str());
        if (outfile.fail()) Error("Unable to open output posteriors file for writing.");
        sparse.PrintSparseBPSEQ(outfile, sstruct.GetSequence());
        outfile.close();
    }
    
    if (file_descriptions[unit.index].output_parens_filename == "" &&
        file_descriptions[unit.index].output_bpseq_filename == "" &&
        file_descriptions[unit.index].output_posteriors_filename == "")
    {
        WriteProgressMessage("");
        solution->WriteParens(std::cout, unit.toggle_complementary_only);
    }
    
    delete solution;
}
