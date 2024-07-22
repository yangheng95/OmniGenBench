//////////////////////////////////////////////////////////////////////
// ComputationEngine.cpp
//////////////////////////////////////////////////////////////////////

#include "ComputationEngine.hpp"

//////////////////////////////////////////////////////////////////////
// ComputationEngine::ComputationEngine()
// ComputationEngine::~ComputationEngine()
//
// Constructor and destructor.
//////////////////////////////////////////////////////////////////////

template<class RealT>
ComputationEngine<RealT>::ComputationEngine(const Options &options,
                                            const std::vector<FileDescription> &descriptions,
                                            InferenceEngine<RealT> &inference_engine,
                                            ParameterManager<RealT> &parameter_manager) :
    DistributedComputation<RealT, SharedInfo<RealT>, NonSharedInfo>(options.GetBoolValue("verbose_output")),
    options(options),
    descriptions(descriptions),
    inference_engine(inference_engine),
    parameter_manager(parameter_manager)
{}

template<class RealT>
ComputationEngine<RealT>::~ComputationEngine()
{}

//////////////////////////////////////////////////////////////////////
// ComputationEngine::DoComputation()
//
// Decide what type of computation needs to be done and then
// pass the work on to the appropriate routine.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void ComputationEngine<RealT>::DoComputation(std::vector<RealT> &result, 
                                             const SharedInfo<RealT> &shared,
                                             const NonSharedInfo &nonshared)
{
    switch (shared.command)
    {
        case CHECK_PARSABILITY:
            CheckParsability(result, nonshared);
            break;
        case COMPUTE_SOLUTION_NORM_BOUND:
            ComputeSolutionNormBound(result, shared, nonshared);
            break;
        case COMPUTE_GRADIENT_NORM_BOUND:
            ComputeGradientNormBound(result, nonshared);
            break;
        case COMPUTE_LOSS:
            ComputeLoss(result, shared, nonshared);
            break;    
        case COMPUTE_FUNCTION:
            ComputeFunctionAndGradient(result, shared, nonshared, false);
            break;    
        case COMPUTE_GRADIENT:
            ComputeFunctionAndGradient(result, shared, nonshared, true);
            break;    
        case COMPUTE_HV:
            ComputeHessianVectorProduct(result, shared, nonshared);
            break;    
        case PREDICT:
            Predict(result, shared, nonshared);
            break;
        default: 
            Assert(false, "Unknown command type.");
            break;
    }
}

//////////////////////////////////////////////////////////////////////
// ComputationEngine::CheckParsability()
//
// Check to see if a sequence is parsable or not.  Return a
// vector with a "0" in the appropriate spot indicating that a
// file is not parsable.
//////////////////////////////////////////////////////////////////////

template <class RealT>
void ComputationEngine<RealT>::CheckParsability(std::vector<RealT> &result, 
                                                const NonSharedInfo &nonshared)
{
    // load training example
    const SStruct &sstruct = descriptions[nonshared.index].sstruct;
    inference_engine.LoadSequence(sstruct);

    // conditional inference
    inference_engine.LoadValues(std::vector<RealT>(parameter_manager.GetNumLogicalParameters()));
    inference_engine.UseConstraints(sstruct.GetMapping());
    inference_engine.ComputeViterbi();
    RealT conditional_score = inference_engine.GetViterbiScore();

    // check for bad parse
    result.clear();
    result.resize(descriptions.size());
    result[nonshared.index] = (conditional_score < RealT(NEG_INF/2) ? 0 : 1);
}

//////////////////////////////////////////////////////////////////////
// ComputationEngine::ComputeSolutionNormBound()
//
// Compute the max entropy and loss possible for an example.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void ComputationEngine<RealT>::ComputeSolutionNormBound(std::vector<RealT> &result, 
                                                        const SharedInfo<RealT> &shared,
                                                        const NonSharedInfo &nonshared)
{
    RealT max_entropy = RealT(0);
    RealT max_loss = RealT(0);

    // load training example
    const SStruct &sstruct = descriptions[nonshared.index].sstruct;
    inference_engine.LoadSequence(sstruct);

    // load parameters
    const std::vector<RealT> w(parameter_manager.GetNumLogicalParameters(), RealT(0));
    inference_engine.LoadValues(w);

    // perform computation
#if !SMOOTH_MAX_MARGIN
    if (!options.GetBoolValue("viterbi_parsing"))
#endif
    {
        inference_engine.ComputeInside();
        max_entropy += inference_engine.ComputeLogPartitionCoefficient();
    }
        
#if defined(HAMMING_LOSS)
    inference_engine.UseLoss(sstruct.GetMapping(), RealT(HAMMING_LOSS));
    inference_engine.ComputeViterbi();
    max_loss += inference_engine.GetViterbiScore();
#endif

    result.clear();
    result.resize(descriptions.size());
    result[nonshared.index] = max_entropy / shared.log_base + max_loss;

    result *= RealT(descriptions[nonshared.index].weight);
}

//////////////////////////////////////////////////////////////////////
// ComputationEngine::ComputeGradientNormBound()
//
// Compute the max L1 norm for the features of an example.
// Return a vector with this value in the appropriate spot for
// this example.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void ComputationEngine<RealT>::ComputeGradientNormBound(std::vector<RealT> &result,
                                                        const NonSharedInfo &nonshared)
{
    // load training example
    const SStruct &sstruct = descriptions[nonshared.index].sstruct;
    inference_engine.LoadSequence(sstruct);

    // load parameters
    const std::vector<RealT> w(parameter_manager.GetNumLogicalParameters(), RealT(1));
    inference_engine.LoadValues(w);

    // perform inference
    inference_engine.ComputeViterbi();
    const RealT max_L1_norm = inference_engine.GetViterbiScore();

    result.clear();
    result.resize(descriptions.size());
    result[nonshared.index] = max_L1_norm;
}

//////////////////////////////////////////////////////////////////////
// ComputationEngine::ComputeLoss()
//
// Return a vector containing a single entry with the loss value.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void ComputationEngine<RealT>::ComputeLoss(std::vector<RealT> &result, 
                                           const SharedInfo<RealT> &shared,
                                           const NonSharedInfo &nonshared)
{
    // load training example
    const SStruct &sstruct = descriptions[nonshared.index].sstruct;
    inference_engine.LoadSequence(sstruct);

    // load parameters
    const std::vector<RealT> w(shared.w, shared.w + parameter_manager.GetNumLogicalParameters());
    inference_engine.LoadValues(w * shared.log_base);

    // perform inference
    SStruct *solution;
    if (options.GetBoolValue("viterbi_parsing"))
    {
        inference_engine.ComputeViterbi();
        solution = new SStruct(sstruct);
        solution->SetMapping(inference_engine.PredictPairingsViterbi());
    }
    else
    {
        inference_engine.ComputeInside();
        inference_engine.ComputeOutside();
        inference_engine.ComputePosterior();
        solution = new SStruct(sstruct);
        solution->SetMapping(inference_engine.PredictPairingsPosterior(shared.gamma));
    }

    // compute loss
    if (!shared.use_loss) Error("Must be using loss function in order to compute loss.");
#if defined(HAMMING_LOSS)
    inference_engine.UseLoss(sstruct.GetMapping(), shared.log_base * RealT(HAMMING_LOSS));
#endif
    inference_engine.LoadValues(std::vector<RealT>(w.size()));
    inference_engine.UseConstraints(solution->GetMapping());
    inference_engine.ComputeViterbi();

    delete solution;

    result.clear();
    result.push_back(inference_engine.GetViterbiScore());

    result *= RealT(descriptions[nonshared.index].weight);
    result.back() /= shared.log_base;
}

//////////////////////////////////////////////////////////////////////
// ComputationEngine::ComputeFunctionAndGradient();
//
// Return a vector containing the gradient and function value.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void ComputationEngine<RealT>::ComputeFunctionAndGradient(std::vector<RealT> &result, 
                                                          const SharedInfo<RealT> &shared,
                                                          const NonSharedInfo &nonshared,
                                                          bool need_gradient)
{
    // load training example
    const SStruct &sstruct = descriptions[nonshared.index].sstruct;
    inference_engine.LoadSequence(sstruct);

    // load parameters
    const std::vector<RealT> w(shared.w, shared.w + parameter_manager.GetNumLogicalParameters());
    inference_engine.LoadValues(w * shared.log_base);
#if defined(HAMMING_LOSS)
    if (shared.use_loss) inference_engine.UseLoss(sstruct.GetMapping(), shared.log_base * RealT(HAMMING_LOSS));
#endif
    
    // unconditional inference
    RealT unconditional_score;
    std::vector<RealT> unconditional_counts;

    if (shared.use_nonsmooth)
    {
        inference_engine.ComputeViterbi();
        unconditional_score = inference_engine.GetViterbiScore();
        if (need_gradient) unconditional_counts = inference_engine.ComputeViterbiFeatureCounts();
    }
    else
    {
        inference_engine.ComputeInside();
        unconditional_score = inference_engine.ComputeLogPartitionCoefficient();
        if (need_gradient)
        {
            inference_engine.ComputeOutside();
            unconditional_counts = inference_engine.ComputeFeatureCountExpectations();
        }
    }

    // conditional inference
    RealT conditional_score;
    std::vector<RealT> conditional_counts;

    inference_engine.UseConstraints(sstruct.GetMapping());
    if (shared.use_nonsmooth)
    {
        inference_engine.ComputeViterbi();
        conditional_score = inference_engine.GetViterbiScore();
        if (need_gradient) conditional_counts = inference_engine.ComputeViterbiFeatureCounts();
    }
    else
    {
        inference_engine.ComputeInside();
        conditional_score = inference_engine.ComputeLogPartitionCoefficient();
        if (need_gradient)
        {
            inference_engine.ComputeOutside();
            conditional_counts = inference_engine.ComputeFeatureCountExpectations();
        }
    }

    result.clear();

    // compute subgradient
    if (need_gradient) result = unconditional_counts - conditional_counts;
    
    // compute function value
    Assert(conditional_score <= unconditional_score, "Conditional score cannot exceed unconditional score.");
    result.push_back(unconditional_score - conditional_score);
    
    // check for bad parse
    if (conditional_score < RealT(NEG_INF/2))
    {
        std::cerr << "Unexpected bad parse for file: " << descriptions[nonshared.index].input_filename << std::endl;
        fill(result.begin(), result.end(), RealT(0));
        return;
    }

    if (NONCONVEX_MULTIPLIER != 0)
    {
        
#if STOCHASTIC_GRADIENT
        if (shared.use_loss) inference_engine.UseLoss(sstruct.GetMapping(), RealT(0));
        
        // unconditional counts
        inference_engine.UseMapping(std::vector<int>(sstruct.GetLength() + 1, UNKNOWN));
        if (shared.use_nonsmooth)
        {
            inference_engine.ComputeViterbi();
            unconditional_score = inference_engine.GetViterbiScore();
            if (need_gradient) unconditional_counts = inference_engine.ComputeViterbiFeatureCounts();
        }
        else
        {
            inference_engine.ComputeInside();
            unconditional_score = inference_engine.ComputeLogPartitionCoefficient();
            if (need_gradient)
            {
                inference_engine.ComputeOutside();
                unconditional_counts = inference_engine.ComputeFeatureCountExpectations();
            }
        }
        
        // conditional counts
        inference_engine.UseMapping(sstruct.GetMapping());
        if (shared.use_nonsmooth)
        {
            inference_engine.ComputeViterbi();
            unconditional_score = inference_engine.GetViterbiScore();
            if (need_gradient) unconditional_counts = inference_engine.ComputeViterbiFeatureCounts();
        }
        else
        {
            inference_engine.ComputeInside();
            unconditional_score = inference_engine.ComputeLogPartitionCoefficient();
            if (need_gradient)
            {
                inference_engine.ComputeOutside();
                unconditional_counts = inference_engine.ComputeFeatureCountExpectations();
            }
        }
        
        std::vector<RealT> result2;
        
        // compute subgradient
        if (need_gradient) result2 = unconditional_counts - conditional_counts;
        
        // compute function value
        Assert(conditional_score <= unconditional_score, "Conditional score cannot exceed unconditional score.");
        result2.push_back(unconditional_score - conditional_score);
        
        // check for bad parse
        if (conditional_score < RealT(NEG_INF/2))
        {
            std::cerr << "Unexpected bad parse for file: " << descriptions[nonshared.index].input_filename << std::endl;
            fill(result.begin(), result.end(), 0);
            return;
        }
        
        result -= NONCONVEX_MULTIPLIER * result2;
#endif
    }

    // avoid precision problems
    if (result.back() < 0)
    {
        if (result.back() < -1e-6)
        {
            std::cerr << "Encountered negative function value for " << descriptions[nonshared.index].input_filename << ": " << result.back() << std::endl;
            parameter_manager.WriteToFile(SPrintF("neg_params.%s", GetBaseName(descriptions[nonshared.index].input_filename).c_str()), w);
            exit(0);
        }
        std::fill(result.begin(), result.end(), RealT(0));
        return;
    }

    result *= RealT(descriptions[nonshared.index].weight);
    result.back() /= shared.log_base;
}

//////////////////////////////////////////////////////////////////////
// ComputationEngine::ComputeHessianVectorProduct()
//
// Return a vector containing Hv.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void ComputationEngine<RealT>::ComputeHessianVectorProduct(std::vector<RealT> &result, 
                                                           const SharedInfo<RealT> &shared,
                                                           const NonSharedInfo &nonshared)
{
    const std::vector<RealT> w(shared.w, shared.w + parameter_manager.GetNumLogicalParameters());
    const std::vector<RealT> v(shared.v, shared.v + parameter_manager.GetNumLogicalParameters());

    if (options.GetBoolValue("viterbi_parsing"))
    {
        Error("Should not use Hessian-vector products with Viterbi parsing.");
    }
    
    const RealT EPSILON = RealT(1e-8);
    SharedInfo<RealT> shared_temp(shared);
    std::vector<RealT> result2;

    for (size_t i = 0; i < parameter_manager.GetNumLogicalParameters(); i++)
        shared_temp.w[i] = shared.w[i] + EPSILON * v[i];
    ComputeFunctionAndGradient(result, shared_temp, nonshared, true);
    
    for (size_t i = 0; i < parameter_manager.GetNumLogicalParameters(); i++)
        shared_temp.w[i] = shared.w[i] - EPSILON * v[i];
    ComputeFunctionAndGradient(result2, shared_temp, nonshared, true);
    
    result = (result - result2) / (RealT(2) * EPSILON);
}

//////////////////////////////////////////////////////////////////////
// ComputationEngine::Predict()
//
// Predict structure of a single sequence.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void ComputationEngine<RealT>::Predict(std::vector<RealT> &result, 
                                       const SharedInfo<RealT> &shared,
                                       const NonSharedInfo &nonshared)
{
    result.clear();
    
    // load sequence, with constraints if necessary
    const SStruct &sstruct = descriptions[nonshared.index].sstruct;
    inference_engine.LoadSequence(sstruct);
    if (options.GetBoolValue("use_constraints")) inference_engine.UseConstraints(sstruct.GetMapping());
    
    // load parameters
    const std::vector<RealT> w(shared.w, shared.w + parameter_manager.GetNumLogicalParameters());
    inference_engine.LoadValues(w * shared.log_base);

    // perform inference
    SStruct *solution;
    if (options.GetBoolValue("viterbi_parsing"))
    {
        inference_engine.ComputeViterbi();
        if (options.GetBoolValue("partition_function_only"))
        {
            std::cout << "Viterbi score for \"" << descriptions[nonshared.index].input_filename << "\": " 
                      << inference_engine.GetViterbiScore() << std::endl;
            return;
        }
        solution = new SStruct(sstruct);
        solution->SetMapping(inference_engine.PredictPairingsViterbi());
    }
    else
    {
        inference_engine.ComputeInside();
        if (options.GetBoolValue("partition_function_only"))
        {
            std::cout << "Log partition coefficient for \"" << descriptions[nonshared.index].input_filename << "\": " 
                      << inference_engine.ComputeLogPartitionCoefficient() << std::endl;
            return;
        }
        inference_engine.ComputeOutside();
        inference_engine.ComputePosterior();
        solution = new SStruct(sstruct);
        solution->SetMapping(inference_engine.PredictPairingsPosterior(shared.gamma));
    }

    // write output
    if (options.GetStringValue("output_parens_destination") != "")
    {
        const std::string filename = MakeOutputFilename(descriptions[nonshared.index].input_filename,
                                                        options.GetStringValue("output_parens_destination"),
                                                        options.GetRealValue("gamma") < 0,
                                                        shared.gamma);
        std::ofstream outfile(filename.c_str());
        if (outfile.fail()) Error("Unable to open output parens file '%s' for writing.", filename.c_str());
        solution->WriteParens(outfile);
        outfile.close();
    }
  
    if (options.GetStringValue("output_bpseq_destination") != "")
    {
        const std::string filename = MakeOutputFilename(descriptions[nonshared.index].input_filename,
                                                        options.GetStringValue("output_bpseq_destination"),
                                                        options.GetRealValue("gamma") < 0,
                                                        shared.gamma);
        std::ofstream outfile(filename.c_str());
        if (outfile.fail()) Error("Unable to open output bpseq file '%s' for writing.", filename.c_str());
        solution->WriteBPSEQ(outfile);
        outfile.close();
    }
    
    if (options.GetStringValue("output_posteriors_destination") != "")
    {
        const std::string filename = MakeOutputFilename(descriptions[nonshared.index].input_filename,
                                                        options.GetStringValue("output_posteriors_destination"),
                                                        options.GetRealValue("gamma") < 0,
                                                        shared.gamma);
        RealT *posterior = inference_engine.GetPosterior(options.GetRealValue("output_posteriors_cutoff"));
        SparseMatrix<RealT> sparse(posterior, sstruct.GetLength()+1, RealT(0));
        delete [] posterior;
        std::ofstream outfile(filename.c_str());
        if (outfile.fail()) Error("Unable to open output posteriors file '%s' for writing.", filename.c_str());
        sparse.PrintSparseBPSEQ(outfile, sstruct.GetSequences()[0]);
        outfile.close();
    }
    
    if (options.GetStringValue("output_parens_destination") == "" &&
        options.GetStringValue("output_bpseq_destination") == "" &&
        options.GetStringValue("output_posteriors_destination") == "")
    {
        WriteProgressMessage("");
        solution->WriteParens(std::cout);
    }
    
    delete solution;
}

//////////////////////////////////////////////////////////////////////
// ComputationEngine::MakeOutputFilename()
//
// Decide on output filename, if any.  The arguments to this function
// consist of (1) a boolean variable indicating whether the output
// destination should be treated as the name of an output directory
// (and the output filename is chosen to match the input file) or
// whether the output destination should be interpreted as the output
// filename; (2) the name of the input file to be processed; and (3)
// the supplied output destination.
//////////////////////////////////////////////////////////////////////

template<class RealT>
std::string ComputationEngine<RealT>::MakeOutputFilename(const std::string &input_filename,
                                                         const std::string &output_destination,
                                                         const bool cross_validation,
                                                         const RealT gamma) const 
{
    if (output_destination == "") return "";

    const std::string dir_name = GetDirName(output_destination);
    const std::string base_name = GetBaseName(output_destination);

    const std::string prefix = (dir_name != "" ? (dir_name + DIR_SEPARATOR_CHAR) : std::string(""));
    
    // check if output directory required
    if (descriptions.size() > 1)
    {
        if (cross_validation)
        {
            return SPrintF("%s%s%c%s.gamma=%lf%c%s",
                           prefix.c_str(),
                           base_name.c_str(),
                           DIR_SEPARATOR_CHAR,
                           base_name.c_str(),
                           double(gamma),
                           DIR_SEPARATOR_CHAR,
                           GetBaseName(input_filename).c_str());
        }
        return SPrintF("%s%s%c%s",
                       prefix.c_str(),
                       base_name.c_str(),
                       DIR_SEPARATOR_CHAR,
                       GetBaseName(input_filename).c_str());
    }
    
    if (cross_validation)
    {
        return SPrintF("%s%s%c%s.gamma=%lf",
                       prefix.c_str(),
                       base_name.c_str(),
                       DIR_SEPARATOR_CHAR,
                       base_name.c_str(),
                       double(gamma));
    }
    return SPrintF("%s%s",
                   prefix.c_str(),
                   base_name.c_str());
}
