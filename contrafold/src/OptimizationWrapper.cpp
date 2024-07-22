/////////////////////////////////////////////////////////////////
// OptimizationWrapper.cpp
/////////////////////////////////////////////////////////////////

#include "OptimizationWrapper.hpp"

/////////////////////////////////////////////////////////////////
// OptimizationWrapper::OptimizationWrapper()
//
// Constructor.
/////////////////////////////////////////////////////////////////

OptimizationWrapper::OptimizationWrapper(Computation &computation, 
                                         Parameters &params,
                                         const std::vector<double> &base_values) :
    computation(computation), params(params), base_values(base_values), indent(0)
{
    logfile.open("optimize.log");
    if (logfile.fail()) Error("Could not open log file for writing.");
}

/////////////////////////////////////////////////////////////////
// OptimizationWrapper::~OptimizationWrapper()
//
// Destructor.
/////////////////////////////////////////////////////////////////

OptimizationWrapper::~OptimizationWrapper()
{
  logfile.close();
}

/////////////////////////////////////////////////////////////////
// OptimizationWrapper::Indent()
// OptimizationWrapper::Unindent()
// OptimizationWrapper::PrintMessage()
//
// Print indented message.
/////////////////////////////////////////////////////////////////

void OptimizationWrapper::Indent() { indent++; }
void OptimizationWrapper::Unindent() { indent--; Assert(indent >= 0, "Cannot unindent!"); }
void OptimizationWrapper::PrintMessage(const std::string &s)
{
    for (int i = 0; i < indent; i++) std::cerr << "    ";
    for (int i = 0; i < indent; i++) logfile << "    ";
    std::cerr << s << std::endl;
    logfile << s << std::endl;
}

/////////////////////////////////////////////////////////////////
// OptimizationWrapper::TransformHyperparameters()
//
// Compute hyperparameter transformation.
/////////////////////////////////////////////////////////////////

std::vector<double> OptimizationWrapper::TransformHyperparameters(const std::vector<double> &C) {
    return Exp(C);
}

/////////////////////////////////////////////////////////////////
// OptimizationWrapper::TransformHyperparametersDerivative()
//
// Compute hyperparameter transformation.
/////////////////////////////////////////////////////////////////

std::vector<double> OptimizationWrapper::TransformHyperparametersDerivative(const std::vector<double> &C) {
    return Exp(C);
}

/////////////////////////////////////////////////////////////////
// OptimizationWrapper::Train()
//
// Run optimization algorithm with fixed regularization
// constants.
/////////////////////////////////////////////////////////////////

void OptimizationWrapper::Train(const std::vector<int> &units,
                                std::vector<double> &values,
                                const std::vector<double> &C,
                                bool toggle_viterbi)
{
    static std::vector<int> cached_units;
    static std::vector<double> cached_initial_values;
    static std::vector<double> cached_C;
    static std::vector<double> cached_learned_values;
    
    if (cached_units != units ||
        cached_initial_values != values ||
        cached_C != C)
    {
        cached_units = units;
        cached_initial_values = values;
        cached_C = C;
        cached_learned_values = values;
        
        WriteProgressMessage ("Starting training...");
        if (toggle_viterbi)
        {
            Error("Not yet implemented.");
        }
        else
        {
            InnerOptimizationWrapper optimizer(this, units, C);
            optimizer.Minimize(cached_learned_values);
        }
        
    }
    else
    {
        PrintMessage ("Using cached result from Train()...");
    }
    
  values = cached_learned_values;
}

/////////////////////////////////////////////////////////////////
// OptimizationWrapper::LearnHyperparameters()
//
// Use staged holdout cross-validation in order estimate
// regularization constants.
/////////////////////////////////////////////////////////////////

void OptimizationWrapper::LearnHyperparameters(std::vector<int> units,
                                               std::vector<double> &values,
                                               double holdout_ratio,
                                               bool toggle_viterbi)
{
    
    // split data into training and holdout sets
    
    std::sort(units.begin(), units.end(), UnitComparator(computation));
    //std::random_shuffle(units.begin(), units.end());
    std::vector<int> holdout(units.begin(), units.begin() + int(units.size() * holdout_ratio));
    std::vector<int> training(units.begin() + int(units.size() * holdout_ratio), units.end());
    
    if (training.size() == 0 || holdout.size() == 0) 
        Error("Not enough training samples for cross-validation.");
    
    // do hyperparameter optimization
    
    PrintMessage("Starting hyperparameter optimization...");
    Indent();
    
    PrintMessage("List of hyperparameters:");
    Indent();
    const std::vector<HyperparameterGroup> &groups = params.GetHyperparameterGroups();
    for (size_t i = 0; i < groups.size(); i++)
        PrintMessage(SPrintF("Hyperparameter group %d: %s", i+1, groups[i].name.c_str()));
    Unindent();
    
    std::vector<double> C = std::vector<double>(params.GetNumHyperparameterGroups(), 5);
    
    if (toggle_viterbi)
    {
        Error("Not yet implemented.");
    }
    else
    {
        OuterOptimizationWrapper optimizer(this, values, training, holdout);
        optimizer.Minimize(C);
    }
    
    Unindent();
    std::ostringstream oss;
    const std::vector<double> Cp = TransformHyperparameters(C);
    oss << "Chose hyperparameters, C = " << Cp;
    PrintMessage(oss.str());
    
    // Now, retrain on all data
    
    PrintMessage("Retraining on entire training set...");
    Indent();
    Train(units, values, Cp, toggle_viterbi);
    Unindent();
}
