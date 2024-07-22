//////////////////////////////////////////////////////////////////////
// OptimizationWrapper.ipp
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// OptimizationWrapper<RealT>::OptimizationWrapper()
//
// Constructor.
//////////////////////////////////////////////////////////////////////

template<class RealT>
OptimizationWrapper<RealT>::OptimizationWrapper(ComputationWrapper<RealT> &computation_wrapper) :
    computation_wrapper(computation_wrapper),
    indent(0)
{
    logfile.open("optimize.log");
    if (logfile.fail()) Error("Could not open log file for writing.");
}

//////////////////////////////////////////////////////////////////////
// OptimizationWrapper<RealT>::~OptimizationWrapper()
//
// Destructor.
//////////////////////////////////////////////////////////////////////

template<class RealT>
OptimizationWrapper<RealT>::~OptimizationWrapper()
{
    logfile.close();
}

//////////////////////////////////////////////////////////////////////
// OptimizationWrapper<RealT>::Indent()
// OptimizationWrapper<RealT>::Unindent()
// OptimizationWrapper<RealT>::PrintMessage()
//
// Print indented message.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void OptimizationWrapper<RealT>::Indent() { indent++; }

template<class RealT>
void OptimizationWrapper<RealT>::Unindent() { indent--; Assert(indent >= 0, "Cannot unindent!"); }

template<class RealT>
void OptimizationWrapper<RealT>::PrintMessage(const std::string &s)
{
    for (int i = 0; i < indent; i++) std::cerr << "    ";
    for (int i = 0; i < indent; i++) logfile << "    ";
    std::cerr << s << std::endl;
    logfile << s << std::endl;
}

//////////////////////////////////////////////////////////////////////
// OptimizationWrapper<RealT>::Train()
//
// Run optimization algorithm with fixed regularization
// constants.
//////////////////////////////////////////////////////////////////////

template<class RealT>
RealT OptimizationWrapper<RealT>::Train(const std::vector<int> &units,
                                        std::vector<RealT> &w,
                                        const std::vector<RealT> &C)
{
    static std::vector<int> cached_units;
    static std::vector<RealT> cached_initial_w;
    static std::vector<RealT> cached_C;
    static std::vector<RealT> cached_learned_w;
    static RealT cached_f;

    if (cached_units != units ||
        cached_initial_w != w ||
        cached_C != C)
    {
        cached_units = units;
        cached_initial_w = w;
        cached_C = C;
        cached_learned_w = w;
        
        WriteProgressMessage("Starting training...");

#if STOCHASTIC_GRADIENT
        Error("Not yet implemented.");
#else

        const std::vector<RealT> Ce = GetParameterManager().ExpandParameterGroupValues(C);
        const RealT log_base = RealT(GetOptions().GetRealValue("log_base"));
        
        if (GetOptions().GetBoolValue("viterbi_parsing"))
        {
            std::vector<RealT> bias(w.size());

            
            
#if SMOOTH_MAX_MARGIN
            InnerOptimizationWrapperLBFGS<RealT> inner_optimization_wrapper(this, units, Ce);
#else
#if BMRM_AVAILABLE
            InnerOptimizationWrapperBundleMethod<RealT> inner_optimization_wrapper(this, units, Ce);
#else
            InnerOptimizationWrapperSubgradientMethod<RealT> inner_optimization_wrapper(this, units, Ce);
            PrintMessage("BMRM not available, so defaulting to subgradient algorithm.");
#endif
#endif
                
            for (int i = 0; i < NUM_CCCP_STEPS; i++)
            {
                PrintMessage(SPrintF("Starting inner loop (pass %d)...", i));
                if (i > 0) bias = -RealT(NONCONVEX_MULTIPLIER) * computation_wrapper.ComputeGradient(units, cached_learned_w, true, false, log_base);
                std::cerr << bias << std::endl;
                inner_optimization_wrapper.LoadBias(bias);
                cached_f = inner_optimization_wrapper.Minimize(cached_learned_w);
                GetParameterManager().WriteToFile(SPrintF("optimize.params.stage%d", i+1), cached_learned_w);
                
                RealT loss = computation_wrapper.ComputeLoss(units, cached_learned_w, log_base);
                PrintMessage(SPrintF("Current loss: %lf", loss));
                if (RealT(NONCONVEX_MULTIPLIER) == RealT(0)) break;
            }

        }
        else
        {
            InnerOptimizationWrapperLBFGS<RealT> inner_optimization_wrapper(this, units, Ce);
            
            cached_f = inner_optimization_wrapper.Minimize(cached_learned_w);
        }

#endif
    }
    else
    {
        PrintMessage ("Using cached result from Train()...");
    }
    
    w = cached_learned_w;
    return cached_f;
}


//////////////////////////////////////////////////////////////////////
// OptimizationWrapper<RealT>::LearnHyperparameters()
//
// Use holdout cross validation in order to estimate
// regularization constants.
//////////////////////////////////////////////////////////////////////

#if HYPERPARAMETER_GRID_SEARCH
template<class RealT>
void OptimizationWrapper<RealT>::LearnHyperparameters(std::vector<int> units,
                                                      std::vector<RealT> &w)
{
    // split data into training and holdout sets
    //std::random_shuffle(units.begin(), units.end());
    
    const RealT holdout_ratio = GetOptions().GetRealValue("holdout_ratio");
    const std::vector<int> holdout(units.begin(), units.begin() + int(units.size() * holdout_ratio));
    const std::vector<int> training(units.begin() + int(units.size() * holdout_ratio), units.end());

    if (training.size() == 0 || holdout.size() == 0) 
        Error("Not enough training samples for cross-validation.");

    // do hyperparameter optimization
    PrintMessage("Starting hyperparameter optimization...");
    Indent();
    
    PrintMessage("List of hyperparameters:");
    Indent();
    const std::vector<ParameterGroup> &groups = GetParameterManager().GetParameterGroups();
    for (size_t i = 0; i < groups.size(); i++)
        PrintMessage(SPrintF("Parameter group %d: %s", i+1, groups[i].name.c_str()));
    Unindent();

    RealT best_C = 0, best_holdout_loss = 1e20;
    std::vector<RealT> C = std::vector<RealT>(GetParameterManager().GetNumParameterGroups());

    // perform cross-validation
    for (int k = -5; k <= 10; k++)
    {
        // perform training
        std::fill(C.begin(), C.end(), Pow(2.0, RealT(k)));
        PrintMessage(SPrintF("Performing optimization using C = %lf", C[0]));
        Indent();
        std::vector<RealT> x(w);
        const RealT f = Train(training, x, C);
        Unindent();

        // compute holdout loss
#if CROSS_VALIDATE_USING_LOGLOSS
        if (GetOptions().GetBoolValue("viterbi_parsing")) Error("Cannot use logloss for cross validation if Viterbi parsing.");
        RealT loss = computation_wrapper.ComputeFunction(holdout, x, false, false);
#else
        RealT loss = computation_wrapper.ComputeLoss(holdout, x, true);
#endif
        
        PrintMessage(SPrintF("Using C = %lf, regularized training loss = %lf, holdout loss = %lf", double(C[0]), double(f), double(loss)));
        
        if (loss < best_holdout_loss)
        {
            best_holdout_loss = loss;
            best_C = C[0];
        }
    }

    Unindent();
    PrintMessage(SPrintF("Chose C = %lf, holdout loss = %lf", best_C, best_holdout_loss));
    std::fill(C.begin(), C.end(), best_C / (1.0 - holdout_ratio));
    
    // now, retrain on all data
    PrintMessage("Retraining on entire training set...");
    Indent();
    Train(units, w, C);
    Unindent();
}
#endif

//////////////////////////////////////////////////////////////////////
// OptimizationWrapper<RealT>::LearnHyperparameters()
//
// Use gradient-based holdout cross-validation in order estimate
// regularization constants.
//////////////////////////////////////////////////////////////////////

#if HYPERPARAMETER_GRADIENT_OPTIMIZATION
template<class RealT>
void OptimizationWrapper<RealT>::LearnHyperparameters(std::vector<int> units,
                                                      std::vector<RealT> &w)
{
    // split data into training and holdout sets
    //std::random_shuffle(units.begin(), units.end());
    
    const RealT holdout_ratio = GetOptions().GetRealValue("holdout_ratio");
    const std::vector<int> holdout(units.begin(), units.begin() + int(units.size() * holdout_ratio));
    const std::vector<int> training(units.begin() + int(units.size() * holdout_ratio), units.end());

    if (training.size() == 0 || holdout.size() == 0) 
        Error("Not enough training samples for cross-validation.");

    // do hyperparameter optimization
    PrintMessage("Starting hyperparameter optimization...");
    Indent();
    
    PrintMessage("List of hyperparameters:");
    Indent();
    const std::vector<ParameterGroup> &groups = GetParameterManager().GetParameterGroups();
    for (size_t i = 0; i < groups.size(); i++)
        PrintMessage(SPrintF("Parameter group %d: %s", i+1, groups[i].name.c_str()));
    Unindent();

    std::vector<RealT> log_C = std::vector<RealT>(GetParameterManager().GetNumParameterGroups(), RealT(INITIAL_LOG_C));
    
    if (GetOptions().GetBoolValue("viterbi_parsing"))
    {
        Error("Not yet implemented.");
    }
    else
    {
        OuterOptimizationWrapper<RealT> outer_optimization_wrapper(this, w, training, holdout);
        outer_optimization_wrapper.Minimize(log_C);
    }
    
    Unindent();
    std::ostringstream oss;
    const std::vector<RealT> C = Exp(log_C);
    oss << "Chose hyperparameters, C = " << C;
    PrintMessage(oss.str());
    
    // Now, retrain on all data
    PrintMessage("Retraining on entire training set...");
    Indent();
    Train(units, w, C);
    Unindent();
}
#endif

//////////////////////////////////////////////////////////////////////
// OptimizationWrapper<RealT>::LearnHyperparameters()
//
// Use Bayesian hyperparameter selection algorithm in order to
// estimate regularization constants.
//////////////////////////////////////////////////////////////////////

#if HYPERPARAMETER_MAJORIZATION_MINIMIZATION
template<class RealT>
void OptimizationWrapper<RealT>::LearnHyperparameters(std::vector<int> units,
                                               std::vector<RealT> &w,
                                               RealT holdout_ratio,
                                               bool toggle_viterbi)
{
    // do hyperparameter optimization
    
    PrintMessage("Starting hyperparameter optimization...");
    Indent();
    
    PrintMessage("List of hyperparameters:");
    Indent();
    const std::vector<HyperparameterGroup> &groups = params.GetHyperparameterGroups();
    for (size_t i = 0; i < groups.size(); i++)
        PrintMessage(SPrintF("Hyperparameter group %d: %s", i+1, groups[i].name.c_str()));
    Unindent();
    
    std::vector<RealT> C = std::vector<RealT>(params.GetNumHyperparameterGroups(), 1);
    
    // iterative relinearization

    for (int iters = 0; iters < NUM_ITERATIVE_RELINEARIZATION_STEPS; iters++)
    {
        // show current set of hyperparameters
        
        PrintMessage("Current hyperparameters:");
        Indent();
        const std::vector<HyperparameterGroup> &groups = params.GetHyperparameterGroups();
        for (size_t i = 0; i < groups.size(); i++)
            PrintMessage(SPrintF("Hyperparameter group %d (%s): %lf", i+1, groups[i].name.c_str(), C[i]));
        Unindent();

        // perform training

        std::ostringstream oss;
        const std::vector<RealT> Ce = params.ExpandHyperparameters(C);
        oss << "Performing optimization using C = " << C;
        PrintMessage(oss.str());
        Indent();
        std::vector<RealT> x(w);
        const RealT f = Train(units, x, C, toggle_viterbi);
        Unindent();

        // compute new hyperparameters

        for (size_t g = 0; g < groups.size(); g++)
        {
            RealT numerator = (groups[g].end - groups[g].begin + 1.0) / 2.0;
            RealT denominator = RealT(MM_SMOOTHING);
            for (int i = groups[g].begin; i < groups[g].end; i++)
                denominator += 0.5 * x[i] * x[i];
            C[g] = numerator / denominator;
        }

        // adjust for Viterbi mode

        if (toggle_viterbi)
        {
            const RealT loss = f - 0.5 * DotProduct(Ce, x*x);
            const RealT loss_multiplier = RealT(units.size()) / (RealT(MM_SMOOTHING) + loss);
            C /= loss_multiplier;
        }
    }
    
    Unindent();
    std::ostringstream oss;
    oss << "Chose hyperparameters, C = " << C;
    PrintMessage(oss.str());
    
    // now, retrain on all data
    
    PrintMessage("Retraining on entire training set...");
    Indent();
    Train(units, w, C, toggle_viterbi);
    Unindent();
}
#endif
