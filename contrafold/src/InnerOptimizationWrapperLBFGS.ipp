//////////////////////////////////////////////////////////////////////
// InnerOptimizationWrapperLBFGS.ipp
//
// Implementation of functors needed for optimization.
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// InnerOptimizationWrapperLBFGS::InnerOptimizationWrapperLBFGS()
//
// Constructor.
//////////////////////////////////////////////////////////////////////

template<class RealT>
InnerOptimizationWrapperLBFGS<RealT>::InnerOptimizationWrapperLBFGS(OptimizationWrapper<RealT> *optimization_wrapper,
                                                                    const std::vector<int> &units,
                                                                    const std::vector<RealT> &C) :
    //LBFGS<RealT>(20, 1e-5, 1000, 1e-4),
    LBFGS<RealT>(20, 1e-5, 100),
    InnerOptimizationWrapper<RealT>(optimization_wrapper, units, C),
    log_base(optimization_wrapper->GetOptions().GetRealValue("log_base"))
{}

//////////////////////////////////////////////////////////////////////
// InnerOptimizationWrapperLBFGS::ComputeFunction()
//
// Compute the regularized logloss using a particular
// parameter set and fixed regularization hyperparameters.
//////////////////////////////////////////////////////////////////////

template<class RealT>
RealT InnerOptimizationWrapperLBFGS<RealT>::ComputeFunction(const std::vector<RealT> &w)
{
    return this->optimization_wrapper->GetComputationWrapper().ComputeFunction(this->units, w, false, true, log_base) + RealT(0.5) * DotProduct(this->C, w*w) + DotProduct(w, this->bias);
}

//////////////////////////////////////////////////////////////////////
// InnerOptimizationWrapperLBFGS::ComputeGradient()
//
// Compute the regularized logloss gradient using a particular
// parameter set and fixed regularization hyperparameters.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void InnerOptimizationWrapperLBFGS<RealT>::ComputeGradient(std::vector<RealT> &g, const std::vector<RealT> &w)
{
    g = this->optimization_wrapper->GetComputationWrapper().ComputeGradient(this->units, w, false, true, log_base) + this->C * w + this->bias;
}

//////////////////////////////////////////////////////////////////////
// InnerOptimizationWrapperLBFGS::Report()
//
// Routines for printing results and messages from the optimizer.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void InnerOptimizationWrapperLBFGS<RealT>::Report(int iteration, const std::vector<RealT> &w, RealT f, RealT step_size)
{
    // write results to disk
    this->optimization_wrapper->GetParameterManager().WriteToFile(SPrintF("optimize.params.iter%d", iteration), w);
    
    // write results to console
    this->optimization_wrapper->PrintMessage(SPrintF("Inner iteration %d: f = %lf (%lf) [%lf], |w| = %lf, step = %lf, efficiency = %lf%%", 
                                                     iteration, double(f), double(f - RealT(0.5) * DotProduct(this->C, w*w)),
                                                     double(this->optimization_wrapper->GetComputationWrapper().ComputeFunction(this->units, w, true, true, log_base) + RealT(0.5) * DotProduct(this->C, w*w) + DotProduct(w, this->bias)),
                                                     double(Norm(w)), double(step_size),
                                                     double(this->optimization_wrapper->GetComputationEngine().GetEfficiency())));
}

template<class RealT>
void InnerOptimizationWrapperLBFGS<RealT>::Report(const std::string &s) 
{
    this->optimization_wrapper->PrintMessage(SPrintF("Inner message: %s", s.c_str()));
}

//////////////////////////////////////////////////////////////////////
// InnerOptimizationWrapperLBFGS::Minimize()
//
// Perform LBFGS optimization.
//////////////////////////////////////////////////////////////////////

template<class RealT>
RealT InnerOptimizationWrapperLBFGS<RealT>::Minimize(std::vector<RealT> &x0)
{
    return LBFGS<RealT>::Minimize(x0);
        
    /*
    RealT f = RealT(1e20);
    
    for (log_base = 1; log_base < RealT(1e6); log_base *= 2)
    {
        this->optimization_wrapper->PrintMessage(SPrintF("Inner optimization using LOG_BASE = %lf", double(log_base)));
        this->optimization_wrapper->Indent();
        f = LBFGS<RealT>::Minimize(x0);        
        this->optimization_wrapper->Unindent();
    }
    
    return f;
    */
}
