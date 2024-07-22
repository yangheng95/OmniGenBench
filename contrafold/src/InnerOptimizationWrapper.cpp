/////////////////////////////////////////////////////////////////
// InnerOptimizationWrapper.cpp
//
// Implementation of functors needed for optimization.
/////////////////////////////////////////////////////////////////

#include "InnerOptimizationWrapper.hpp"

/////////////////////////////////////////////////////////////////
// InnerOptimizationWrapper::InnerOptimizationWrapper()
//
// Constructor.
/////////////////////////////////////////////////////////////////

InnerOptimizationWrapper::InnerOptimizationWrapper(OptimizationWrapper *optimizer,
                                                   const std::vector<int> &units,
                                                   const std::vector<double> &C) :
    LBFGS<double>(), optimizer(optimizer), units(units), C(C), best_f(std::numeric_limits<double>::infinity())
{}

/////////////////////////////////////////////////////////////////
// InnerOptimizationWrapper::ComputeFunction()
//
// Compute the regularized logloss using a particular
// parameter set and fixed regularization hyperparameters.
/////////////////////////////////////////////////////////////////

double InnerOptimizationWrapper::ComputeFunction(const std::vector<double> &x)
{
    std::vector<double> Ce = optimizer->params.ExpandHyperparameters(C);
    return optimizer->computation.ComputeFunction(units, x + optimizer->base_values, false) + 0.5 * DotProduct(Ce, x*x);  
}

/////////////////////////////////////////////////////////////////
// InnerOptimizationWrapper::ComputeGradient()
//
// Compute the regularized logloss gradient using a particular
// parameter set and fixed regularization hyperparameters.
/////////////////////////////////////////////////////////////////

void InnerOptimizationWrapper::ComputeGradient(std::vector<double> &g, const std::vector<double> &x)
{
    std::vector<double> Ce = optimizer->params.ExpandHyperparameters(C);
    g = optimizer->computation.ComputeGradient(units, x + optimizer->base_values, false) + Ce * x;
}

/////////////////////////////////////////////////////////////////
// InnerOptimizationWrapper::Report()
//
// Routines for printing results and messages from the optimizer.
/////////////////////////////////////////////////////////////////

void InnerOptimizationWrapper::Report(int iteration, const std::vector<double> &x, double f, double step_size)
{
    // write results to disk
    
    if (f < best_f)
    {
        best_f = f;
        best_x = x;
        optimizer->params.WriteToFile(SPrintF("optimize.params.iter%d", iteration), best_x + optimizer->base_values);
    }
    
    // write results to console
    
    std::vector<double> Ce = optimizer->params.ExpandHyperparameters(C);
    const double unregularized = f - 0.5 * DotProduct(Ce, x*x);
    optimizer->PrintMessage(SPrintF("Inner iteration %d: f = %lf (%lf), |x| = %lf, step = %lf, efficiency = %lf%%", 
                                    iteration, f, unregularized, Norm(x), step_size, optimizer->computation.GetEfficiency()));
}

void InnerOptimizationWrapper::Report(const std::string &s) 
{
    optimizer->PrintMessage(SPrintF("Inner message: %s", s.c_str()));
}

