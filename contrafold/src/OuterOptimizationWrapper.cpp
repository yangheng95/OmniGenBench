/////////////////////////////////////////////////////////////////
// OuterOptimizationWrapper.cpp
//
// Implementation of functors needed for optimization.
/////////////////////////////////////////////////////////////////

#include "OuterOptimizationWrapper.hpp"

/////////////////////////////////////////////////////////////////
// OuterOptimizationWrapper::OuterOptimizationWrapper()
//
// Constructor.
/////////////////////////////////////////////////////////////////

OuterOptimizationWrapper::OuterOptimizationWrapper(OptimizationWrapper *optimizer,
                                                   const std::vector<double> &initial_values,
                                                   const std::vector<int> &training,
                                                   const std::vector<int> &holdout):
    LBFGS<double>(20,1e-5,1000,1e-6,3,1), optimizer(optimizer), initial_values(initial_values), training(training), holdout(holdout)
{}

/////////////////////////////////////////////////////////////////
// OuterOptimizationWrapper::ComputeFunction()
//
// Compute function for outer iteration.
/////////////////////////////////////////////////////////////////

double OuterOptimizationWrapper::ComputeFunction(const std::vector<double> &C)
{
    std::ostringstream oss;
    oss << "Computing outer function using C = " << C;
    optimizer->PrintMessage(oss.str());
    optimizer->Indent();
    
    // values = solution of OPT1 for current C
    
    std::vector<double> values = initial_values;
    optimizer->PrintMessage("Solving OPT1...");
    optimizer->Indent();
    optimizer->Train(training, values, optimizer->TransformHyperparameters(C), false);
    optimizer->Unindent();
    
    // compute holdout logloss
    
    double ret = optimizer->computation.ComputeFunction(holdout, values + optimizer->base_values, false);
    
    optimizer->Unindent();
    optimizer->PrintMessage(SPrintF("Finished outer function: %lf", ret));
    return ret;
}

/////////////////////////////////////////////////////////////////
// OuterOptimizationWrapper::ComputeGradient()
//
// Compute the regularized logloss gradient using a particular
// parameter set and fixed regularization hyperparameters.
/////////////////////////////////////////////////////////////////

void OuterOptimizationWrapper::ComputeGradient(std::vector<double> &g, const std::vector<double> &C)
{
    std::ostringstream oss;
    oss << "Computing outer gradient using C = " << C;
    optimizer->PrintMessage(oss.str());
    optimizer->Indent();
    
    // show current set of hyperparameters
    
    optimizer->PrintMessage("Current hyperparameters:");
    optimizer->Indent();
    const std::vector<HyperparameterGroup> &groups = optimizer->params.GetHyperparameterGroups();
    const std::vector<double> Cp = optimizer->TransformHyperparameters(C);
    for (size_t i = 0; i < groups.size(); i++)
        optimizer->PrintMessage(SPrintF("Hyperparameter group %d (%s): %lf", i+1, groups[i].name.c_str(), Cp[i]));
    optimizer->Unindent();
    
    // values = solution of OPT1 for current C
    
    std::vector<double> values = initial_values;
    optimizer->PrintMessage("Solving OPT1...");
    optimizer->Indent();
    optimizer->Train(training, values, Cp, false);
    optimizer->Unindent();
    
    // compute holdout logloss
    
    std::vector<double> holdout_gradient = optimizer->computation.ComputeGradient(holdout, values + optimizer->base_values, false);
    
    // solve linear system

    CGOptimizationWrapper cg_linear(optimizer, training, values, Cp);
    std::vector<double> x(holdout_gradient.size());
    
    optimizer->PrintMessage("Solving linear system...");
    optimizer->Indent();
    cg_linear.Minimize(holdout_gradient,x);
    optimizer->Unindent();
    
    // form "B" matrix
    
    const std::vector<double> Cd = optimizer->TransformHyperparametersDerivative(C);
    std::vector<std::vector<double> > B(x.size(), std::vector<double>(optimizer->params.GetNumHyperparameterGroups()));
    for (size_t i = 0; i < groups.size(); i++)
        for (int j = groups[i].begin; j <= groups[i].end; j++)
            B[j][i] = values[j] * Cd[i];
    
    // compute gradient

    g.clear();
    g.resize(C.size());
    for (size_t i = 0; i < B.size(); i++)
        g -= x[i] * B[i];
    
    optimizer->Unindent();
    optimizer->PrintMessage(SPrintF("Finished outer gradient: norm = %lf", Norm(g)));
}

/////////////////////////////////////////////////////////////////
// OuterOptimizationWrapper::Report()
//
// Routines for printing results and messages from the optimizer.
/////////////////////////////////////////////////////////////////

void OuterOptimizationWrapper::Report(int iteration, const std::vector<double> &x, double f, double step_size)
{
    std::ostringstream oss;
    oss << "Outer iteration " << iteration << ": holdout f = " << f << ", C = " << x << ", step length = " << step_size << ", efficiency = " << optimizer->computation.GetEfficiency() << "%";
    optimizer->PrintMessage(oss.str());
}

void OuterOptimizationWrapper::Report(const std::string &s) 
{
    optimizer->PrintMessage(SPrintF("Outer message: %s", s.c_str()));
}

