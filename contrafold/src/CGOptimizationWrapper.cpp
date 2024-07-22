/////////////////////////////////////////////////////////////////
// CGOptimizationWrapper.cpp
//
// CG optimization code.
/////////////////////////////////////////////////////////////////

#include "CGOptimizationWrapper.hpp"

/////////////////////////////////////////////////////////////////
// CGOptimizationWrapper::CGOptimizationWrapper()
//
// Constructor.
/////////////////////////////////////////////////////////////////

CGOptimizationWrapper::CGOptimizationWrapper(OptimizationWrapper *optimizer,
                                             const std::vector<int> &units,
                                             const std::vector<double> &values,
                                             const std::vector<double> &C) :
    CGLinear<double>(), optimizer(optimizer), units(units), values(values), C(C)
{}

/////////////////////////////////////////////////////////////////
// CGOptimizationWrapper::ComputeAx()
//
// Compute Hessian-vector product.
/////////////////////////////////////////////////////////////////

void CGOptimizationWrapper::ComputeAx(std::vector<double> &Ax, const std::vector<double> &x)
{
    std::vector<double> Ce = optimizer->params.ExpandHyperparameters(C);
    Ax = optimizer->computation.ComputeHv(units, values + optimizer->base_values, x) + Ce * x;
}

/////////////////////////////////////////////////////////////////
// CGOptimizationWrapper::Report()
//
// Provide progress report on CG algorithm.
/////////////////////////////////////////////////////////////////

void CGOptimizationWrapper::Report(int iteration, const std::vector<double> &x, double f, double step_size)
{
    std::ostringstream oss;
    oss << "CG iteration " << iteration
        << ": f = " << f
        << ", |x| = " << Norm(x)
        << ", step size = " << step_size
        << ", efficiency = " << optimizer->computation.GetEfficiency() << "%";
    optimizer->PrintMessage(oss.str());
}

void CGOptimizationWrapper::Report(const std::string &s)
{
    optimizer->PrintMessage(SPrintF("CG message: %s", s.c_str()));
}
