/////////////////////////////////////////////////////////////////
// ProcessingUnit.cc
//
// Class for storing individual work units to be performed by
// the DistributedComputationWrapper class.
/////////////////////////////////////////////////////////////////

#include "ProcessingUnit.hpp"

/////////////////////////////////////////////////////////////////
// ProcessingUnit::ProcessingUnit()
//
// Simple constructors.
/////////////////////////////////////////////////////////////////

ProcessingUnit::ProcessingUnit() : 
    WorkUnit(), command(0), index(0), size(0), gammar(0),
    toggle_complementary_only(false), toggle_use_constraints(false), 
    toggle_partition(false), toggle_viterbi(false), toggle_exact(false), posterior_cutoff(0)
{}

ProcessingUnit::ProcessingUnit(int command, 
                               int index, 
                               int size,
                               double gammar,
                               bool toggle_complementary_only, 
                               bool toggle_use_constraints,
                               bool toggle_partition,
                               bool toggle_viterbi,
                               bool toggle_exact,
                               double posterior_cutoff) :
    WorkUnit(), command(command), index(index), size(size), gammar(gammar),
    toggle_complementary_only(toggle_complementary_only), 
    toggle_use_constraints(toggle_use_constraints),
    toggle_partition(toggle_partition), toggle_viterbi(toggle_viterbi),
    toggle_exact(toggle_exact), posterior_cutoff(posterior_cutoff)
{}

ProcessingUnit::ProcessingUnit(const ProcessingUnit &rhs) : 
    WorkUnit(), command(rhs.command), index(rhs.index), size(rhs.size), gammar(rhs.gammar),
    toggle_complementary_only(rhs.toggle_complementary_only), 
    toggle_use_constraints(rhs.toggle_use_constraints),
    toggle_partition(rhs.toggle_partition), toggle_viterbi(rhs.toggle_viterbi),
    toggle_exact(rhs.toggle_exact), posterior_cutoff(rhs.posterior_cutoff)
{}

/////////////////////////////////////////////////////////////////
// ProcessingUnit::operator=()
//
// Assignment operator.
/////////////////////////////////////////////////////////////////

ProcessingUnit &ProcessingUnit::operator=(const ProcessingUnit &rhs)
{
    if (this != &rhs)
    {
        command = rhs.command;
        index = rhs.index;
        size = rhs.size;
        gammar = rhs.gammar;
        toggle_complementary_only = rhs.toggle_complementary_only;
        toggle_use_constraints = rhs.toggle_use_constraints;
        toggle_partition = rhs.toggle_partition;
        toggle_viterbi = rhs.toggle_viterbi;
        posterior_cutoff = rhs.posterior_cutoff;
        toggle_exact = rhs.toggle_exact;
    }
    return *this;
}

/////////////////////////////////////////////////////////////////
// ProcessingUnit::GetDescriptionSize()
//
// Return the length of the object.
/////////////////////////////////////////////////////////////////

int ProcessingUnit::GetDescriptionSize() const
{
    return sizeof(ProcessingUnit);
}

/////////////////////////////////////////////////////////////////
// ProcessingUnit::GetEstimatedTime()
//
// Return the estimated amount of time this example will take.
/////////////////////////////////////////////////////////////////

double ProcessingUnit::GetEstimatedTime() const
{
    return double(size);
}

/////////////////////////////////////////////////////////////////
// ProcessingUnit::GetSummary()
//
// Return a summary string description the work unit.
/////////////////////////////////////////////////////////////////

std::string ProcessingUnit::GetSummary() const
{
    const char *description[] =
        {
            "Checking parsability",
            "Compute function only",
            "Computing function and subgradient",
            "Computing Hessian-vector product",
            "Predicting structure"
        };
    
    return SPrintF("%s for input file %d.", description[command], index);
}

