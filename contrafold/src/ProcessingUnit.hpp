/////////////////////////////////////////////////////////////////
// ProcessingUnit.hpp
/////////////////////////////////////////////////////////////////

#ifndef PROCESSINGUNIT_HPP
#define PROCESSINGUNIT_HPP

#include "DistributedComputation.hpp"
#include "InferenceEngine.hpp"
#include "SparseMatrix.hpp"
#include <string>

enum ProcessingType
{ 
    CHECK_PARSABILITY,
    COMPUTE_FUNCTION,
    COMPUTE_GRADIENT,
    COMPUTE_HV,
    PREDICT_STRUCTURE
};

/////////////////////////////////////////////////////////////////
// struct ProcessingUnit
/////////////////////////////////////////////////////////////////

struct ProcessingUnit : public WorkUnit
{
    int command;
    int index;
    int size;
    double gammar;
    bool toggle_complementary_only;
    bool toggle_use_constraints;
    bool toggle_partition;
    bool toggle_viterbi;
    bool toggle_exact;
    double posterior_cutoff;
    
    ProcessingUnit();
    
    ProcessingUnit(int command, 
                   int index, 
                   int size,
                   double gammar,
                   bool toggle_complementary_only,
                   bool toggle_use_constraints,
                   bool toggle_partition,
                   bool toggle_viterbi,
                   bool toggle_exact,
                   double posterior_cutoff);
    
    ProcessingUnit (const ProcessingUnit &rhs);
    ProcessingUnit &operator= (const ProcessingUnit &rhs);
    
    int GetDescriptionSize() const;
    double GetEstimatedTime() const;
    std::string GetSummary() const;
};

#endif
