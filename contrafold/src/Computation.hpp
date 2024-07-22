/////////////////////////////////////////////////////////////////
// Computation.hpp
//
// This class provides a wrapper around a generic
// DistributedComputation object that provides specific
// functionality such as
// 
//   - forming output filenames
//   - predicting structures for all input sequences
//   - performing computations for the optimizer.
/////////////////////////////////////////////////////////////////

#ifndef COMPUTATION_HPP
#define COMPUTATION_HPP

#ifdef MULTI
#include <mpi.h>
#endif

#include "Config.hpp"
#include "LogSpace.hpp"
#include "Utilities.hpp"
#include "SStruct.hpp"
#include "DistributedComputation.hpp"
#include "ProcessingUnit.hpp"
#include "FileDescription.hpp"

/////////////////////////////////////////////////////////////////
// class Computation
//
// Wrapper class for DistributedComputation.
/////////////////////////////////////////////////////////////////

class Computation : public DistributedComputation
{
    friend class UnitComparator;
    
    bool toggle_complementary_only;
    bool toggle_use_constraints;
    bool toggle_partition;
    bool toggle_viterbi;
    double posterior_cutoff;
    
    std::vector<FileDescription> file_descriptions;
    Parameters temp_params;
    
    // the following member variables are used to "cache" work to
    // ensure that it is not repeated unnecessarily
    
    std::vector<int> cached_units;
    std::vector<double> cached_values;
    std::vector<double> cached_v;
    std::vector<double> cached_function;
    std::vector<double> cached_gradient;
    std::vector<double> cached_Hv;
    
    std::string FormatOutputFilename(bool output_to_directory, const std::string &input_filename, const std::string &output_filename) const;
    
    bool Includes(const std::vector<WorkUnit *> &work_units1,
                  const std::vector<WorkUnit *> &work_units2) const;
    
public:
    
    // accessible methods
    
    Computation(const std::vector<std::string> &filenames,
                const std::string &output_parens_destination,
                const std::string &output_bpseq_destination,
                const std::string &output_posteriors_destination,
                bool toggle_complementary_only,
                bool toggle_use_constraints,
                bool toggle_partition,
                bool toggle_viterbi,
                bool toggle_verbose,
                double posterior_cutoff);
    ~Computation();
    
    std::vector<int> GetAllUnits() const;
    
    // methods to act on vectors of work units
    
    std::vector<int> FilterNonparsable(const std::vector<int> &units, const std::vector<double> &values);
    void PredictStructure(const std::vector<int> &units, const std::vector<double> &values, double gammar, bool exact);
    double ComputeFunction(const std::vector<int> &units, const std::vector<double> &values, bool exact);
    std::vector<double> ComputeGradient(const std::vector<int> &units, const std::vector<double> &values, bool exact);
    std::vector<double> ComputeHv(const std::vector<int> &units, const std::vector<double> &values, const std::vector<double> &v);
    
    void SanityCheckGradient(const std::vector<int> &units, const std::vector<double> &x);
    const std::vector<int> SortUnitsByDecreasingSize(const std::vector<int> &units);
    
    // methods to act on individual work units
    
    void DoComputation(std::vector<double> &ret, const WorkUnit *unit, const std::vector<double> &values);
    void CheckParsability(const ProcessingUnit &unit, std::vector<double> &ret, const std::vector<double> &values);
    template<class T> void ComputeFunction(const ProcessingUnit &unit, std::vector<double> &ret, const std::vector<double> &values);
    template<class T> void ComputeGradient(const ProcessingUnit &unit, std::vector<double> &ret, const std::vector<double> &values);
    void ComputeHv(const ProcessingUnit &unit, std::vector<double> &ret, const std::vector<double> &values);
    template<class T> void PredictStructure(const ProcessingUnit &unit, const std::vector<double> &values);
    
};  

/////////////////////////////////////////////////////////////////
// class UnitComparator
/////////////////////////////////////////////////////////////////

class UnitComparator
{
    Computation &computation;
public:
    UnitComparator(Computation &computation);
    bool operator()(int i, int j) const;
};

#endif
