//////////////////////////////////////////////////////////////////////
// ComputationWrapper.cpp
//////////////////////////////////////////////////////////////////////

#include "ComputationWrapper.hpp"

//////////////////////////////////////////////////////////////////////
// ComputationWrapper::ComputationWrapper()
// ComputationWrapper::~ComputationWrapper()
//
// Constructor and destructor.
//////////////////////////////////////////////////////////////////////

template<class RealT>
ComputationWrapper<RealT>::ComputationWrapper(ComputationEngine<RealT> &computation_engine) :
    computation_engine(computation_engine)
{}

template<class RealT>
ComputationWrapper<RealT>::~ComputationWrapper()
{}

//////////////////////////////////////////////////////////////////////
// ComputationWrapper::GetAllUnits()
//
// Return a vector containing the index of every input file.
//////////////////////////////////////////////////////////////////////

template<class RealT>
std::vector<int> ComputationWrapper<RealT>::GetAllUnits() const 
{
    std::vector<int> ret;
    for (size_t i = 0; i < GetDescriptions().size(); i++)
        ret.push_back(int(i));
    return ret;
}

//////////////////////////////////////////////////////////////////////
// ComputationWrapper::ComputeSolutionNormBound()
//
// Return a bound on the norm for each batch gradient iteration, not
// including regularization.
//////////////////////////////////////////////////////////////////////

template<class RealT>
RealT ComputationWrapper<RealT>::ComputeSolutionNormBound(const std::vector<int> &units,
                                                          const std::vector<RealT> &C,
                                                          RealT log_base)
{
    Assert(computation_engine.IsMasterNode(), "Routine should only be called by master process.");

    static std::vector<int> cached_units;
    static std::vector<RealT> cached_C;
    static RealT cached_bound = 0;

    // check cache
    if (cached_units != units || cached_C != C)
    {
        // set up computation        
        shared_info.command = COMPUTE_SOLUTION_NORM_BOUND;
        shared_info.log_base = log_base;

        nonshared_info.resize(units.size());
        for (size_t i = 0; i < units.size(); i++)
        {
            nonshared_info[i].index = units[i];
        }
        
        // perform computation
        std::vector<RealT> entropy_plus_loss;
        computation_engine.DistributeComputation(entropy_plus_loss, shared_info, nonshared_info);
        cached_bound = Sqrt(Sum(entropy_plus_loss) / (Min(C) + RealT(1e-10)));
        std::cerr << "Solution norm bound: " << cached_bound << std::endl;

        // save cache
        cached_units = units;
        cached_C = C;
    }
    
    return cached_bound;
}

//////////////////////////////////////////////////////////////////////
// ComputationWrapper::ComputeGradientNormBound()
//
// Return a bound on the norm for each batch gradient iteration, not
// including regularization.
//////////////////////////////////////////////////////////////////////

template<class RealT>
RealT ComputationWrapper<RealT>::ComputeGradientNormBound(const std::vector<int> &units,
                                                          const std::vector<RealT> &C,
                                                          RealT log_base)
{
    Assert(computation_engine.IsMasterNode(), "Routine should only be called by master process.");

    // compute the max L1 norm for feature vectors; the max L1
    // norm also serves as a bound on the max L2 norm

    shared_info.command = COMPUTE_GRADIENT_NORM_BOUND;
    shared_info.log_base = log_base;

    nonshared_info.resize(units.size());
    for (size_t i = 0; i < units.size(); i++)
    {
        nonshared_info[i].index = units[i];
    }

    std::vector<RealT> max_feature_L1_norm;
    computation_engine.DistributeComputation(max_feature_L1_norm, shared_info, nonshared_info);

    RealT bound = Max(C) * ComputeSolutionNormBound(units, C, log_base) + RealT(2) * Sum(max_feature_L1_norm);
    std::cerr << "Gradient norm bound: " << bound << std::endl;
    
    return bound;
}

//////////////////////////////////////////////////////////////////////
// ComputationWrapper::FilterNonparsable()
//
// Filter a vector of units, removing any units whose supplied
// structures are not parsable.
//////////////////////////////////////////////////////////////////////

template<class RealT>
std::vector<int> ComputationWrapper<RealT>::FilterNonparsable(const std::vector<int> &units)
{
    Assert(computation_engine.IsMasterNode(), "Routine should only be called by master process.");

    std::vector<RealT> parsable;

    shared_info.command = CHECK_PARSABILITY;
        
    nonshared_info.resize(units.size());
    for (size_t i = 0; i < units.size(); i++)
    {
        nonshared_info[i].index = units[i];
    }
    
    computation_engine.DistributeComputation(parsable, shared_info, nonshared_info);

    std::vector<int> ret;
    for (size_t i = 0; i < units.size(); i++)
    {
        Assert(units[i] >= 0 && units[i] < int(parsable.size()), "Out-of-bounds index.");
        if (parsable[units[i]])
        {
            ret.push_back(units[i]);
        }
        else
        {
            std::cerr << "No valid parse for file: " << GetDescriptions()[units[i]].input_filename << std::endl;
        }      
    }
    
    return ret;
}

//////////////////////////////////////////////////////////////////////
// ComputationWrapper::ComputeLoss()
//
// Compute loss function for model over a fixed set of work units
// using a particular setting of the parameters.
//////////////////////////////////////////////////////////////////////

template<class RealT>
RealT ComputationWrapper<RealT>::ComputeLoss(const std::vector<int> &units,
                                             const std::vector<RealT> &w,
                                             RealT log_base)
{
    Assert(computation_engine.IsMasterNode(), "Routine should only be called by master process.");
    if (int(w.size()) > SHARED_PARAMETER_SIZE) Error("SHARED_PARAMETER_SIZE in Config.hpp too small; increase to at least %d.", int(w.size()));
    
    std::vector<RealT> ret;

    shared_info.command = COMPUTE_LOSS;
    for (size_t i = 0; i < w.size(); i++)
    {
        shared_info.w[i] = w[i];
    }
    shared_info.use_loss = true;
    shared_info.log_base = log_base;
    
    nonshared_info.resize(units.size());
    for (size_t i = 0; i < units.size(); i++)
    {
        nonshared_info[i].index = units[i];
    }
    
    computation_engine.DistributeComputation(ret, shared_info, nonshared_info);
    return ret[0];
}

//////////////////////////////////////////////////////////////////////
// ComputationWrapper::ComputeFunction()
//
// Compute negative log-likelihood of the model over a fixed set
// of work units using a particular setting of the parameters.
//////////////////////////////////////////////////////////////////////

template<class RealT>
RealT ComputationWrapper<RealT>::ComputeFunction(const std::vector<int> &units,
                                                 const std::vector<RealT> &w,
                                                 bool toggle_use_nonsmooth,
                                                 bool toggle_use_loss,
                                                 RealT log_base)
{
#if STOCHASTIC_GRADIENT
    Error("Should not get here.");
#endif

    Assert(computation_engine.IsMasterNode(), "Routine should only be called by master process.");
    if (int(w.size()) > SHARED_PARAMETER_SIZE) Error("SHARED_PARAMETER_SIZE in Config.hpp too small; increase to at least %d.", int(w.size()));

    // check cache
    if (cached_units != units ||
        cached_w != w ||
        cached_toggle_use_nonsmooth != toggle_use_nonsmooth ||
        cached_toggle_use_loss != toggle_use_loss ||
        cached_function.size() == 0)
    {
        // set up computation
        shared_info.command = COMPUTE_FUNCTION;
        for (size_t i = 0; i < w.size(); i++)
        {
            shared_info.w[i] = w[i];
        }
        shared_info.use_nonsmooth = toggle_use_nonsmooth;
        shared_info.use_loss = toggle_use_loss;
        shared_info.log_base = log_base;
        
        nonshared_info.resize(units.size());
        for (size_t i = 0; i < units.size(); i++)
        {
            nonshared_info[i].index = units[i];
        }

        // perform computation
        computation_engine.DistributeComputation(cached_function, shared_info, nonshared_info);
        Assert(cached_function.size() == 1, "Unexpected return value size.");

        // replace cache
        cached_units = units;
        cached_w = w;
        cached_toggle_use_nonsmooth = toggle_use_nonsmooth;
        cached_toggle_use_loss = toggle_use_loss;
        cached_gradient.clear();
    }
            
    return cached_function[0];
}

//////////////////////////////////////////////////////////////////////
// ComputationWrapper::ComputeGradient()
//
// Compute gradient of the negative log-likelihood of the model 
// over a fixed set of work units using a particular setting of 
// the parameters.
//////////////////////////////////////////////////////////////////////

template<class RealT>
std::vector<RealT> ComputationWrapper<RealT>::ComputeGradient(const std::vector<int> &units,
                                                              const std::vector<RealT> &w,
                                                              bool toggle_use_nonsmooth,
                                                              bool toggle_use_loss,
                                                              RealT log_base)
{
#if STOCHASTIC_GRADIENT
    Error("Should not get here.");
#endif

    Assert(computation_engine.IsMasterNode(), "Routine should only be called by master process.");
    if (int(w.size()) > SHARED_PARAMETER_SIZE) Error("SHARED_PARAMETER_SIZE in Config.hpp too small; increase to at least %d.", int(w.size()));

    // check cache
    if (cached_units != units ||
        cached_w != w ||
        cached_toggle_use_nonsmooth != toggle_use_nonsmooth ||
        cached_toggle_use_loss != toggle_use_loss ||
        cached_gradient.size() == 0)
    {
        // set up computation
        shared_info.command = COMPUTE_GRADIENT;
        for (size_t i = 0; i < w.size(); i++)
        {
            shared_info.w[i] = w[i];
        }
        shared_info.use_nonsmooth = toggle_use_nonsmooth;
        shared_info.use_loss = toggle_use_loss;
        shared_info.log_base = log_base;
        
        nonshared_info.resize(units.size());
        for (size_t i = 0; i < units.size(); i++)
        {
            nonshared_info[i].index = units[i];
        }

        // perform computation
        computation_engine.DistributeComputation(cached_gradient, shared_info, nonshared_info);
        Assert(cached_gradient.size() == GetParameterManager().GetNumLogicalParameters() + 1, "Unexpected return value size.");

        // replace cache
        cached_units = units;
        cached_w = w;
        cached_toggle_use_nonsmooth = toggle_use_nonsmooth;
        cached_toggle_use_loss = toggle_use_loss;
        cached_function.clear();
        cached_function.push_back(cached_gradient.back());
        cached_gradient.pop_back();
    }
            
    return cached_gradient;
}

//////////////////////////////////////////////////////////////////////
// ComputationWrapper::ComputeHessianVectorProduct()
//
// Compute product of the Hessian with an arbitrary vector v.
//////////////////////////////////////////////////////////////////////

template<class RealT>
std::vector<RealT> ComputationWrapper<RealT>::ComputeHessianVectorProduct(const std::vector<int> &units,
                                                                          const std::vector<RealT> &w,
                                                                          const std::vector<RealT> &v,
                                                                          bool toggle_use_loss,
                                                                          RealT log_base)
{
    Assert(computation_engine.IsMasterNode(), "Routine should only be called by master process.");
    if (int(w.size()) > SHARED_PARAMETER_SIZE) Error("SHARED_PARAMETER_SIZE in Config.hpp too small; increase to at least %d.", int(w.size()));
    if (int(v.size()) > SHARED_PARAMETER_SIZE) Error("SHARED_PARAMETER_SIZE in Config.hpp too small; increase to at least %d.", int(w.size()));
    if (GetOptions().GetBoolValue("viterbi_parsing")) Error("Hessian-vector products should not be needed when using Viterbi parsing.");

    std::vector<RealT> ret;

    shared_info.command = COMPUTE_HV;
    for (size_t i = 0; i < w.size(); i++)
    {
        shared_info.w[i] = w[i];
    }
    for (size_t i = 0; i < v.size(); i++)
    {
        shared_info.v[i] = v[i];
    }
    shared_info.use_nonsmooth = false;
    shared_info.use_loss = toggle_use_loss;
    shared_info.log_base = log_base;
    
    nonshared_info.resize(units.size());
    for (size_t i = 0; i < units.size(); i++)
    {
        nonshared_info[i].index = units[i];
    }
    
    computation_engine.DistributeComputation(ret, shared_info, nonshared_info);
    Assert(ret.size() == GetParameterManager().GetNumLogicalParameters() + 1, "Unexpected return value size.");
    ret.pop_back();

    return ret;
}


//////////////////////////////////////////////////////////////////////
// ComputationWrapper::Predict()
//
// Run prediction algorithm on each of the work units.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void ComputationWrapper<RealT>::Predict(const std::vector<int> &units,
                                        const std::vector<RealT> &w,
                                        RealT gamma,
                                        RealT log_base)
{
    Assert(computation_engine.IsMasterNode(), "Routine should only be called by master process.");
    if (int(w.size()) > SHARED_PARAMETER_SIZE) Error("SHARED_PARAMETER_SIZE in Config.hpp too small; increase to at least %d.", int(w.size()));

    if (GetOptions().GetBoolValue("verbose_output"))
    {
        std::cerr << "Performing predictions with gamma=" << double(gamma) << "..." << std::endl;
    }
        
    std::vector<RealT> ret;

    shared_info.command = PREDICT;
    for (size_t i = 0; i < w.size(); i++)
    {
        shared_info.w[i] = w[i];
    }
    shared_info.gamma = gamma;
    shared_info.log_base = log_base;
    
    nonshared_info.resize(units.size());
    for (size_t i = 0; i < units.size(); i++)
    {
        nonshared_info[i].index = units[i];
    }
    
    computation_engine.DistributeComputation(ret, shared_info, nonshared_info);
}

//////////////////////////////////////////////////////////////////////
// ComputationWrapper::SanityCheckGradient()
//
// Perform sanity check for the gradient computation.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void ComputationWrapper<RealT>::SanityCheckGradient(const std::vector<int> &units,
                                                    const std::vector<RealT> &x)
{
    const int NUM_PER_GROUP = 5;
    const int ATTEMPTS = 8;
    
    std::cerr << "Starting gradient sanity check..." << std::endl;
    
    std::vector<RealT> g = ComputeGradient(units, x, false, true, GetOptions().GetRealValue("log_base"));
    RealT f = ComputeFunction(units, x, false, true, GetOptions().GetRealValue("log_base"));
    std::vector<RealT> xp = x;

    const std::vector<ParameterGroup> &groups = GetParameterManager().GetParameterGroups();
    for (size_t k = 0; k < groups.size(); k++)
    {
        int num_left = NUM_PER_GROUP;
        
        // perform sanity check for a group of parameters
        
        std::cerr << "Performing sanity check for parameter group: " << groups[k].name 
                  << " (indices " << groups[k].begin << " to " << groups[k].end << ", limit " << num_left << ")" << std::endl;
        
        for (int i = groups[k].begin; num_left && i < groups[k].end; i++)
        {
            // perform sanity check for a single parameter
            
            std::vector<RealT> gp(ATTEMPTS);
            for (int j = 0; j < ATTEMPTS; j++)
            {
                RealT EPSILON = Pow(10.0, RealT(-j));
                xp[i] += EPSILON;
                gp[j] = (ComputeFunction(units, xp, false, true, GetOptions().GetRealValue("log_base")) - f) / EPSILON;
                xp[i] = x[i];
                
                if (g[i] == gp[j]) break;
                if (Abs(g[i] - gp[j]) / (Abs(g[i]) + Abs(gp[j])) < 1e-5) break;
            }
            
            // print results of sanity check
            
            if (g[i] != 0 || g[i] != gp[0])
            {
                std::cerr << std::setw(13) << i << std::setw(13) << g[i];
                for (int j = 0; j < ATTEMPTS; j++)
                    std::cerr << std::setw(13) << gp[j];
                std::cerr << std::endl;
                num_left--;
            }
        }
    }
    
    std::cerr << "Gradient sanity check complete." << std::endl;
}
