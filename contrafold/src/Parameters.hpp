/////////////////////////////////////////////////////////////////
// Parameters.hpp
/////////////////////////////////////////////////////////////////

#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include "Config.hpp"
#include "Utilities.hpp"

/////////////////////////////////////////////////////////////////
// struct HyperparameterGroup
/////////////////////////////////////////////////////////////////

struct HyperparameterGroup
{
    std::string name;
    int begin, end;
    
    HyperparameterGroup();
    HyperparameterGroup(const std::string &name, int begin, int end);
    HyperparameterGroup(const HyperparameterGroup &rhs);  
    HyperparameterGroup &operator=(const HyperparameterGroup &rhs);
};

/////////////////////////////////////////////////////////////////
// class Parameters
/////////////////////////////////////////////////////////////////

class Parameters
{
    std::vector<double> values;
    std::vector<std::string> names;
    std::vector<HyperparameterGroup> groups;
    std::map<std::string, int> name_to_index;
    
    void BeginGroup(const std::string &name);
    void NewParameter(const std::string &s);
    void AddParameterAlias(const std::string &alias, const std::string &name);
    
public:
    
    Parameters();
    Parameters(const Parameters &rhs);
    Parameters &operator= (const Parameters &rhs);
    ~Parameters();
    
    std::vector<double> GetRandomValues();
    std::vector<double> GetDefaultValues();
    std::vector<double> GetDefaultComplementaryValues();
    
    size_t GetNumParameters() const;
    const std::vector<double> GetValues() const;
    const std::vector<std::string> GetNames() const;
    std::vector<double> ReadFromFile(const std::string &filename);
    void WriteToFile(const std::string &filename, const std::vector<double> &values);
    
    int GetParamIndex(const std::string &s) const;
    
    std::vector<double> ExpandHyperparameters(const std::vector<double> &v) const;
    const std::vector<HyperparameterGroup> &GetHyperparameterGroups() const;
    size_t GetNumHyperparameterGroups() const;
};

#endif
