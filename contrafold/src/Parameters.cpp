/////////////////////////////////////////////////////////////////
// Parameters.cpp
/////////////////////////////////////////////////////////////////

#include "Parameters.hpp"
#include "Defaults.hpp"

/////////////////////////////////////////////////////////////////
// HyperparameterGroup::HyperparameterGroup()
// 
// Default constructor.
/////////////////////////////////////////////////////////////////

HyperparameterGroup::HyperparameterGroup() {}

/////////////////////////////////////////////////////////////////
// HyperparameterGroup::HyperparameterGroup()
// 
// Constructor.
/////////////////////////////////////////////////////////////////

HyperparameterGroup::HyperparameterGroup (const std::string &name, int begin, int end) :
    name(name), begin(begin), end(end) {}

/////////////////////////////////////////////////////////////////
// HyperparameterGroup::HyperparameterGroup()
// 
// Copy constructor.
/////////////////////////////////////////////////////////////////

HyperparameterGroup::HyperparameterGroup (const HyperparameterGroup &rhs) :
    name(rhs.name), begin(rhs.begin), end(rhs.end) {}

/////////////////////////////////////////////////////////////////
// HyperparameterGroup::operator=()
// 
// Assignment operator.
/////////////////////////////////////////////////////////////////

HyperparameterGroup &HyperparameterGroup::operator= (const HyperparameterGroup &rhs)
{
    if (this != &rhs)
    {
        name = rhs.name;
        begin = rhs.begin;
        end = rhs.end;
    }
    return *this;
}

/////////////////////////////////////////////////////////////////
// Parameters::Parameters()
//
// Constructor.
/////////////////////////////////////////////////////////////////

Parameters::Parameters() : names(0)
{
    char buffer[1000];
    char buffer2[1000];
    
#if SINGLE_HYPERPARAMETER
    BeginGroup("all_params");
#endif
    
#if PARAMS_BASE_PAIR
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("base_pair");
#endif
    for (int i = 0; i < M; i++)
    {
        for (int j = i; j < M; j++)
        {
            sprintf(buffer, "base_pair_%c%c", alphabet[i], alphabet[j]);
            NewParameter(buffer);
            if (i < j)
            {
                sprintf(buffer2, "base_pair_%c%c", alphabet[j], alphabet[i]);
                AddParameterAlias(buffer2, buffer);
            }
        }
    }
#endif

#if PARAMS_BASE_PAIR_DIST
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("base_pair_dist_at_least");
#endif
    for (int i = 0; i < D_MAX_BP_DIST_THRESHOLDS; i++){
        sprintf(buffer, "base_pair_dist_at_least_%d", BP_DIST_THRESHOLDS[i]);
        NewParameter(buffer);
    }
#endif
    
#if PARAMS_TERMINAL_MISMATCH
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("terminal_mismatch");
#endif
    for (int i1 = 0; i1 < M; i1++)
    {
        for (int j1 = 0; j1 < M; j1++)
        {
            for (int i2 = 0; i2 < M; i2++)
            {
                for (int j2 = 0; j2 < M; j2++)
                {
                    sprintf(buffer, "terminal_mismatch_%c%c%c%c",
                            alphabet[i1], alphabet[j1],
                            alphabet[i2], alphabet[j2]);
                    NewParameter(buffer);
                }
            }
        }
    }
#endif
    
#if PARAMS_HAIRPIN_LENGTH
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("hairpin_length_at_least");
#endif
    for (int i = 0; i <= D_MAX_HAIRPIN_LENGTH; i++)
    {
        sprintf(buffer, "hairpin_length_at_least_%d", i);
        NewParameter(buffer);
    }
    
#endif
    
#if PARAMS_HAIRPIN_3_NUCLEOTIDES
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("hairpin_3_nucleotides");  
#endif
    for (int i1 = 0; i1 < M; i1++)
    {
        for (int i2 = 0; i2 < M; i2++)
        {
            for (int i3 = 0; i3 < M; i3++)
            {
                sprintf(buffer, "hairpin_3_nucleotides_%c%c%c",
                        alphabet[i1], alphabet[i2], alphabet[i3]);
                NewParameter(buffer);
            }
        }
    }
#endif
    
#if PARAMS_HAIRPIN_4_NUCLEOTIDES
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("hairpin_4_nucleotides");  
#endif
    for (int i1 = 0; i1 < M; i1++)
    {
        for (int i2 = 0; i2 < M; i2++)
        {
            for (int i3 = 0; i3 < M; i3++)
            {
                for (int i4 = 0; i4 < M; i4++)
                {
                    sprintf(buffer, "hairpin_4_nucleotides_%c%c%c%c",
                            alphabet[i1], alphabet[i2], alphabet[i3], alphabet[i4]);
                    NewParameter(buffer);
                }
            }
        }
    }
#endif

#if PARAMS_HELIX_LENGTH
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("helix_length_at_least");
#endif
    for (int k = 3; k <= D_MAX_HELIX_LENGTH; k++)
    {
        sprintf(buffer, "helix_length_at_least_%d", k);
        NewParameter(buffer);
    }
#endif
  
#if PARAMS_ISOLATED_BASE_PAIR
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("isolated_base_pair");
#endif
    NewParameter("isolated_base_pair");
#endif
    
#if PARAMS_INTERNAL_EXPLICIT
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("internal_explicit");
#endif
    for (int i = 1; i <= D_MAX_INTERNAL_EXPLICIT_LENGTH; i++)
    {
        for (int j = i; j <= D_MAX_INTERNAL_EXPLICIT_LENGTH; j++)
        {
            sprintf(buffer, "internal_explicit_%d_%d", i, j);
            NewParameter(buffer);
            if (i < j)
            {
                sprintf(buffer2, "internal_explicit_%d_%d", j, i);
                AddParameterAlias(buffer2, buffer);
            }
        }
    }
#endif
    
#if PARAMS_BULGE_LENGTH
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("bulge_length_at_least");
#endif
    for (int i = 1; i <= D_MAX_BULGE_LENGTH; i++)
    {
        sprintf(buffer, "bulge_length_at_least_%d", i);
        NewParameter(buffer);
    }
#endif
    
#if PARAMS_INTERNAL_LENGTH
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("internal_length_at_least");
#endif
    for (int i = 2; i <= D_MAX_INTERNAL_LENGTH; i++)
    {
        sprintf(buffer, "internal_length_at_least_%d", i);
        NewParameter(buffer);
    }
#endif
    
#if PARAMS_INTERNAL_SYMMETRY
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("internal_symmetric_length_at_least");
#endif
    for (int i = 1; i <= D_MAX_INTERNAL_SYMMETRIC_LENGTH; i++)
    {
        sprintf(buffer, "internal_symmetric_length_at_least_%d", i);
        NewParameter(buffer);
    }
#endif
    
#if PARAMS_INTERNAL_ASYMMETRY
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("internal_asymmetry_at_least");
#endif
    for (int i = 1; i <= D_MAX_INTERNAL_ASYMMETRY; i++)
    {
        sprintf(buffer, "internal_asymmetry_at_least_%d", i);
        NewParameter(buffer);
    }
#endif

#if PARAMS_BULGE_0x1_NUCLEOTIDES
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("bulge_0x1_nucleotides");
#endif
    for (int i1 = 0; i1 < M; i1++)
    {
        sprintf(buffer, "bulge_0x1_nucleotides_%c", 
                alphabet[i1]);
        NewParameter(buffer);
        sprintf(buffer2, "bulge_1x0_nucleotides_%c", 
                alphabet[i1]);
        AddParameterAlias(buffer2, buffer);
    }
#endif
    
#if PARAMS_BULGE_0x2_NUCLEOTIDES
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("bulge_0x2_nucleotides");
#endif
    for (int i1 = 0; i1 < M; i1++)
    {
        for (int i2 = 0; i2 < M; i2++)
        {
            sprintf(buffer, "bulge_0x2_nucleotides_%c%c", 
                    alphabet[i1], alphabet[i2]);
            NewParameter(buffer);
            sprintf(buffer2, "bulge_2x0_nucleotides_%c%c",
                    alphabet[i1], alphabet[i2]);
            AddParameterAlias(buffer2, buffer);
        }
    }
#endif
    
#if PARAMS_BULGE_0x3_NUCLEOTIDES
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("bulge_0x3_nucleotides");
#endif
    for (int i1 = 0; i1 < M; i1++)
    {
        for (int i2 = 0; i2 < M; i2++)
        {
            for (int i3 = 0; i3 < M; i3++)
            {
                sprintf(buffer, "bulge_0x3_nucleotides_%c%c%c", 
                        alphabet[i1], alphabet[i2], alphabet[i3]);
                NewParameter(buffer);
                sprintf(buffer2, "bulge_3x0_nucleotides_%c%c%c", 
                        alphabet[i1], alphabet[i2], alphabet[i3]);
                AddParameterAlias(buffer2, buffer);
            }
        }
    }
#endif
    
#if PARAMS_INTERNAL_1x1_NUCLEOTIDES
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("internal_1x1_nucleotides");
#endif
    for (int i1 = 0; i1 < M; i1++)
    {
        for (int i2 = 0; i2 < M; i2++)
        {
            sprintf(buffer, "internal_1x1_nucleotides_%c%c", 
                    alphabet[i1], alphabet[i2]);
            NewParameter(buffer);
        }
    }
#endif
    
#if PARAMS_INTERNAL_1x2_NUCLEOTIDES
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("internal_1x2_nucleotides");
#endif
    for (int i1 = 0; i1 < M; i1++)
    {
        for (int i2 = 0; i2 < M; i2++)
        {
            for (int i3 = 0; i3 < M; i3++)
            {
                sprintf(buffer, "internal_1x2_nucleotides_%c%c%c", 
                        alphabet[i1], alphabet[i2], alphabet[i3]);
                NewParameter(buffer);
                sprintf(buffer2, "internal_2x1_nucleotides_%c%c%c", 
                        alphabet[i2], alphabet[i3], alphabet[i1]);
                AddParameterAlias(buffer2, buffer);
            }
        }
    }
#endif

#if PARAMS_INTERNAL_2x2_NUCLEOTIDES
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("internal_2x2_nucleotides");
#endif
    for (int i1 = 0; i1 < M; i1++)
    {
        for (int i2 = 0; i2 < M; i2++)
        {
            for (int i3 = 0; i3 < M; i3++)
            {
                for (int i4 = 0; i4 < M; i4++)
                {
                    sprintf(buffer, "internal_2x2_nucleotides_%c%c%c%c", 
                            alphabet[i1], alphabet[i2], alphabet[i3], alphabet[i4]);
                    NewParameter(buffer);
                }
            }
        }
    }
#endif
    
#if PARAMS_HELIX_STACKING
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("helix_stacking");
#endif
    
    for (int i1 = 0; i1 < M; i1++)
    {
        for (int j1 = 0; j1 < M; j1++)
        {
            for (int i2 = 0; i2 < M; i2++)
            {
                for (int j2 = 0; j2 < M; j2++)
                {
                    sprintf(buffer, "helix_stacking_%c%c%c%c", 
                            alphabet[i1], alphabet[j1],
                            alphabet[i2], alphabet[j2]);
                    sprintf(buffer2, "helix_stacking_%c%c%c%c",
                            alphabet[j2], alphabet[i2],
                            alphabet[j1], alphabet[i1]);
                    
                    if (strcmp(buffer, buffer2) < 0)
                    {
                        NewParameter(buffer);
                        AddParameterAlias(buffer2, buffer);
                    }
                    else if (!strcmp(buffer, buffer2))
                    {
                        NewParameter(buffer);
                    }
                }
            }
        }
    }
#endif
    
#if PARAMS_HELIX_CLOSING
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("helix_closing");
#endif
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < M; j++)
        {
            sprintf(buffer, "helix_closing_%c%c", alphabet[i], alphabet[j]);
            NewParameter(buffer);
        }
    }
#endif
    
#if PARAMS_MULTI_LENGTH
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("multi_length");
#endif
    NewParameter("multi_base");
    NewParameter("multi_unpaired");
    NewParameter("multi_paired");
#endif
    
#if PARAMS_DANGLE
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("dangle");
    //BeginGroup("dangle_left");
#endif
    for (int i1 = 0; i1 < M; i1++)
    {
        for (int j1 = 0; j1 < M; j1++)
        {
            for (int i2 = 0; i2 < M; i2++)
            {
                sprintf(buffer, "dangle_left_%c%c%c",
                        alphabet[i1], alphabet[j1], alphabet[i2]);
                NewParameter(buffer);
            }
        }
    }
    
#if MULTIPLE_HYPERPARAMETERS
  //BeginGroup("dangle_right");
#endif
    for (int i1 = 0; i1 < M; i1++)
    {
        for (int j1 = 0; j1 < M; j1++)
        {
            for (int i2 = 0; i2 < M; i2++)
            {
                sprintf(buffer, "dangle_right_%c%c%c",
                        alphabet[i1], alphabet[j1], alphabet[i2]);
                NewParameter(buffer);
            }
        }
    }
#endif

#if PARAMS_EXTERNAL_LENGTH
#if MULTIPLE_HYPERPARAMETERS
    BeginGroup("external");
#endif
    NewParameter("external_unpaired");
    NewParameter("external_paired");
#endif

}

/////////////////////////////////////////////////////////////////
// Parameters::Parameters()
//
// Copy constructor.
/////////////////////////////////////////////////////////////////

Parameters::Parameters(const Parameters &rhs) :
    names(rhs.names), groups(rhs.groups),
    name_to_index(rhs.name_to_index) {}

/////////////////////////////////////////////////////////////////
// Parameters::operator=()
//
// Assignment operator.
/////////////////////////////////////////////////////////////////

Parameters &Parameters::operator= (const Parameters &rhs){
    if (this != &rhs)
    {
        names = rhs.names;
        groups = rhs.groups;
        name_to_index = rhs.name_to_index;
    }
    return *this;
}

/////////////////////////////////////////////////////////////////
// Parameters::~Parameters()
//
// Destructor.
/////////////////////////////////////////////////////////////////

Parameters::~Parameters() {}

/////////////////////////////////////////////////////////////////
// Parameters::GetRandomValues()
//
// Get vector of small random values.
/////////////////////////////////////////////////////////////////

std::vector<double> Parameters::GetRandomValues(){
    std::vector<double> values(names.size());
  
  // initialize random number generator to fixed seed, for reproducibility

    srand48(0);  
    
    for (size_t i = 0; i < values.size(); i++)
    {
        values[i] = 0;
    }
    return values;
}

/////////////////////////////////////////////////////////////////
// Parameters::GetValues();
//
// Retrieve values from Parameter object.
/////////////////////////////////////////////////////////////////

const std::vector<double> Parameters::GetValues() const
{
    return values;
}

/////////////////////////////////////////////////////////////////
// Parameters::GetNames()
//
// Retrieve names from Parameter object.
/////////////////////////////////////////////////////////////////

const std::vector<std::string> Parameters::GetNames() const
{
    return names;
}

/////////////////////////////////////////////////////////////////
// Parameters::ReadFromFile()
//
// Read parameters from file.
/////////////////////////////////////////////////////////////////

std::vector<double> Parameters::ReadFromFile(const std::string &filename)
{
    std::map<std::string, double> params;
    double value;
    std::string name;
    std::string s;
    
    std::ifstream infile(filename.c_str());
    if (infile.fail()) Error(("Could not open file \"" + filename + "\" for reading.").c_str());
    values.clear();
    while (getline (infile, s))
    {
        if (s.length() == 0 || s[0] == '#') continue;
        std::istringstream iss(s);
        if (iss >> name >> value)
        {
            if (params.find ("name") != params.end())
                Error("Parameter file contains a duplicate parameter: %s", name.c_str());
            params[name] = value;
        }
    }
    infile.close();
    
    std::vector<double> values(names.size());
    for (size_t i = 0; i < names.size(); i++)
    {
        std::map<std::string, double>::iterator iter = params.find(names[i]);
        if (iter == params.end())
            Error("Parameter file missing parameter: %s", names[i].c_str());
        values[i] = iter->second;
        params.erase (iter);
    }
    
    for (std::map<std::string, double>::iterator iter = params.begin(); iter != params.end(); ++iter)
        Warning ("Parameter file contains extra parameter: %s", iter->first.c_str());

    return values;
}

/////////////////////////////////////////////////////////////////
// Parameters::WriteToFile()
//
// Write parameters to file.
/////////////////////////////////////////////////////////////////

void Parameters::WriteToFile(const std::string &filename, const std::vector<double> &values)
{
    std::ofstream outfile (filename.c_str());
    if (outfile.fail()) Error(("Could not open file \"" + filename + "\" for writing.").c_str());
    for (size_t i = 0; i < values.size(); i++)
        outfile << names[i] << " " << std::setprecision(10) << values[i] << std::endl;
    outfile.close();
}

/////////////////////////////////////////////////////////////////
// Parameters::ExpandHyperparameters()
//
// Expand a hyperparameter vector.
/////////////////////////////////////////////////////////////////

std::vector<double> Parameters::ExpandHyperparameters (const std::vector<double> &v) const
{
    std::vector<double> expanded;
    Assert(v.size() == groups.size(), "Incorrect number of hyperparametrs.");
    for (size_t i = 0; i < groups.size(); i++)
    {
        for (int j = 0; j < groups[i].end - groups[i].begin + 1; j++)
            expanded.push_back (v[i]);
    }
    return expanded;
}

/////////////////////////////////////////////////////////////////
// Parameters::BeginGroup()
//
// Mark the beginning of a new hyperparameter group.
/////////////////////////////////////////////////////////////////

void Parameters::BeginGroup(const std::string &name)
{
    groups.push_back (HyperparameterGroup (name, names.size(), int(names.size()) - 1));
}

/////////////////////////////////////////////////////////////////
// Parameters::NewParameter()
//
// Create new parameter.
/////////////////////////////////////////////////////////////////

void Parameters::NewParameter(const std::string &s)
{
    std::map<std::string,int>::iterator iter = name_to_index.find (s);
    Assert(iter == name_to_index.end(), "Attempt to create parameter with duplicate name: %s", s.c_str());
    
#if ARD_HYPERPARAMETERS
    BeginGroup(s);
#endif
    
    name_to_index[s] = names.size();
    names.push_back (s);
    ++(groups.back().end);
}

/////////////////////////////////////////////////////////////////
// Parameters::AddParameterAlias()
//
// Create new alias for parameter.
/////////////////////////////////////////////////////////////////

void Parameters::AddParameterAlias(const std::string &alias, const std::string &name)
{
    std::map<std::string,int>::iterator iter = name_to_index.find (alias);
    Assert(iter == name_to_index.end(), "Attempt to create parameter with duplicate name: %s", name.c_str());
    std::map<std::string,int>::iterator iter2 = name_to_index.find (name);
    Assert(iter2 != name_to_index.end(), "Request for unknown parameter: %s", name.c_str());
    name_to_index[alias] = iter2->second;
}

/////////////////////////////////////////////////////////////////
// Parameters::GetNumParameters()
//
// Retrieve number of parameters from Parameter object.
/////////////////////////////////////////////////////////////////

size_t Parameters::GetNumParameters() const
{
    return names.size();
}

/////////////////////////////////////////////////////////////////
// Parameters::GetParamIndex()
//
// Retrieve the index for a particular parameter.
/////////////////////////////////////////////////////////////////

int Parameters::GetParamIndex (const std::string &s) const
{
    std::map<std::string,int>::const_iterator iter = name_to_index.find (s);
    Assert(iter != name_to_index.end(), "Request for unknown parameter: %s.", s.c_str());
    return iter->second;
}

/////////////////////////////////////////////////////////////////
// Parameters::GetHyperparameterGroups()
//
// Retrieve hyperparameter groups.
/////////////////////////////////////////////////////////////////

const std::vector<HyperparameterGroup> &Parameters::GetHyperparameterGroups() const
{
    return groups;
}

/////////////////////////////////////////////////////////////////
// Parameters::GetNumHyperparameterGroups()
//
// Retrieve number of hyperparameter groups.
/////////////////////////////////////////////////////////////////

size_t Parameters::GetNumHyperparameterGroups() const
{
    return groups.size();
}

