/////////////////////////////////////////////////////////////////
// DistributedComputation.cpp
//
// Class for performing distributed optimization.
/////////////////////////////////////////////////////////////////

#include "DistributedComputation.hpp"

/////////////////////////////////////////////////////////////////
// WorkUnitComparator()
//
// Comparator to allow sorting of pointers to WorkUnit 
// objects in decreasing order of estimated time.
/////////////////////////////////////////////////////////////////

bool WorkUnitComparator::operator()(const WorkUnit *lhs, const WorkUnit *rhs) const
{
    return lhs->GetEstimatedTime() > rhs->GetEstimatedTime();
}

/////////////////////////////////////////////////////////////////
// DistributedComputation::DistributedComputation()
//
// Constructor.  Performs MPI initializations if MULTI is
// defined.  "Fixes" argc, argv as needed for parameter parsing.
/////////////////////////////////////////////////////////////////

DistributedComputation::DistributedComputation(bool toggle_verbose) : 
    toggle_verbose(toggle_verbose), processing_time(0), total_time(0), id(0), num_procs(1)
{
#ifdef MULTI
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    if (id == 0 && toggle_verbose)
    {
        WriteProgressMessage("");
        std::cerr << "Distributed Optimization Library started.  Using " 
                  << num_procs << " processor(s)." << std::endl;
    }
}

/////////////////////////////////////////////////////////////////
// DistributedComputation::StopComputeNodes()
//
// Destructor.  Tells all compute nodes to terminate ifdef MULTI is 
// defined.  Shuts down MPI.
/////////////////////////////////////////////////////////////////

void DistributedComputation::StopComputeNodes()
{
#ifdef MULTI
    Assert(id == 0, "Routine should only be called by master process.");
    for (int i = 1; i < num_procs; i++)
    {
        int command = CommandType_Quit;
        MPI_Send(&command, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
#endif    
}

/////////////////////////////////////////////////////////////////
// DistributedComputation::RunComputeNode()
//
// Turn into a compute node and process work requests from the 
// master node until the command to quit is sent.  Should only
// be called ifdef MULTI is defined.
/////////////////////////////////////////////////////////////////

void DistributedComputation::RunComputeNode()
{
    Assert(id != 0, "Routine should not be called by master process.");
    
#ifdef MULTI
    MPI_Status status;
    std::vector<double> params;
    std::vector<double> result;
    std::vector<double> partial_result;
    char *buffer;
    
    while (true)
    {
        // block until command received
        
        int command;
        MPI_Recv(&command, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        switch (command){
            
            case CommandType_LoadParameters:
            {
                // get parameter vector size
                
                int size;
                MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
                Assert(size >= 0, "Parameter vector size should be nonnegative.");
                
                // get parameter vector
                
                params.resize(size);
                MPI_Bcast(&params[0], size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
            break;
            
            case CommandType_DoWork:
            {
                // get size of work description object in bytes
                
                int size;
                MPI_Recv(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
                Assert(size > 0, "Work description size should be positive.");
                
                // obtain the work description object
                
                buffer = new char[size];
                MPI_Recv(buffer, size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
                
                // perform computation
                
                processing_time = GetSystemTime();
                DoComputation(partial_result, reinterpret_cast<WorkUnit *>(buffer), params);
                processing_time = GetSystemTime() - processing_time;
                
                // accumulate results
                
                size = int(partial_result.size());
                if (result.size() == 0) result.resize(size);
                Assert(result.size() == size, "Return values of different size.");      
                for (size_t i = 0; i < result.size(); i++)
                {
                    if (std::isnan(partial_result[i]))
                    {
                        Error(SPrintF("Encountered NaN value during computation associated with \"%s\".",
                                      reinterpret_cast<WorkUnit *>(buffer)->GetSummary().c_str()).c_str());
                    }
                    result[i] += partial_result[i];
                }
                
                // return processing time to main node
                
                MPI_Send(&processing_time, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
            break;
            
            case CommandType_GetResultSize:
            {
                // send result size to main node
                
                int size = int(result.size());
                MPI_Send(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
            break;
            
            case CommandType_GetResult:
            {
                // make sure all results are of the same length first
                
                int size;
                MPI_Recv(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
                
                // then send result to main node
                
                if (result.size() == 0) result.resize(size);
                Assert(result.size() == size, "Return values of different size.");      
                if (result.size() > 0) MPI_Reduce(&result[0], NULL, result.size(), 
                                                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            }
            break;
            
            case CommandType_ClearResult:
                result.clear();
                break;
                
            case CommandType_Quit:
                return;
                
        }      
    }
#endif

}

/////////////////////////////////////////////////////////////////
// DistributedComputation::RunMasterNode()
//
// Compute the aggregated result for a list of work description
// units.  Distributes tasks among all nodes (other than 0) if
// MULTI is defined; work units are allocated starting from 
// largest unit size to smallest unit size.
/////////////////////////////////////////////////////////////////

void DistributedComputation::RunMasterNode(std::vector<double> &ret, 
                                           std::vector<WorkUnit *> descriptions, 
                                           std::vector<double> params)
{
    Assert(id == 0, "Routine should only be called by master process.");
    Assert(descriptions.size() > 0, "Must submit at least one work description for processing.");
    
    double starting_time = GetSystemTime();
    int units_complete = 0;
    
#ifdef MULTI
    
    MPI_Status status;
    int command;
    int size;
    
    std::string progress;
    
    int num_procs_in_use = 1;
    int curr_unit = 0;
    
    // sort work units in order of decreasing size
    
    std::vector<int> assignment(num_procs, -1);
    assignment[0] = descriptions.size();
    std::sort(descriptions.begin(), descriptions.end(), WorkUnitComparator());
    
    if (toggle_verbose)
        WriteProgressMessage("Clearing accumulated result on all processors.");
    
    // clear accumulated result on all processors
    
    command = CommandType_ClearResult;
    for (int proc = 1; proc < num_procs; proc++)
    {
        MPI_Send(&command, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
    }
    
    if (toggle_verbose)
        WriteProgressMessage("Broadcasting parameter vector to all processors.");
    
    // broadcast parameter vector to all processors
    
    command = CommandType_LoadParameters;
    for (int proc = 1; proc < num_procs; proc++)
        MPI_Send(&command, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
    
    size = params.size();
    MPI_Bcast(&size, 1, MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&params[0], size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (toggle_verbose)
        WriteProgressMessage("Sending work units to all processors.");
    
    // while there is work to be done
    
    while (num_procs_in_use > 1 || curr_unit < int(descriptions.size()))
    {
        
        // allocate the max number of processors possible
        
        while (num_procs_in_use < num_procs && curr_unit < int(descriptions.size()))
        {
            
            int proc = 0;
            while (proc < int(assignment.size()) && assignment[proc] >= 0) proc++;
            Assert(proc < int(assignment.size()), "Expected to find free processor.");
            
            // send command
            
            command = CommandType_DoWork;
            MPI_Send(&command, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
            
            // send work description
            
            int size = descriptions[curr_unit]->GetDescriptionSize();
            MPI_Send(&size, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
            MPI_Send(descriptions[curr_unit], size, MPI_CHAR, proc, 0, MPI_COMM_WORLD);
            
            num_procs_in_use++;
            assignment[proc] = curr_unit;
            curr_unit++;
        }
        
        // write progress
        
        double current_time = GetSystemTime();
        static double prev_reporting_time = 0;
        
        if (current_time - prev_reporting_time > 1)
        {
            prev_reporting_time = current_time;
            int percent_complete = 100 * units_complete / descriptions.size();
            if (toggle_verbose)
                WriteProgressMessage(SPrintF("Work units %d%% complete.", percent_complete));
        }
        
        // if no processors left, or all work allocated, then wait for results
        
        MPI_Recv(&current_time, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        Assert(current_time > 0, "Expected positive time value for acknowledgment of job completion.");
        processing_time += current_time;
        
        num_procs_in_use--;
        assignment[status.MPI_SOURCE] = -1;
        units_complete++;
    }
    
    if (toggle_verbose)
        WriteProgressMessage("Computing result size.");
    
    // get accumulated result size
    
    command = CommandType_GetResultSize;
    MPI_Send(&command, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    MPI_Recv(&size, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
    Assert(size >= 0, "Expected nonnegative return value size.");
    
    if (toggle_verbose)
        WriteProgressMessage("Requesting results from processors.");
    
    // get accumulated result
    
    command = CommandType_GetResult;
    for (int proc = 1; proc < num_procs; proc++)
    {
        MPI_Send(&command, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
        MPI_Send(&size, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
    }
    
    if (toggle_verbose)
        WriteProgressMessage("Receiving accumulated results from processors.");
    
    ret.clear();
    ret.resize(size);
    std::vector<double> zeros(size);
    if (size > 0) MPI_Reduce(&zeros[0], &ret[0], size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
#else
    
    ret.clear();
    std::vector<double> partial_ret;
    
    if (toggle_verbose)
        WriteProgressMessage("Starting first work unit.");
    
    for (size_t j = 0; j < descriptions.size(); j++)
    {
        DoComputation(partial_ret, descriptions[j], params);
        
        if (ret.size() == 0)
        {
            ret.resize(partial_ret.size());
        }
        else
        {
            Assert(ret.size() == partial_ret.size(), "Return values of different size.");
        }
    
        // accumulate results
        
        for (size_t i = 0; i < ret.size(); i++)
        {
            if (std::isnan(partial_ret[i]))
            {
                Error(SPrintF("Encountered NaN value during computation associated with \"%s\".",
                              descriptions[j]->GetSummary().c_str()).c_str());
            }
            ret[i] += partial_ret[i];
        }
        
        units_complete++;
        
        // write progress
        
        double current_time = GetSystemTime();
        static double prev_reporting_time = 0;
        if (current_time - prev_reporting_time > 1)
        {
            prev_reporting_time = current_time;
            int percent_complete = 100 * units_complete / descriptions.size();
            if (toggle_verbose)
                WriteProgressMessage(SPrintF("Work units %d%% complete.", percent_complete));
        }
    }
    
#endif
    
    if (toggle_verbose)
        WriteProgressMessage("");
    total_time += (GetSystemTime() - starting_time);
}

/////////////////////////////////////////////////////////////////
// DistributedComputation::IsComputeNode()
//
// Returns true if current process is a compute node.
/////////////////////////////////////////////////////////////////

bool DistributedComputation::IsComputeNode() const
{
    return (id != 0);
}

/////////////////////////////////////////////////////////////////
// DistributedComputation::IsMasterNode()
//
// Returns true if current process is the master node.
/////////////////////////////////////////////////////////////////

bool DistributedComputation::IsMasterNode() const
{
    return (id == 0);
}

/////////////////////////////////////////////////////////////////
// DistributedComputation::GetNumNodes()
//
// Returns the total number of nodes.
/////////////////////////////////////////////////////////////////

int DistributedComputation::GetNumNodes() const
{
    return num_procs;
}

/////////////////////////////////////////////////////////////////
// DistributedComputation::GetNodeID()
//
// Returns the id of the current node.
/////////////////////////////////////////////////////////////////

int DistributedComputation::GetNodeID() const
{
    return id;
}

/////////////////////////////////////////////////////////////////
// DistributedComputation::GetEfficiency()
//
// Compute the processor usage efficiency.
/////////////////////////////////////////////////////////////////

double DistributedComputation::GetEfficiency() const
{
    Assert(id == 0, "Routine should only be called by master process.");
#ifdef MULTI
    return 100.0 * (processing_time / (num_procs - 1)) / (1e-10 + total_time);
#else
    return 100.0;
#endif
}

/////////////////////////////////////////////////////////////////
// DistributedComputation::GetElapsedTime()
//
// Compute time elapsed since this function was last called.
/////////////////////////////////////////////////////////////////

double DistributedComputation::GetElapsedTime()
{
    Assert(id == 0, "Routine should only be called by master process.");
    static double last_time = 0;
    double ret = total_time - last_time;
    last_time = total_time;
    return ret;
}
