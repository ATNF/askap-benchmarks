/*! \file ThreadAffinityUtil.cpp
 *  \brief Get thread to core affinity
 */

#include "gpuCommon.h"
#include "LoggerUtil.h"

namespace logger_util {

    /*
    Code to facilitate core binding reporting
    Borrowed from VELOCIraptor, which itself
    borrowed from util-linux-2.13-pre7/schedutils/taskset.c
    */
    #ifdef __APPLE__

    static inline void
    CPU_ZERO(cpu_set_t *cs) { cs->count = 0; }

    static inline void
    CPU_SET(int num, cpu_set_t *cs) { cs->count |= (1 << num); }

    static inline int
    CPU_ISSET(int num, cpu_set_t *cs) { return (cs->count & (1 << num)); }

    int sched_getaffinity(pid_t pid, size_t cpu_size, cpu_set_t *cpu_set)
    {
        int32_t core_count = 0;
        size_t  len = sizeof(core_count);
        int ret = sysctlbyname(SYSCTL_CORE_COUNT, &core_count, &len, 0, 0);
        if (ret) {
            printf("error while get core count %d\n", ret);
            return -1;
        }
        cpu_set->count = 0;
        for (int i = 0; i < core_count; i++) cpu_set->count |= (1 << i);
        return 0;
    }
    #endif

    void cpuset_to_cstr(cpu_set_t *mask, char *str)
    {
        char *ptr = str;
        int i, j, entry_made = 0;
        for (i = 0; i < CPU_SETSIZE; i++) {
            if (CPU_ISSET(i, mask)) {
                int run = 0;
                entry_made = 1;
                for (j = i + 1; j < CPU_SETSIZE; j++) {
                    if (CPU_ISSET(j, mask)) run++;
                    else break;
                }
                if (!run) {
                    sprintf(ptr, "%d ", i);
                }
                else if (run == 1) {
                    sprintf(ptr, "%d,%d ", i, i + 1);
                    i++;
                } else {
                    sprintf(ptr, "%d-%d ", i, i + run);
                    i += run;
                }
                while (*ptr != 0) ptr++;
            }
        }
        ptr -= entry_made;
        ptr = nullptr;
    }

    std::string MPICallingRank(int task){
        char s[20];
        sprintf(s,"MPI [%04d]: ",task);
        return std::string(s);
    }

    std::string ReportParallelAPI() 
    {
        std::string s;
        s = "Parallel API's \n ======== \n";
#ifdef _MPI
        int rank, size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        s += "MPI Comm world size " + std::to_string(size);
        s += "\n";
#endif 
#ifdef _OPENMP 
        s += "OpenMP version " + std::to_string(_OPENMP);
        s += " with total number of threads = " + std::to_string(omp_get_max_threads());
        s += " with total number of allowed levels " + std::to_string(omp_get_max_active_levels());
#ifdef _WITH_GPU
        int numdevices = omp_get_num_devices();
        int defaultdevice = omp_get_default_device();
        int ninfo[2];
        if (numdevices > 0) 
        {
            #pragma omp target map(tofrom:ninfo)
            {
                int team = omp_get_team_num();
                int tid = omp_get_thread_num();
                if (tid == 0 && team == 0)
                {
                    auto nteams = omp_get_num_teams();
                    auto nthreads = omp_get_num_threads();
                    ninfo[0] = nteams;
                    ninfo[1] = nthreads;
                }
            }
            s += "\n";
            s += "OpenMP Target : ";
            s += "Number of devices "+ std::to_string(numdevices);
            s += "Default device "+ std::to_string(defaultdevice);
            s += "Number of Compute Units "+ std::to_string(ninfo[1]);
        }
#endif
        s += "\n";
#endif
        int nDevices = 0;
        gpuErrorCheck(gpuGetDeviceCount(&nDevices));
        s += "Using GPUs: Running with " +std::string(__GPU_API__) + " and found " + std::to_string(nDevices) + " devices\n";
        return s;
    }

    std::string ReportBinding()
    {
        std::string binding_report;
        int ThisTask=0, NProcs=1;
#ifdef _MPI
        MPI_Comm_size(MPI_COMM_WORLD, &NProcs);
        MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
#endif
        if (ThisTask == 0) binding_report = "Core Binding \n ======== \n";
        cpu_set_t coremask;
        char clbuf[7 * CPU_SETSIZE], hnbuf[64];
        memset(clbuf, 0, sizeof(clbuf));
        memset(hnbuf, 0, sizeof(hnbuf));
        (void)gethostname(hnbuf, sizeof(hnbuf));
        std::string result;
        result = "\t On node " + std::string(hnbuf) + " : ";
#ifdef _MPI
        result += "MPI Rank " + std::to_string(ThisTask) + " : ";
#endif
#ifdef _OPENMP
        #pragma omp parallel \
        default(none) shared(binding_report, hnbuf, ThisTask) \
        private(coremask, clbuf) \
        firstprivate(result)
#endif
        {
            (void)sched_getaffinity(0, sizeof(coremask), &coremask);
            cpuset_to_cstr(&coremask, clbuf);
#ifdef _OPENMP
            auto thread = omp_get_thread_num();
            auto level = omp_get_level();
            result +=" OMP Thread " + std::to_string(thread) + " : ";
            result +=" at nested level " + std::to_string(level) + " : ";
#endif
            result += " Core affinity = " + std::string(clbuf) + " \n ";
#ifdef _OPENMP
            #pragma omp critical
#endif
            {
                binding_report += result;
            }
        }
        int nDevices = 0;
        gpuErrorCheck(gpuGetDeviceCount(&nDevices));
        if (nDevices > 0) {
            // binding_report += std::string(_GPU_API) + " API ";
            // binding_report += std::to_string(nDevices) + " devices, device info : \n";
            char busid[64];
            for (auto i=0;i<nDevices;i++)
            {
                gpuDeviceProp_t prop;
                std::string s;
                // gpuErrorCheck(gpuSetDevice(i));
                gpuErrorCheck(gpuGetDeviceProperties(&prop, i));
                // Get the PCIBusId for each GPU and use it to query for UUID
                gpuErrorCheck(gpuDeviceGetPCIBusId(busid, 64, i));
                s = "\t On node " + std::string(hnbuf) + " : ";
#ifdef _MPI
                s += "MPI Rank " + std::to_string(ThisTask) + " : ";
#endif
                s += "GPU device " + std::to_string(i);
                s += " Device_Name=" + std::string(prop.name);
                s += " Bus_ID=" + std::string(busid);
                s += " Compute_Units=" + std::to_string(prop.multiProcessorCount);
                s += " Max_Work_Group_Size=" + std::to_string(prop.warpSize);
                s += " Local_Mem_Size=" + std::to_string(prop.sharedMemPerBlock);
                s += " Global_Mem_Size=" + std::to_string(prop.totalGlobalMem);
                s += "\n";
                binding_report +=s;
            }
            // gpuErrorCheck(gpuSetDevice(0));
        }
#ifdef _MPI
        // gather all strings to for outputing info 
        std::vector<int> recvcounts(NProcs);
        std::vector<int> offsets(NProcs);
        int size = binding_report.length();
        auto p1 = recvcounts.data(); 
        MPI_Allgather(&size, 1, MPI_INTEGER, p1, 1, MPI_INTEGER, MPI_COMM_WORLD);
        size = recvcounts[0];
        offsets[0] = 0;
        for (auto i=1;i<NProcs;i++) {size += recvcounts[i]; offsets[i] = offsets[i-1] + recvcounts[i-1];}
        char newbindingreport[size];
        auto p2 = binding_report.c_str();
        MPI_Allgatherv(p2, binding_report.length(), MPI_CHAR,
                newbindingreport, recvcounts.data(), offsets.data(), MPI_CHAR,
                MPI_COMM_WORLD);
        newbindingreport[size-1] = '\0';
        binding_report = std::string(newbindingreport);
#endif
        
        return binding_report;
    }
    /// return binding as called within openmp region 
    std::string ReportThreadAffinity(std::string func, std::string line)
    {
        std::string result;
        cpu_set_t coremask;
        char clbuf[7 * CPU_SETSIZE], hnbuf[64];
        memset(clbuf, 0, sizeof(clbuf));
        memset(hnbuf, 0, sizeof(hnbuf));
        (void)gethostname(hnbuf, sizeof(hnbuf));
        result = "Thread affinity report @ " + func + " L" + line + " : ";
        (void)sched_getaffinity(0, sizeof(coremask), &coremask);
        cpuset_to_cstr(&coremask, clbuf);
        int thread = 0, level = 1;
#ifdef _OPENMP
        thread = omp_get_thread_num();
        level = omp_get_level();
#endif
        result += " Thread " + std::to_string(thread);
        result +=" at level " + std::to_string(level) + " : ";
        result += " Core affinity = " + std::string(clbuf) + " ";
        result += " Core placement = " + std::to_string(sched_getcpu()) + " ";
        result += "\n";

        return result;
    }

    /// return binding as called within openmp region, MPI aware 
#ifdef _MPI 
    std::string MPIReportThreadAffinity(std::string func, std::string line, MPI_Comm &comm)
    {
        std::string result;
        int ThisTask=0, NProcs=1;
        cpu_set_t coremask;
        char clbuf[7 * CPU_SETSIZE], hnbuf[64];

        MPI_Comm_size(comm, &NProcs);
        MPI_Comm_rank(comm, &ThisTask);
        memset(hnbuf, 0, sizeof(hnbuf));
        memset(clbuf, 0, sizeof(clbuf));
        (void)gethostname(hnbuf, sizeof(hnbuf));
        result = "Thread affinity report @ " + func + " L" + line + " : ";
        result += "::\t On node " + std::string(hnbuf) + " : ";
        result += "MPI Rank " + std::to_string(ThisTask) + " : ";
        (void)sched_getaffinity(0, sizeof(coremask), &coremask);
        cpuset_to_cstr(&coremask, clbuf);
        int thread = 0, level = 1;
#ifdef _OPENMP
        thread = omp_get_thread_num();
        level = omp_get_level();
#endif
        result += " Thread " + std::to_string(thread);
        result +=" at level " + std::to_string(level) + " : ";

        result += " Core affinity = " + std::string(clbuf) + " ";
        result += " Core placement = " + std::to_string(sched_getcpu()) + " ";
        result += "\n";

        return result;
    }
#endif

}
