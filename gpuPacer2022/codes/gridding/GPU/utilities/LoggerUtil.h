/*! \file LoggerUtil.h
 *  \brief this file contains all function prototypes for logging containing in MaxUtil.cpp, ThreadAffinityUtil.cpp and TimeUtil.cpp codes. Based on github.com/pelahi/profile_util repo
 */

#ifndef _LOGGER_UTIL
#define _LOGGER_UTIL

#include <cstring>
#include <string>
#include <vector>
#include <tuple>
#include <ostream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <array>
#include <algorithm>

#include <sched.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/sysinfo.h>
#include "gpuCommon.h"

#ifdef _MPI
#include <mpi.h>
#endif 
#ifdef _OPENMP 
#include <omp.h>
#endif

namespace logger_util {

    /// function that returns a string of the time at when it is called. 
    std::string __when();
    /// function that converts the mask of thread affinity to human readable string 
    void cpuset_to_cstr(cpu_set_t *mask, char *str);
    /// reports the parallelAPI 
    /// @return string of MPI comm size and OpenMP version and max threads for given rank
    /// \todo needs to be generalized to report parallel API of code and not library
    std::string ReportParallelAPI();
    /// reports binding of MPI comm world and each ranks thread affinity 
    /// @return string of MPI comm rank and thread core affinity 
    std::string ReportBinding();
    /// reports thread affinity within a given scope, thus depends if called within OMP region 
    /// @param func function where called in code, useful to provide __func__ and __LINE
    /// @param line code line number where called
    /// @return string of thread core affinity 
    std::string ReportThreadAffinity(std::string func, std::string line);
#ifdef _MPI
    /// reports thread affinity within a given scope, thus depends if called within OMP region, MPI aware
    /// @param func function where called in code, useful to provide __func__ and __LINE
    /// @param line code line number where called
    /// @param comm MPI communicator
    /// @return string of MPI comm rank and thread core affinity 
    std::string MPIReportThreadAffinity(std::string func, std::string line, MPI_Comm &comm);
#endif

    /// run a command
    /// @param cmd string of command to run on system
    /// @return string of MPI comm rank and thread core affinity 
    std::string exec_sys_cmd(std::string cmd);

    namespace detail {

        template <int N, typename T>
        struct _fixed {
            T _val;
        };

        template <typename T, int N, typename VT>
        inline
        std::basic_ostream<T> &operator<<(std::basic_ostream<T> &os, detail::_fixed<N, VT> v)
        {
            os << std::setprecision(N) << std::fixed << v._val;
            return os;
        }

    } // namespace detail

    ///
    /// Sent to a stream object, this manipulator will print the given value with a
    /// precision of N decimal places.
    ///
    /// @param v The value to send to the stream
    ///
    template <int N, typename T>
    inline
    detail::_fixed<N, T> fixed(T v) {
        return {v};
    }

    namespace detail {

        struct _memory_amount {
            std::size_t _val;
        };

        struct _microseconds_amount {
            std::chrono::microseconds::rep _val;
        };

        template <typename T>
        inline
        std::basic_ostream<T> &operator<<(std::basic_ostream<T> &os, const detail::_memory_amount &m)
        {

            if (m._val < 1024) {
                os << m._val << " [B]";
                return os;
            }

            float v = m._val / 1024.;
            const char *suffix = " [KiB]";

            if (v > 1024) {
                v /= 1024;
                suffix = " [MiB]";
            }
            if (v > 1024) {
                v /= 1024;
                suffix = " [GiB]";
            }
            if (v > 1024) {
                v /= 1024;
                suffix = " [TiB]";
            }
            if (v > 1024) {
                v /= 1024;
                suffix = " [PiB]";
            }
            if (v > 1024) {
                v /= 1024;
                suffix = " [EiB]";
            }
            // that should be enough...

            os << fixed<3>(v) << suffix;
            return os;
        }

        template <typename T>
        inline
        std::basic_ostream<T> &operator<<(std::basic_ostream<T> &os, const detail::_microseconds_amount &t)
        {
            auto time = t._val;
            if (time < 1000) {
                os << time << " [us]";
                return os;
            }

            time /= 1000;
            if (time < 1000) {
                os << time << " [ms]";
                return os;
            }

            float ftime = time / 1000.f;
            const char *prefix = " [s]";
            if (ftime > 60) {
                ftime /= 60;
                prefix = " [min]";
                if (ftime > 60) {
                    ftime /= 60;
                    prefix = " [h]";
                    if (ftime > 24) {
                        ftime /= 24;
                        prefix = " [d]";
                    }
                }
            }
            // that should be enough...

            os << fixed<3>(ftime) << prefix;
            return os;
        }

    } // namespace detail

    ///
    /// Sent to a stream object, this manipulator will print the given amount of
    /// memory using the correct suffix and 3 decimal places.
    ///
    /// @param v The value to send to the stream
    ///
    inline
    detail::_memory_amount memory_amount(std::size_t amount) {
        return {amount};
    }

    ///
    /// Sent to a stream object, this manipulator will print the given amount of
    /// nanoseconds using the correct suffix and 3 decimal places.
    ///
    /// @param v The value to send to the stream
    ///
    inline
    detail::_microseconds_amount us_time(std::chrono::microseconds::rep amount) {
        return {amount};
    }

    struct memory_stats {
        std::size_t current = 0;
        std::size_t peak = 0;
        std::size_t change = 0;
    };

    struct memory_usage {
        memory_stats vm;
        memory_stats rss;
        memory_usage operator+=(const memory_usage& rhs)
        {
            this->vm.current += rhs.vm.current;
            if (this->vm.peak < rhs.vm.peak) this->vm.peak = rhs.vm.peak;
            this->vm.change += rhs.vm.change;

            this->rss.current += rhs.rss.current;
            if (this->rss.peak < rhs.rss.peak) this->vm.peak = rhs.rss.peak;
            this->rss.change += rhs.rss.change;
            return *this;
        };
    };

    struct sys_memory_stats
    {
        std::size_t total;
        std::size_t used;
        std::size_t free;
        std::size_t shared;
        std::size_t cache;
        std::size_t avail;
    };

    ///get memory usage
    memory_usage get_memory_usage();
    ///report memory usage from within a specific function/scope
    ///usage would be from within a function use 
    ///auto l=std::to_string(__LINE__); auto f = __func__; GetMemUsage(f,l);
    std::string ReportMemUsage(const std::string &f, const std::string &l);
    /// like above but also reports change relative to another sampling of memory 
    std::string ReportMemUsage(const memory_usage &prior_mem_use, const std::string &f, const std::string &l);
    /// like ReportMemUsage but also returns the mem usage 
    std::tuple<std::string, memory_usage> GetMemUsage(const std::string &f, const std::string &l);
    std::tuple<std::string, memory_usage> GetMemUsage(const memory_usage &prior_mem_use, const std::string &f, const std::string &l);
    /// Get memory usage on all hosts 
    #ifdef _MPI
    std::string MPIReportNodeMemUsage(MPI_Comm &comm, 
    const std::string &function, 
    const std::string &line_num
    );
    std::tuple<std::string, std::vector<std::string>, std::vector<memory_usage>> MPIGetNodeMemUsage(MPI_Comm &comm, 
    const std::string &function, 
    const std::string &line_num
    );
    #endif


    /// get the memory of the system using free
    sys_memory_stats get_system_memory();
    ///report memory state of the system from within a specific function/scope
    ///usage would be from within a function use 
    ///auto l=std::to_string(__LINE__); auto f = __func__; GetMemUsage(f,l);
    std::string ReportSystemMem(const std::string &f, const std::string &l);
    /// like above but also reports change relative to another sampling of memory 
    std::string ReportSystemMem(const sys_memory_stats &prior_mem_use, const std::string &f, const std::string &l);
    /// like ReportSystemMem but also returns the system memory
    std::tuple<std::string, sys_memory_stats> GetSystemMem(const std::string &f, const std::string &l);
    std::tuple<std::string, sys_memory_stats> GetSystemMem(const sys_memory_stats &prior_mem_use, const std::string &f, const std::string &l);
    #ifdef _MPI
    std::string MPIReportNodeSystemMem(MPI_Comm &comm, const std::string &function, const std::string &line_num);
    std::tuple<std::string, std::vector<std::string>, std::vector<sys_memory_stats>> MPIGetNodeSystemMem(MPI_Comm &comm, const std::string &function, const std::string &line_num);
    #endif

    /// Timer class. 
    /// In code create an instance of time and then just a mantter of 
    /// creating an instance and then reporting it. 
    class Timer {

    public:

        using clock = std::chrono::high_resolution_clock;
        using duration = typename std::chrono::microseconds::rep;
        

        /*!
         * Returns whether timer has timer on device and not just host
         *
         * @return boolean on whether timer on device [us]
         */
        inline
        bool get_use_device() const {return use_device;}
        /*!
         * Returns the number of milliseconds elapsed since the reference time
         * of the timer
         *
         * @return The time elapsed since the creation of the timer, in [us]
         */
        inline
        duration get() const {
            return std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - tref).count();
        }

        /*!
         * Returns the number of milliseconds elapsed since the creation
         * of the timer
         *
         * @return The time elapsed since the creation of the timer, in [us]
         */
        inline
        duration get_creation() const {
            return std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();
        }

        /*!
         * Returns the elapsed time on device since the reference time
         * of the device event
         *
         * @return The time elapsed since the creation of the timer, in [us]
         */
#if defined(_GPU)
        inline
        float get_on_device() const 
        {
            if (!use_device) return 0;
            gpuEvent_t t1_event;
            gpuErrorCheck(gpuEventCreate(&t1_event));
            gpuErrorCheck(gpuEventRecord(t1_event)); 
            gpuErrorCheck(gpuEventSynchronize(t1_event));
            float telapsed;
            gpuErrorCheck(gpuEventElapsedTime(&telapsed,t0_event,t1_event));
            telapsed *= _GPU_TO_SECONDS; // to convert to seconds 
            gpuErrorCheck(gpuEventDestroy(t1_event));
            return telapsed;
        }
#endif

        void set_ref(const std::string &new_ref)
        {
            ref = new_ref;
            t0 = clock::now();
#if defined(_GPU)
            if (use_device) {
                gpuErrorCheck(gpuEventRecord(t0_event)); 
                gpuErrorCheck(gpuEventSynchronize(t0_event));
            }
#endif
        };
        std::string get_ref() const 
        {
            return ref;
        };

        Timer(const std::string &f, const std::string &l, bool _use_device=true) {
            ref="@"+f+" L"+l;
            t0 = clock::now();
            tref = t0;
            use_device = _use_device;
#if defined(_GPU)
            if (use_device) {
                gpuErrorCheck(gpuEventCreate(&t0_event));
                gpuErrorCheck(gpuEventRecord(t0_event)); 
                gpuErrorCheck(gpuEventSynchronize(t0_event));
            }
#endif
        }
#if defined(_GPU)
        ~Timer()
        {
            if (use_device) gpuErrorCheck(gpuEventDestroy(t0_event));
        }
#endif

    private:
        clock::time_point t0;
        clock::time_point tref;
        std::string ref;
        bool use_device = true;
#if defined(_GPU)
        gpuEvent_t t0_event;
#endif
    };

    /// get the time taken between some reference time (which defaults to creation of timer )
    /// and current call
    std::string ReportTimeTaken(const Timer &t, const std::string &f, const std::string &l);
    float GetTimeTaken(const Timer &t, const std::string &f, const std::string &l);

#if defined(_GPU)
    std::string ReportTimeTakenOnDevice(const Timer &t, const std::string &f, const std::string &l);
    float GetTimeTakenOnDevice(const Timer &t, const std::string &f, const std::string &l);
#endif
}


/// \def logger utility definitions 
//@{
#define _where_calling_from "@"<<__func__<<" L"<<std::to_string(__LINE__)<<" "
#define _when_calling_from "("<<logger_util::__when()<<") : "
#ifdef _MPI 
#define _MPI_calling_rank(rank) "["<<std::setw(5) << std::setfill('0')<<rank<<"] "<<std::setw(0)
#define _log_header __MPI_calling_rank(__rank)<<_where_calling_from<<_when_callling_from
#else 
#define _log_header _where_calling_from<<_when_calling_from
#endif

//@}

/// \def gerenal logging  
//@{
#ifdef __MPI
#define LocalLogger(logger, __rank) logger<<_log_header
#define LocalLog(__rank) std::cout<<_log_header
#else 
#define LocalLogger(logger) logger<<_log_header
#define LocalLog() std::cout<<_log_header
#endif

/// \defgroup LogAffinity
/// Log thread affinity and parallelism either to std or an ostream
//@{
#define LogParallelAPI() std::cout<<_log_header<<"\n"<<logger_util::ReportParallelAPI()<<std::endl;
#define LogBinding() std::cout<<_where_calling_from<<"\n"<<logger_util::ReportBinding()<<std::endl;
#define LogThreadAffinity() printf("%s \n", logger_util::ReportThreadAffinity(__func__, std::to_string(__LINE__)).c_str());
#define LoggerThreadAffinity(logger) logger<<logger_util::ReportThreadAffinity(__func__, std::to_string(__LINE__))<<std::endl;
#ifdef _MPI
#define MPILog0ThreadAffinity() if(ThisTask == 0) printf("%s \n", logger_util::ReportThreadAffinity(__func__, std::to_string(__LINE__)).c_str());
#define MPILogger0ThreadAffinity(logger) if(ThisTask == 0)logger<<logger_util::ReportThreadAffinity(__func__, std::to_string(__LINE__))<<std::endl;
#define MPILogThreadAffinity(comm) printf("%s \n", logger_util::MPIReportThreadAffinity(__func__, std::to_string(__LINE__), comm).c_str());
#define MPILoggerThreadAffinity(logger, comm) logger<<logger_util::MPIReportThreadAffinity(__func__, std::to_string(__LINE__), comm)<<std::endl;
#define MPILog0ParallelAPI() if(ThisTask==0) std::cout<<_where_calling_from<<"\n"<<logger_util::ReportParallelAPI()<<std::endl;
#define MPILog0Binding() {auto s =logger_util::ReportBinding(); if (ThisTask == 0) std::cout<<_where_calling_from<<"\n"<<s<<std::endl;}
#endif
//@}

/// \defgroup LogMem
/// Log memory usage either to std or an ostream
//@{
#define LogMemUsage() std::cout<<logger_util::ReportMemUsage(__func__, std::to_string(__LINE__))<<std::endl;
#define LoggerMemUsage(logger) logger<<logger_util::ReportMemUsage(__func__, std::to_string(__LINE__))<<std::endl;

#ifdef _MPI
#define MPILogMemUsage() std::cout<<_MPI_calling_rank(ThisTask)<<logger_util::ReportMemUsage(__func__, std::to_string(__LINE__))<<std::endl;
#define MPILoggerMemUsage(logger) logger<<_MPI_calling_rank(ThisTask)<<logger_util::ReportMemUsage(__func__, std::to_string(__LINE__))<<std::endl;
#define MPILog0NodeMemUsage(comm) {auto report=logger_util::MPIReportNodeMemUsage(comm, __func__, std::to_string(__LINE__));if (ThisTask == 0) {std::cout<<_MPI_calling_rank(ThisTask)<<report<<std::endl;}}
#define MPILogger0NodeMemUsage(logger, comm) {auto report=logger_util::MPIReportNodeMemUsage(comm, __func__, std::to_string(__LINE__));if (ThisTask == 0) logger<<_MPI_calling_rank(ThisTask)<<report<<std::endl;}
#endif

#define LogSystemMem() std::cout<<logger_util::ReportSystemMem(__func__, std::to_string(__LINE__))<<std::endl;
#define LoggerSystemMem(logger) logger<<logger_util::ReportSystemMem(__func__, std::to_string(__LINE__))<<std::endl;

#ifdef _MPI
#define MPILogSystemMem() std::cout<<_MPI_calling_rank(ThisTask)<<logger_util::ReportSystemMem(__func__, std::to_string(__LINE__))<<std::endl;
#define MPILoggerSystemMem(logger) logger<<_MPI_calling_rank(ThisTask)<<logger_util::ReportSystemMem(__func__, std::to_string(__LINE__))<<std::endl;
#define MPILog0NodeSystemMem(comm) {auto report=logger_util::MPIReportNodeSystemMem(comm, __func__, std::to_string(__LINE__));if (ThisTask == 0){std::cout<<_MPI_calling_rank(ThisTask)<<report<<std::endl;}}
#define MPILogger0NodeSystemMem(logger, comm) {auto report = logger_util::MPIReportNodeSystemMem(__func__, std::to_string(__LINE__));if (ThisTask == 0) {logger<<_MPI_calling_rank(ThisTask)<<report<<std::endl;}}
#endif
//@}


/// \defgroup LogTime
/// Log time taken either to std or an ostream
//@{
#define LogTimeTaken(timer) std::cout<<logger_util::ReportTimeTaken(timer, __func__, std::to_string(__LINE__))<<std::endl;
#define LoggerTimeTaken(logger,timer) logger<<logger_util::ReportTimeTaken(timer,__func__, std::to_string(__LINE__))<<std::endl;
#define LogTimeTakenOnDevice(timer) std::cout<<logger_util::ReportTimeTakenOnDevice(timer, __func__, std::to_string(__LINE__))<<std::endl;
#define LoggerTimeTakenOnDevice(logger,timer) logger<<logger_util::ReportTimeTakenOnDevice(timer,__func__, std::to_string(__LINE__))<<std::endl;
#ifdef _MPI
#define MPILogTimeTaken(timer) std::cout<<_MPI_calling_rank(ThisTask)<<logger_util::ReportTimeTaken(timer, __func__, std::to_string(__LINE__))<<std::endl;
#define MPILoggerTimeTaken(logger,timer) logger<<_MPI_calling_rank(ThisTask)<<logger_util::ReportTimeTaken(timer,__func__, std::to_string(__LINE__))<<std::endl;
#define MPILogTimeTakenOnDevice(timer) std::cout<<_MPI_calling_rank(ThisTask)<<logger_util::ReportTimeTakenOnDevice(timer, __func__, std::to_string(__LINE__))<<std::endl;
#define MPILoggerTimeTakenOnDevice(logger,timer) logger<<_MPI_calling_rank(ThisTask)<<logger_util::ReportTimeTakenOnDevice(timer,__func__, std::to_string(__LINE__))<<std::endl;
#endif 
#define NewTimer() logger_util::Timer(__func__, std::to_string(__LINE__));
#define NewTimerHostOnly() logger_util::Timer(__func__, std::to_string(__LINE__), false);
//@}

#endif
