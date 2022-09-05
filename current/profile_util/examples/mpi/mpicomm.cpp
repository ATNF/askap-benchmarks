#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>
#include <tuple>
#include <thread>
#include <profile_util.h>


#ifdef USEOPENMP
#include <omp.h>
#endif


#include <mpi.h>

// if want to try running code but not do any actual communication
// #define TURNOFFMPI
#define MEMFOOTPRINTTEST

int ThisTask, NProcs;
std::chrono::system_clock::time_point logtime;
std::time_t log_time;
char wherebuff[1000];
std::string whenbuff;

#define Where() sprintf(wherebuff,"[%04d] @%sL%d ", ThisTask,__func__, __LINE__);
#define When() logtime = std::chrono::system_clock::now(); log_time = std::chrono::system_clock::to_time_t(logtime);whenbuff=std::ctime(&log_time);whenbuff.erase(std::find(whenbuff.begin(), whenbuff.end(), '\n'), whenbuff.end());
#define LocalLogger() Where();std::cout<<wherebuff<<" : " 
#define Rank0LocalLogger() Where();if (ThisTask==0) std::cout<<wherebuff<<" : " 
#define LocalLoggerWithTime() Where();When(); std::cout<<wherebuff<<" ("<<whenbuff<<") : "
#define Rank0LocalLoggerWithTime() Where();When(); if (ThisTask==0) std::cout<<wherebuff<<" ("<<whenbuff<<") : "
#define LogMPITest() Rank0LocalLoggerWithTime()<<" running "<<mpifunc<< " test"<<std::endl;
#define LogMPIBroadcaster() if (ThisTask == itask) LocalLoggerWithTime()<<" running "<<mpifunc<<" broadcasting "<<sendsize<<" GB"<<std::endl;
#define LogMPISender() LocalLoggerWithTime()<<" Running "<<mpifunc<<" sending "<<sendsize<<" GB"<<std::endl;
#define LogMPIReceiver() if (ThisTask == itask) LocalLoggerWithTime()<<" running "<<mpifunc<<std::endl;
#define LogMPIAllComm() Rank0LocalLoggerWithTime()<<" running "<<mpifunc<<" all "<<sendsize<<" GB"<<std::endl;
#define Rank0ReportMem() if (ThisTask==0) {Where();When();std::cout<<wherebuff<<" ("<<whenbuff<<") : ";LogMemUsage();std::cout<<wherebuff<<" ("<<whenbuff<<") : ";LogSystemMem();}

/// define what type of sends to use 
#define USESEND 0
#define USESSEND 1
#define USEISEND 2

struct Options
{
    /// what types of communication to test
    bool igather = true;
    bool ireduce = true;
    bool iscatter = true;
    bool ibcast = false;
    bool isendrecv = true;
    bool isendrecvsinglerank = false;
    bool ilongdelay = false;
    bool icorrectvalues = false;
    /// root task that will get all the receives
    int roottask = 0;
    int othertask = 0;
    int usesend = USEISEND;
    int delay = 600;
    /// max message size in GB
    double maxgb = 1.0;
    /// max message size in number of doubles 
    int msize = 1000;
    int Niter = 1;
};

std::tuple<int,
    std::vector<MPI_Comm> ,
    std::vector<std::string> ,
    std::vector<int>, 
    std::vector<int>, 
    std::vector<int>>
    MPIAllocateComms()
{
    // number of comms is 2, 4, 8, ... till MPI_COMM_WORLD;
    int numcoms = std::floor(log(static_cast<double>(NProcs))/log(2.0))+1;
    int numcomsold = numcoms;
    std::vector<MPI_Comm> mpi_comms(numcoms);
    std::vector<std::string> mpi_comms_name(numcoms);
    std::vector<int> ThisLocalTask(numcoms), NProcsLocal(numcoms), NLocalComms(numcoms);

    for (auto i=0;i<=numcomsold;i++) 
    {
        NLocalComms[i] = NProcs/pow(2,i+1);
        if (NLocalComms[i] < 2) 
        {
            numcoms = i+1;
            break;
        }
        auto ThisLocalCommFlag = ThisTask % NLocalComms[i];
        MPI_Comm_split(MPI_COMM_WORLD, ThisLocalCommFlag, ThisTask, &mpi_comms[i]);
        MPI_Comm_rank(mpi_comms[i], &ThisLocalTask[i]);
        MPI_Comm_size(mpi_comms[i], &NProcsLocal[i]);
        int tasktag = ThisTask;
        MPI_Bcast(&tasktag, 1, MPI_INTEGER, 0, mpi_comms[i]);
        mpi_comms_name[i] = "Tag_" + std::to_string(static_cast<int>(pow(2,i+1)))+"_worldrank_" + std::to_string(tasktag);
    }
    mpi_comms[numcoms-1] = MPI_COMM_WORLD;
    ThisLocalTask[numcoms-1] = ThisTask;
    NProcsLocal[numcoms-1] = NProcs;
    NLocalComms[numcoms-1] = 1;
    mpi_comms_name[numcoms-1] = "Tag_world";
    ThisLocalTask.resize(numcoms);
    NLocalComms.resize(numcoms);
    NProcsLocal.resize(numcoms);
    mpi_comms.resize(numcoms);
    mpi_comms_name.resize(numcoms);
    // for (auto i=0;i<numcoms;i++) 
    // {
    //     if (ThisLocalTask[i]==0) LocalLoggerWithTime()<<" MPI communicator "<<mpi_comms_name[i]<<" has size of "<<NProcsLocal[i]<<" and there are "<<NLocalComms[i]<<" communicators"<<std::endl;
    // }

    MPI_Barrier(mpi_comms[numcoms-1]);
    return std::make_tuple(numcoms,
        std::move(mpi_comms),
        std::move(mpi_comms_name), 
        std::move(ThisLocalTask), 
        std::move(NProcsLocal), 
        std::move(NLocalComms)
        );
}

void MPIFreeComms(std::vector<MPI_Comm> &mpi_comms, std::vector<std::string> &mpi_comms_name){
    for (auto i=0;i<mpi_comms.size()-1;i++) {
        Rank0LocalLoggerWithTime()<<"Freeing "<<mpi_comms_name[i]<<std::endl;
        MPI_Comm_free(&mpi_comms[i]);
    }
}

std::vector<unsigned long long> MPISetSize(double maxgb) 
{
    std::vector<unsigned long long> sizeofsends(4);
    sizeofsends[0] = 1024.0*1024.0*1024.0*maxgb/sizeof(double);
    for (auto i=1;i<sizeofsends.size();i++) sizeofsends[i] = sizeofsends[i-1]/8;
    std::sort(sizeofsends.begin(),sizeofsends.end());
    
    if (ThisTask==0) {for (auto &x: sizeofsends) {LocalLoggerWithTime()<<"Messages of "<<x<<" elements and "<<x*sizeof(double)/1024./1024./1024.<<" GB"<<std::endl;}}
    MPI_Barrier(MPI_COMM_WORLD);
    return sizeofsends;
}

std::vector<float> MPIGatherTimeStats(profiling_util::Timer time1, std::string f, std::string l)
{
    std::vector<float> times(NProcs);
    auto p = times.data();
    auto time_taken = profiling_util::GetTimeTaken(time1, f, l);
    MPI_Gather(&time_taken, 1, MPI_FLOAT, p, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    return times;
}

std::tuple<float, float, float, float> TimeStats(std::vector<float> times) 
{
    auto ave = 0.0, std = 0.0;
    auto mint = times[0];
    auto maxt = times[0];
    for (auto &t:times)
    {
        ave += t;
        std += t*t;
        mint = std::min(mint,t);
        maxt = std::max(maxt,t);
    }
    float n = times.size();
    ave /= n;
    if (n>1) std = sqrt((std - ave*ave*n)/(n-1.0));
    else std = 0;
    return std::make_tuple(ave, std, mint, maxt);
}

void MPIReportTimeStats(std::vector<float> times, 
    std::string commname, std::string message_size, 
    std::string f, std::string l)
{
    auto[ave, std, mint, maxt] = TimeStats(times);
    Rank0LocalLoggerWithTime()<<"MPI Comm="<<commname<<" @"<<f<<":L"<<l<<" - message size="<<message_size<<" timing [ave,std,min,max]=[" <<ave<<","<<std<<","<<mint<<","<<maxt<<"] (microseconds)"<<std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
}

void MPIReportTimeStats(profiling_util::Timer time1, 
    std::string commname, std::string message_size, 
    std::string f, std::string l)
{
    auto times = MPIGatherTimeStats(time1, f, l);
    MPIReportTimeStats(times, commname, message_size, f, l);
}

/// \defgroup Performance 
//@{
void MPITestBcast(Options &opt) 
{
    MPI_Status status;
    std::string mpifunc;
    auto[numcoms, mpi_comms, mpi_comms_name, ThisLocalTask, NProcsLocal, NLocalComms] = MPIAllocateComms();
    std::vector<double> data;
    
    double * p1 = nullptr, *p2 = nullptr;
    auto  sizeofsends = MPISetSize(opt.maxgb);
   
    // run broadcast 
    mpifunc = "Bcast";
    LogMPITest();
    for (auto i=0;i<sizeofsends.size();i++) 
    {
        auto sendsize = sizeofsends[i]*sizeof(double)/1024./1024./1024.;
        LogMPIAllComm();
        data.resize(sizeofsends[i]);
        Rank0ReportMem();
        for (auto &d:data) d = pow(2.0,ThisTask);
        p1 = data.data();
        auto time1 = NewTimer();
        for (auto j=0;j<mpi_comms.size();j++) 
        {
#ifdef TURNOFFMPI
#else
            if (ThisLocalTask[j] == 0) {std::cout<<ThisTask<<" / "<<ThisLocalTask[j]<<" : Communicating using comm "<<mpi_comms_name[j]<<std::endl;}
            std::vector<float> times;
            for (auto itask=0;itask<NProcs;itask++) 
            {
                for (auto iter=0;iter<opt.Niter;iter++) {
                    auto time2 = NewTimer();
                    MPI_Bcast(p1, sizeofsends[i], MPI_DOUBLE, itask, mpi_comms[j]);
                    auto times_tmp = MPIGatherTimeStats(time2, __func__, std::to_string(__LINE__));
                    times.insert(times.end(), times_tmp.begin(), times_tmp.end());
                }
            }
            MPIReportTimeStats(times, mpi_comms_name[j], std::to_string(sizeofsends[i]), __func__, std::to_string(__LINE__));
#endif
        }
        if (ThisTask==0) LogTimeTaken(time1);
    }
    p1 = p2 = nullptr;
    data.clear();
    data.shrink_to_fit();
    MPIFreeComms(mpi_comms, mpi_comms_name);
}

void MPITestSendRecvSingleRank(Options &opt) 
{
    MPI_Status status;
    std::string mpifunc;
    std::vector<double> senddata, receivedata;
    
    double * p1 = nullptr, *p2 = nullptr;
    auto  sizeofsends = MPISetSize(opt.maxgb);

    // now allreduce 
    mpifunc = "sendrecv_singlerank";
    // use full comms world
    auto commsname = "Tag_world";
    LogMPITest();
    for (auto i=0;i<sizeofsends.size();i++) 
    {
        auto sendsize = sizeofsends[i]*sizeof(double)/1024./1024./1024.;
        LogMPIAllComm();
        senddata.resize(sizeofsends[i]);
        receivedata.resize(sizeofsends[i]);
        Rank0ReportMem();
        for (auto &d:senddata) d = pow(2.0,ThisTask);
        p1 = senddata.data();
        p2 = receivedata.data();
        auto time1 = NewTimer();
        MPI_Status stat;
        std::vector<std::string> messages;
        for (auto itask=0;itask<NProcs;itask++) 
        {
            if (itask == opt.roottask) continue;
            for (auto iter=0;iter<opt.Niter;iter++) 
            {
                if (ThisTask == opt.roottask) 
                {
                    auto time1 = NewTimer();
                    MPI_Sendrecv(p1, sizeofsends[i], MPI_DOUBLE, itask, itask, 
                        p2, sizeofsends[i], MPI_DOUBLE, itask, itask, MPI_COMM_WORLD, &stat);
                    auto timetaken = profiling_util::GetTimeTaken(time1, __func__, std::to_string(__LINE__));
                    // times.push_back(timetaken);
                    messages.push_back(std::to_string(itask) + ": " + std::to_string(timetaken));
                }
                else if (itask == ThisTask) {
                    MPI_Sendrecv(p1, sizeofsends[i], MPI_DOUBLE, opt.roottask, itask, 
                        p2, sizeofsends[i], MPI_DOUBLE, opt.roottask, itask, MPI_COMM_WORLD, &stat);
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (ThisTask == opt.roottask) {
            Rank0LocalLoggerWithTime()<<"MPI Comm="<<commsname<<" @"<<__func__<<":L"<<std::to_string(__LINE__)<<" - message size="<<sizeofsends[i]<<" timing in microseconds from "<<opt.roottask<<" to "<<std::endl;
            for (auto &m:messages) std::cout<<"\t"<<m<<std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    senddata.clear();
    senddata.shrink_to_fit();
    receivedata.clear();
    receivedata.shrink_to_fit();
}

void MPITestSendRecv(Options &opt) 
{
    MPI_Status status;
    std::string mpifunc;
    auto[numcoms, mpi_comms, mpi_comms_name, ThisLocalTask, NProcsLocal, NLocalComms] = MPIAllocateComms();
    std::vector<double> senddata, receivedata;
    
    double * p1 = nullptr, *p2 = nullptr;
    auto  sizeofsends = MPISetSize(opt.maxgb);

    // now allreduce 
    mpifunc = "sendrecv";
    LogMPITest();
#ifdef MEMFOOTPRINTTEST
    {
        auto i = sizeofsends.size()-1;
#else 
    for (auto i=0;i<sizeofsends.size();i++) 
    {
#endif
        auto sendsize = sizeofsends[i]*sizeof(double)/1024./1024./1024.;
        LogMPIAllComm();
        senddata.resize(sizeofsends[i]);
        receivedata.resize(sizeofsends[i]);
        Rank0ReportMem();
        for (auto &d:senddata) d = pow(2.0,ThisTask);
        p1 = senddata.data();
        p2 = receivedata.data();
#ifdef MEMFOOTPRINTTEST
        Rank0LocalLoggerWithTime()" tasks sleeping after allocating memory"<<std::endl;
        sleep(10);
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        auto time1 = NewTimer();
        for (auto j=0;j<mpi_comms.size();j++)
        {
#ifdef TURNOFFMPI
#else
            if (ThisLocalTask[j] == 0) {LocalLoggerWithTime()<<"Communicating using comm "<<mpi_comms_name[j]<<std::endl;}
            std::vector<float> times;
            for (auto iter=0;iter<opt.Niter;iter++) {
                auto time2 = NewTimer();
                std::vector<MPI_Request> sendreqs, recvreqs;
                for (auto isend=0;isend<NProcsLocal[j];isend++) 
                {
                    if (isend != ThisLocalTask[j]) 
                    {
                        MPI_Request request;
                        int tag = isend*NProcsLocal[j]+ThisLocalTask[j];
                        MPI_Isend(p1, sizeofsends[i], MPI_DOUBLE, isend, tag, mpi_comms[j], &request);
                        sendreqs.push_back(request);
                    }
                }
                LocalLoggerWithTime()<<" Placed isends "<<std::endl;
                for (auto irecv=0;irecv<NProcsLocal[j];irecv++) 
                {
                    if (irecv != ThisLocalTask[j]) 
                    {
                        MPI_Request request;
                        int tag = ThisLocalTask[j]*NProcsLocal[j]+irecv;
                        MPI_Irecv(p2, sizeofsends[i], MPI_DOUBLE, irecv, tag, mpi_comms[j], &request);
                        recvreqs.push_back(request);
                    }
                }
                Rank0ReportMem();
                MPI_Waitall(recvreqs.size(), recvreqs.data(), MPI_STATUSES_IGNORE);
                LocalLoggerWithTime()<<" Received ireceives "<<std::endl;
                auto times_tmp = MPIGatherTimeStats(time2, __func__, std::to_string(__LINE__));
                times.insert(times.end(), times_tmp.begin(), times_tmp.end());
            }
            MPI_Barrier(MPI_COMM_WORLD);
            MPIReportTimeStats(times, mpi_comms_name[j], std::to_string(sizeofsends[i]), __func__, std::to_string(__LINE__));
#endif
        }
        if (ThisTask==0) LogTimeTaken(time1);
    }
    senddata.clear();
    senddata.shrink_to_fit();
    receivedata.clear();
    receivedata.shrink_to_fit();
    Rank0ReportMem();
    MPI_Barrier(MPI_COMM_WORLD);
}


void MPITestAllGather(Options &opt) 
{

}

void MPITestAllScatter(Options &opt) 
{

}

void MPITestAllReduce(Options &opt) 
{
    MPI_Status status;
    std::string mpifunc;
    auto[numcoms, mpi_comms, mpi_comms_name, ThisLocalTask, NProcsLocal, NLocalComms] = MPIAllocateComms();
    std::vector<double> data, allreducesum;
    
    double * p1 = nullptr, *p2 = nullptr;
    auto  sizeofsends = MPISetSize(opt.maxgb);

    // now allreduce 
    mpifunc = "allreduce";
    LogMPITest();
#ifdef TURNOFFMPI
    data.resize(sizeofsends[sizeofsends.size()-1]);
    allreducesum.resize(sizeofsends[sizeofsends.size()-1]);
    for (auto &d:data) d = pow(2.0,ThisTask);
    sleep(10);
    for (auto &d:allreducesum) d = pow(2.0,ThisTask);
#endif
#ifdef MEMFOOTPRINTTEST
    {
        auto i = sizeofsends.size()-1;
#else 
    for (auto i=0;i<sizeofsends.size();i++) 
    {
#endif
        auto sendsize = sizeofsends[i]*sizeof(double)/1024./1024./1024.;
        LogMPIAllComm();
        data.resize(sizeofsends[i]);
        allreducesum.resize(sizeofsends[i]);
        Rank0ReportMem();
        for (auto &d:data) d = pow(2.0,ThisTask);
        p1 = data.data();
        p2 = allreducesum.data();
#ifdef MEMFOOTPRINTTEST
        Rank0LocalLoggerWithTime()<<" tasks sleeping after allocating memory"<<std::endl;
        sleep(10);
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        auto time1 = NewTimer();
        for (auto j=0;j<mpi_comms.size();j++) 
        {
#ifdef TURNOFFMPI
#else
            if (ThisLocalTask[j] == 0) {LocalLoggerWithTime()<<"Communicating using comm "<<mpi_comms_name[j]<<std::endl;}
            std::vector<float> times;
            for (auto iter=0;iter<opt.Niter;iter++) {
                auto time2 = NewTimer();
                MPI_Allreduce(p1, p2, sizeofsends[i], MPI_DOUBLE, MPI_SUM, mpi_comms[j]);
                auto times_tmp = MPIGatherTimeStats(time2, __func__, std::to_string(__LINE__));
                times.insert(times.end(), times_tmp.begin(), times_tmp.end());
            }
            Rank0ReportMem();
            sleep(2);
            MPI_Barrier(MPI_COMM_WORLD);
            MPIReportTimeStats(times, mpi_comms_name[j], std::to_string(sizeofsends[i]), __func__, std::to_string(__LINE__));
#endif
        }
        if (ThisTask==0) LogTimeTaken(time1);
    }
    data.clear();
    data.shrink_to_fit();
    allreducesum.clear();
    allreducesum.shrink_to_fit();
    MPIFreeComms(mpi_comms, mpi_comms_name);
    Rank0ReportMem();
    MPI_Barrier(MPI_COMM_WORLD);
}
//@}

/// \defgroup SanityChecks
//@{

/// Test whether the MPI interface can handle a long delay between a send and the 
/// corresponding receive.
void MPITestLongDelay(Options &opt) 
{
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Status status;
    std::string mpifunc;
    unsigned long size = opt.msize;
    std::vector<double> data(size);
    void * p1 = nullptr, *p2 = nullptr;

    mpifunc = "longdelay";
    for (auto &d:data) d = pow(2.0,ThisTask);
    auto time1 = NewTimer();

    if (ThisTask != opt.othertask) sleep(opt.delay);
    if (ThisTask == opt.roottask) 
    {
        for (auto itask = 0;itask<NProcs;itask++) 
        {
            if (itask == opt.roottask) continue;
            int mpi_err;
            LocalLogger()<<" receiving from "<<itask<<std::endl;
            mpi_err = MPI_Recv(&size, 1, MPI_UNSIGNED_LONG, itask, 0, MPI_COMM_WORLD, &status);
            LocalLogger()<<" size "<<size<<" received from "<<itask<<" with mpi return of " <<mpi_err<<std::endl;
            data.resize(size);
            p1 = data.data();
            mpi_err = MPI_Recv(p1, size, MPI_DOUBLE, itask, 0, MPI_COMM_WORLD, &status);
            LocalLogger()<<" received from "<<itask<<" with mpi return of "<<mpi_err<<std::endl;
        }
    }
    else {
        MPI_Request request;
        int mpi_err;
        LocalLogger()<<" sending to "<<opt.roottask<<" with send type of "<<opt.usesend<<std::endl;
        size = data.size();
        p1 = data.data();
        if (opt.usesend == USESEND) {
            mpi_err = MPI_Send(&size, 1, MPI_UNSIGNED_LONG, opt.roottask, 0, MPI_COMM_WORLD);
            mpi_err = MPI_Send(p1, size, MPI_DOUBLE, opt.roottask, 0, MPI_COMM_WORLD);
        }
        else if (opt.usesend == USEISEND) 
        {
            mpi_err = MPI_Isend(&size, 1, MPI_UNSIGNED_LONG, opt.roottask, 0, MPI_COMM_WORLD, &request);
            mpi_err = MPI_Isend(p1, size, MPI_DOUBLE, opt.roottask, 0, MPI_COMM_WORLD, &request);
        }
        else if (opt.usesend == USESSEND) 
        {
            mpi_err = MPI_Ssend(&size, 1, MPI_UNSIGNED_LONG, opt.roottask, 0, MPI_COMM_WORLD);
            mpi_err = MPI_Ssend(p1, size, MPI_DOUBLE, opt.roottask, 0, MPI_COMM_WORLD);
        }
        LocalLogger()<<" sent to "<<opt.roottask<<" with "<<mpi_err<<std::endl;
    }
    if (ThisTask==0) LogTimeTaken(time1);
    data.clear();
    data.shrink_to_fit();
    p1 = p2 = nullptr;
}

/// Test whether the MPI interface sends the correct data 
void MPITestCorrectSendRecv(Options &opt) 
{
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Status status;
    std::string mpifunc;
    unsigned long size = 5;
    std::vector<double> data(size);
    void * p1 = nullptr, *p2 = nullptr;

    mpifunc = "correct values";
    for (auto &d:data) d = pow(2.0,ThisTask);
    auto time1 = NewTimer();
    if (ThisTask == opt.roottask) 
    {
        auto oldsize = size;
        for (auto itask = 0;itask<NProcs;itask++) 
        {
            if (itask == opt.roottask) continue;
            int mpi_err;
            LocalLogger()<<" receiving from "<<itask<<std::endl;
            mpi_err = MPI_Recv(&size, 1, MPI_UNSIGNED_LONG, itask, 0, MPI_COMM_WORLD, &status);
            LocalLogger()<<" size "<<size<<" received from "<<itask<<" with " <<mpi_err<<std::endl;
            if (size != oldsize) {
                LocalLogger()<<" GOT WRONG SIZE VALUE from "<<itask<<std::endl;
                MPI_Abort(MPI_COMM_WORLD,8);
            }
            data.resize(size);
            p1 = data.data();
            mpi_err = MPI_Recv(p1, size, MPI_DOUBLE, itask, 0, MPI_COMM_WORLD, &status);
            std::vector<double> refdata(oldsize);
            for (auto &d:refdata) d = pow(2.0,itask);
            for (auto i=0;i<oldsize;i++) {
                if (data[i] != refdata[i]) {
                    LocalLogger()<<" GOT WRONG data VALUE from "<<itask<<std::endl;
                    MPI_Abort(MPI_COMM_WORLD,8);
                }
            }

            std::string s;
            for (auto &d:data) s+=std::to_string(d) + " ";
            LocalLogger()<<" received from "<<itask<<" with "<<mpi_err<<std::endl;
        }
    }
    else {
        MPI_Request request;
        int mpi_err;
        LocalLogger()<<" sending to "<<opt.roottask<<" with send type of "<<opt.usesend<<std::endl;
        size = data.size();
        p1 = data.data();
        if (opt.usesend == USESEND) {
            mpi_err = MPI_Send(&size, 1, MPI_UNSIGNED_LONG, opt.roottask, 0, MPI_COMM_WORLD);
            mpi_err = MPI_Send(p1, size, MPI_DOUBLE, opt.roottask, 0, MPI_COMM_WORLD);
        }
        else if (opt.usesend == USEISEND) 
        {
            mpi_err = MPI_Isend(&size, 1, MPI_UNSIGNED_LONG, opt.roottask, 0, MPI_COMM_WORLD, &request);
            mpi_err = MPI_Isend(p1, size, MPI_DOUBLE, opt.roottask, 0, MPI_COMM_WORLD, &request);
        }
        else if (opt.usesend == USESSEND) 
        {
            mpi_err = MPI_Ssend(&size, 1, MPI_UNSIGNED_LONG, opt.roottask, 0, MPI_COMM_WORLD);
            mpi_err = MPI_Ssend(p1, size, MPI_DOUBLE, opt.roottask, 0, MPI_COMM_WORLD);
        }
        LocalLogger()<<" sent to "<<opt.roottask<<" with "<<mpi_err<<std::endl;
    }
    if (ThisTask==0) LogTimeTaken(time1);
    data.clear();
    data.shrink_to_fit();
    p1 = p2 = nullptr;
}
//@}

void MPIRunTests(Options &opt)
{
    if (opt.ilongdelay) {
        MPITestLongDelay(opt);
        return;
    }
    if (opt.icorrectvalues) {
        MPITestCorrectSendRecv(opt);
        return;
    }
    if (opt.isendrecvsinglerank) MPITestSendRecvSingleRank(opt);
    if (opt.igather) MPITestAllGather(opt);
    if (opt.iscatter) MPITestAllScatter(opt);
    for (auto i=0;i<3;i++) {
    if (opt.ireduce) MPITestAllReduce(opt);
    sleep(5);
    Rank0ReportMem();
    sleep(5);
    MPI_Barrier(MPI_COMM_WORLD);
    }
    if (opt.ibcast) MPITestBcast(opt);

    for (auto i=0;i<2;i++) {
    if (opt.isendrecv) MPITestSendRecv(opt);
    sleep(5);
    Rank0ReportMem();
    sleep(5);
    MPI_Barrier(MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &NProcs);
    MPI_Comm_rank(comm, &ThisTask);
    Options opt;

    // init logger time
    logtime = std::chrono::system_clock::now();
    log_time = std::chrono::system_clock::to_time_t(logtime);
    auto start = std::chrono::system_clock::now();
    std::time_t start_time = std::chrono::system_clock::to_time_t(start);
    Rank0LocalLoggerWithTime()<<"Starting job "<<std::endl;
    Rank0ReportMem();
    MPILog0NodeMemUsage(comm);
    MPILog0NodeSystemMem(comm);
    if (argc >= 2) opt.maxgb = atof(argv[1]);
    if (argc == 3) opt.Niter = atof(argv[2]);
    opt.othertask = NProcs/2 + 1;
    // if (argc >= 2) opt.delay = atoi(argv[1]);
    // if (argc >= 3) opt.msize = atoi(argv[2]);
    // if (argc >= 4) opt.othertask = atoi(argv[3]);

    // default value for 2 node tests assuming that same number of tasks per node 
    // ensures that othertask is testing internode communication
    // alter if want intranode communcation to something like opt.roottask + 1;
    opt.othertask = NProcs/2 + 1;
    
    MPILog0ParallelAPI();
    MPILog0Binding();
    MPI_Barrier(MPI_COMM_WORLD);
    MPIRunTests(opt);

    Rank0LocalLoggerWithTime()<<"Ending job "<<std::endl;
    MPI_Finalize();
    return 0;
}
