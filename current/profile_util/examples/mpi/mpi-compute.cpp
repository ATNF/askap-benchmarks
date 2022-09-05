/*! \file mpi-compute.cpp
 *  \brief Unit test for checking scaling of codes
    This code has two phases: 
    
    1) generate data and redistribute : (here pt2pt where each rank likely communicates to all other ranks individualized information). 
    2 Iterate: 
        Tranform data: moves particle data, means that particles will need to be redistributed to appropriate mpi domains)
        Grid data: just grid data locally and use allreduce so each mpi rank has fully updated grid
        FFT: not yet there but useful hook to include fftw scaling
        Calculate some data: random compute that uses the grid and particle data (not yet fleshed out) 
        Redistribute data: pt2pt where communication zone of each rank can be tailored at runtime. 
    
    The arguments are (npoints, Niter, deltap, icompute, iverbose)
    - npoints: total number of points is in fact npoints^3 
    - Niter : number of iterations to do
    - deltap : unitless but particles in transform data are moved by deltap * box size / NProcs. This can be used to increase the amount 
    of communication that happens during the iteration. For instance for deltap < 1, at most the pt2pt communication of a given rank i
    will be limited to i-1, i+1, the surrounding ranks (where communication is wrapped as the system is periodic). Increasing it to say >10 
    or so will ensure that a given rank will communicate with most other ranks, increasing the pt2pt done during each iteration
    - icompute : boolean say whether or not to do heavier compute per iteration
    - iverbose: increase verbosity of output
 */

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

#ifdef USEFFTW
#include <fftw.h>
#endif

#include <mpi.h>

// if want to try running code but not do any actual communication
// #define TURNOFFMPI

int ThisTask, NProcs;
#define STENCILESIZE 1

struct Options
{
    unsigned long long npoints = 256;
    unsigned long long ngrid = 54;
    int Niter = 100;
    double p = 1.0;
    double deltap = 0.1;
    bool iverbose = false;
    bool icompute = true;
};

struct PointData
{
    double x[3] = {0, 0, 0};
    unsigned long long id = 0;
    int type = 0;
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
    for (auto i=0;i<numcoms;i++) 
    {
        if (ThisLocalTask[i]==0) std::cout<<" MPI communicator "<<mpi_comms_name[i]<<" has size of "<<NProcsLocal[i]<<" and there are "<<NLocalComms[i]<<" communicators"<<std::endl;
    }
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
        if (ThisTask==0) std::cout<<"Freeing "<<mpi_comms_name[i]<<std::endl;
        MPI_Comm_free(&mpi_comms[i]);
    }
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
    std::string f, std::string l)
{
    auto[ave, std, mint, maxt] = TimeStats(times);
    if (ThisTask==0) {
        std::cout<<"@"<<f<<":L"<<l<<" timing [ave,std,min,max]=[" <<ave<<","<<std<<","<<mint<<","<<maxt<<"] (microseconds)"<<std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void MPIReportTimeStats(profiling_util::Timer time1, 
    std::string f, std::string l)
{
    auto times = MPIGatherTimeStats(time1, f, l);
    MPIReportTimeStats(times, f, l);
}


std::tuple<unsigned long long,
    std::vector<PointData>> GenerateData(Options &opt)
{
    auto N = opt.npoints*opt.npoints*opt.npoints;
    auto Nlocal = N/static_cast<unsigned long long>(NProcs);
    if (ThisTask == NProcs - 1) Nlocal = N-Nlocal*(NProcs-1);
    std::vector<PointData> data(Nlocal);
    std::cout<<__func__<<": Rank "<<ThisTask<<" producing "<<Nlocal<<std::endl;
    auto time1 = NewTimer();

#if defined(USEOPENMP)
#pragma omp parallel default(shared)
{
#endif
    unsigned seed = 4320;
    seed *= (ThisTask+1);
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> pos(0.0,opt.p);
    double x[3];
#if defined(USEOPENMP)
    #pragma omp for schedule(static)
#endif
    for (auto i = 0; i < Nlocal; i++) {
        for (auto j = 0; j < 3; j++) {
            data[i].x[j] = pos(generator);
        }
        data[i].id = i;
        data[i].type = ThisTask;
    }
#if defined(USEOPENMP)
}
#endif
    MPIReportTimeStats(time1, __func__, std::to_string(__LINE__));
    return std::make_tuple(Nlocal, data);

}

void TransformData(Options &opt, unsigned long long Nlocal, std::vector<PointData> &data)
{
    if (ThisTask==0) std::cout<<__func__<<" transforming data ... "<<std::endl;
    auto time1 = NewTimer();
#if defined(USEOPENMP)
#pragma omp parallel default(shared)
{
#endif
    unsigned seed = 4320;
    seed *= (ThisTask+1);
    std::default_random_engine generator(seed);
    auto delta = opt.deltap*opt.p/static_cast<double>(NProcs);
    std::normal_distribution<double> pos(0,delta);
#if defined(USEOPENMP)
    #pragma omp for schedule(static)
#endif
    for (auto &d:data) {
        for (auto j = 0; j < 3; j++) {
            d.x[j] += pos(generator);
            auto y = static_cast<int>(std::floor(d.x[j]/opt.p));
            if (y!=0) d.x[j] -= y*opt.p;
        }
    }
#if defined(USEOPENMP)
}
#endif
    MPIReportTimeStats(time1, __func__, std::to_string(__LINE__));
}


std::vector<double> GridData(Options &opt, unsigned long long Nlocal, std::vector<PointData> &data)
{
    if (ThisTask==0) std::cout<<__func__<<" gridding ... "<<std::endl;
    auto time1 = NewTimer();
    auto n3 = opt.ngrid * opt.ngrid * opt.ngrid;
    auto n2 = opt.ngrid * opt.ngrid;
    auto n = opt.ngrid;
    double *gtemp = new double[n3];
    std::vector<double> griddata(n3);
    for (auto &g:griddata) g=0;
    auto delta = opt.p/static_cast<double>(opt.ngrid);
    unsigned long long ix,iy,iz, index;
    for (auto &d:data) 
    {
        ix = d.x[0]/delta;
        iy = d.x[1]/delta;
        iz = d.x[2]/delta;
        index = ix*n2 + iy*n + iz;
        griddata[index]++;
    }
    auto time2 = NewTimer();
    void * p1 = griddata.data();
    // MPI_Allreduce(p1, MPI_IN_PLACE, n3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(p1, gtemp, n3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPIReportTimeStats(time2, __func__, std::to_string(__LINE__));
    for (auto i=0ul;i<n3;i++) griddata[i] = gtemp[i];
    delete[] gtemp;
    MPIReportTimeStats(time1, __func__, std::to_string(__LINE__));
    return griddata;
}

void FFTData(Options &opt, std::vector<double> &data)
{
}

#ifdef USEOPENMP 
#pragam omp declare simd
#endif
inline unsigned long long period_wrap(long long i, long long n){
    if (i<0) return n+i;
    else if (i>=n) return n-i;
    else return i;
}

#ifdef USEOPENMP 
#pragam omp declare simd
#endif
inline std::tuple<std::vector<unsigned long long>, std::vector<double>> getstencile(double x, double y, double z, double delta, unsigned long long n2, unsigned long long n) 
{
    std::vector<unsigned long long> indices(27);
    std::vector<double> d2(27);
    long long ix, iy, iz;
    ix = std::floor(x/delta);
    iy = std::floor(y/delta);
    iz = std::floor(z/delta);
    int counter = 0;
    for (auto ixx=ix-STENCILESIZE;ixx<=ix+STENCILESIZE; ixx++) 
    {
        auto iix = period_wrap(ixx, n);
        auto dx = ix - ixx;
        for (auto iyy=iy-STENCILESIZE;iyy<=iy+STENCILESIZE; iyy++) 
        {
            auto iiy = period_wrap(iyy, n);
            auto dy = iy - iyy;
            for (auto izz=iz-STENCILESIZE;izz<=iz+STENCILESIZE; izz++) 
            {
                auto iiz = period_wrap(izz, n);
                auto dz = iz - izz;
                indices[counter] = iix*n2 + iiy*n + iiz;
                d2[counter] = dx*dx + dy*dy + dz*dz;
                counter++;
            }
        }
    }
    return std::make_tuple(indices,d2);

}
std::vector<double> ComputeWithData(Options &opt, unsigned long long Nlocal, std::vector<PointData> &data, std::vector<double> &griddata)
{
    if (ThisTask == 0) std::cout<<__func__<<" computing ... "<<std::endl;
    auto time1 = NewTimer();
    auto n2 = opt.ngrid * opt.ngrid;
    auto n = opt.ngrid;
    auto delta = opt.p/static_cast<double>(opt.ngrid);
    std::vector<double> somedata(n2*n);
#if defined(USEOPENMP)
#pragma omp parallel default(shared)
{
#endif
#if defined(USEOPENMP)
    #pragma omp for schedule(static)
#endif
    for (auto &d:data) 
    {
        // get a stencil around particle's grid point, do random stuff
        auto [indices, d2] = getstencile(d.x[0],d.x[1],d.x[2],delta,n2,n);
        double sum = 0, w;
        unsigned long long ref;
        for (auto j = 0 ; j<indices.size();j++)
        {
            if (d2[j] == 0) {w = 1.0; ref = indices[j];}
            else w = 1.0/d2[j];
            sum += w*griddata[indices[j]];
        }
        somedata[ref] = sum;
    }
#if defined(USEOPENMP)
}
#endif
    MPIReportTimeStats(time1, __func__, std::to_string(__LINE__));
    return somedata;
}

inline int GetProc(double w, double x) {return static_cast<int>(std::floor(x/w));}

void RedistributeData(Options &opt, unsigned long long &Nlocal, std::vector<PointData> &data)
{
    if (NProcs < 2) return;
    if (ThisTask == 0) std::cout<<__func__<<" redistributing ..."<<std::endl;
    std::vector<int> Nsend(NProcs), Nrecv(NProcs*NProcs);
    unsigned long long ntotsend = 0, ntotrecv = 0, nranksend = 0, nrankrecv = 0;
    auto slabwidth = opt.p/static_cast<double>(NProcs);
    for (auto &x:Nsend) x=0;
    for (auto i=0ul; i<Nlocal; i++) 
    {
        auto itask = GetProc(slabwidth, data[i].x[0]);
        data[i].type = itask;
        Nsend[itask]++;
        if (itask == ThisTask) data[i].type = -1;
    }
    std::sort(data.begin(), data.end(), 
        [](const PointData & a, const PointData & b)
        {return a.type < b.type;});
    std::vector<unsigned long long> noffset(NProcs);
    Nsend[ThisTask] = 0;
    noffset[0] = 0; 
    for (auto &x:Nsend) {ntotsend += x; nranksend += (x>0);}
    for (auto itask = 1; itask < NProcs; itask++) {
        noffset[itask] = noffset[itask-1] + Nsend[itask-1];
    }
    for (auto &x:Nsend) x = 0;
    std::vector<PointData> Pbuf(ntotsend);
    for (auto i=Nlocal-ntotsend; i<Nlocal; i++) 
    {
        auto itask = data[i].type;
        Pbuf[Nsend[itask] + noffset[itask]] = data[i];
        Nsend[itask]++;
    }
    for (auto &x:Nrecv) x = 0;
    // do an all reduce on Nsend
    {
        auto p1 = Nsend.data();
        auto p2 = Nrecv.data();
        MPI_Allgather(p1, NProcs, MPI_INTEGER, p2, NProcs, MPI_INTEGER, MPI_COMM_WORLD);
    }
    for (auto i=0;i<NProcs;i++) {ntotrecv += Nrecv[i*NProcs+ThisTask]; nrankrecv += (Nrecv[i*NProcs+ThisTask]>0);}
    std::string message;
    message = "Rank " + std::to_string(ThisTask) + " currently has " + std::to_string(Nlocal);
    message += " total [nsend,nrecv] = [" + std::to_string(ntotsend) + "," + std::to_string(ntotrecv) + "]";
    message += " total [nranksend,nrankrecv] = [" + std::to_string(nranksend) + "," + std::to_string(nrankrecv) + "]";
    if (opt.iverbose) {
        for (auto itask = 0; itask<NProcs; itask++) {
            if (ThisTask != itask) {
                message += "\n\tTask=" + std::to_string(itask) + " [Nsend,Nrecv]=[" + std::to_string(Nsend[itask]) + "," + std::to_string(Nrecv[itask*NProcs + ThisTask]) + "]";
            }
        }
    }
    std::cout<<message<<std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    auto NewNlocal = Nlocal-ntotsend+ntotrecv;
    data.resize(NewNlocal);
    Nlocal -= ntotsend;
    auto time2 = NewTimer();
    std::vector<MPI_Request> sendreqs, recvreqs;
    //asynchronous sends 
    for (auto isend=0;isend<NProcs;isend++) 
    {
        if (isend != ThisTask) 
        {
            MPI_Request request;
            int tag = isend + ThisTask * NProcs;
            if (Nsend[isend] > 0) 
            {
                auto nbytes = Nsend[isend] * sizeof(PointData);
                void *p1 = &Pbuf[noffset[isend]];
                MPI_Isend(p1, nbytes, MPI_BYTE, isend, tag, MPI_COMM_WORLD, &request);
                sendreqs.push_back(request);
                p1 = nullptr;
            }
        }
    }
    //asynchronous receives but after one is received, update local particles
    for (auto irecv=0;irecv<NProcs;irecv++) 
    {
        if (irecv != ThisTask) 
        {
            MPI_Request request;
            int tag = ThisTask + irecv * NProcs;
            auto nrecv = Nrecv[irecv*NProcs + ThisTask];
            int nbytes = nrecv * sizeof(PointData);
            if (nbytes > 0) {
                std::vector<PointData> Precv(nrecv);
                void *p1 = Precv.data();
                MPI_Irecv(p1, nbytes, MPI_BYTE, irecv, tag, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUSES_IGNORE);
                for (auto i=0;i<nrecv;i++) {
                    data[Nlocal++] = Precv[i];
                }
                p1 = nullptr;
            }
        }
    }
    std::cout<<__func__<<": Rank "<<ThisTask<<" now has "<<Nlocal<<std::endl;
    MPIReportTimeStats(time2, __func__, std::to_string(__LINE__));
}



int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &NProcs);
    MPI_Comm_rank(comm, &ThisTask);
    Options opt;

    auto start = std::chrono::system_clock::now();
    std::time_t start_time = std::chrono::system_clock::to_time_t(start);
    if (ThisTask==0) std::cout << "Starting job at " << std::ctime(&start_time);
    if (argc >= 2) opt.npoints = atoi(argv[1]);
    if (argc >= 3) opt.Niter = atoi(argv[2]);
    if (argc >= 4) opt.deltap = atof(argv[3]);
    if (argc >= 5) opt.icompute = atoi(argv[4]);
    if (argc >= 6) opt.iverbose = atoi(argv[5]);

    
    MPILog0ParallelAPI();
    MPI_Barrier(MPI_COMM_WORLD);
    if (opt.iverbose) MPILog0Binding();
    MPI_Barrier(MPI_COMM_WORLD);
    auto timegenerate = NewTimer();
    auto [Nlocal, data] = GenerateData(opt);
    RedistributeData(opt, Nlocal, data);
    LogTimeTaken(timegenerate);
    auto timeloop = NewTimer();
    for (auto i=0;i<opt.Niter;i++) 
    {
        if (ThisTask == 0) std::cout<<"At iteration "<<i<<std::endl;
        TransformData(opt, Nlocal, data);
        auto griddata = GridData(opt, Nlocal, data);
#ifdef USEFFTW
        FFTData(opt, griddata); // currently no external fftw pull
#endif
        if (opt.icompute) {
            auto computedata = ComputeWithData(opt, Nlocal, data, griddata);
        }
        RedistributeData(opt, Nlocal, data);
    }
    LogTimeTaken(timeloop);

    auto end = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    if (ThisTask==0) std::cout << "Ending job at " << std::ctime(&end_time);
    MPI_Finalize();
    return 0;
}
