/// @file mpiperf.cc
///
/// @copyright (c) 2017 CSIRO
/// Australia Telescope National Facility (ATNF)
/// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
/// PO Box 76, Epping NSW 1710, Australia
/// atnf-enquiries@csiro.au
///
/// This file is part of the ASKAP software distribution.
///
/// The ASKAP software distribution is free software: you can redistribute it
/// and/or modify it under the terms of the GNU General Public License as
/// published by the Free Software Foundation; either version 2 of the License,
/// or (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program; if not, write to the Free Software
/// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
///
/// @author Stephen Ord <stephen.ord@csiro.au>

// Include package level header file
#include "askap_mpiperf.h"

// System includes
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <mpi.h>
#include <pthread.h>
#include <string.h> // for memcpy
#include <sys/timeb.h>
#include <chrono>

// ASKAPsoft includes
#include "CommandLineParser.h"
#include "Common/ParameterSet.h"
#include "casacore/casa/OS/Timer.h"

#define BLOCKSIZE 4*1024*1024


// the output file
FILE *fptr=NULL;

// the log file

std::ofstream logfile;

int rank, wsize;

#define INFLUXDB_LOG(func,tag) {\
  { \
    casa::Timer work; \
    work.mark(); \
    func; \
    std::chrono::nanoseconds ns = std::chrono::duration_cast< std::chrono::nanoseconds >(std::chrono::system_clock::now().time_since_epoch()); \
    logfile << tag  << "rank=" << rank << " " << "time_secs=" << work.real() << " " << ns.count() << std::endl; \
  } \
}
#define TIME_FUNC(func,tag) { \
    { \
        casa::Timer work; \
        work.mark(); \
        func; \
        std::cout << "#" << tag << ":" << work.real() << std::endl; \
    } \
}
// Using
using LOFAR::ParameterSet;


/* the mutex lock */

pthread_mutex_t full_lock;
pthread_mutex_t swap_lock;

pthread_cond_t buffer_full;
pthread_cond_t buffer_swapped;

static ParameterSet getParameterSet(int argc, char *argv[])
{
    cmdlineparser::Parser parser;

    // Command line parameter
    cmdlineparser::FlaggedParameter<std::string> inputsPar("-c", "config.in");

    // This parameter is optional, default will be returned if not present
    parser.add(inputsPar, cmdlineparser::Parser::return_default);

    parser.process(argc, const_cast<char**> (argv));

    const std::string parsetFile = inputsPar;

    LOFAR::ParameterSet parset(parsetFile);

    return parset;
}
void doWrite(FILE *fptr,size_t towrite,size_t write_block,char *buffptr) {

    int rtn;

    while (towrite>0) {

      if (towrite < write_block)
          write_block = towrite;

      if (fptr != NULL) {
          rtn=fwrite(buffptr,write_block,1,fptr);
          if (rtn!=1) {
              std::cout << "#WARNING - failed write" << std::endl;
          }
          else {
              towrite = towrite - write_block;
              buffptr = buffptr+write_block;
          }
      }
      else {
          std::cout << "#WARNING - not writing"<< std::endl;
          towrite = 0;
      }
  }
}
void doWorkRoot(void *buffer, size_t buffsize, float *workTime,FILE *fptr) {


    casa::Timer work;


    int rtn=0;


    TIME_FUNC(pthread_mutex_lock(&swap_lock),"mutex action=aquire,");
    std::cout << "#Write: released the swap lock and waiting for signal" << std::endl;
    // wait and release lock
    TIME_FUNC(pthread_cond_wait(&buffer_swapped,&swap_lock),"mutex action=wait,");

    std::cout << "#Write: acquired the swap lock" << std::endl;

    size_t towrite=buffsize;
    size_t write_block=BLOCKSIZE;
    char * buffptr= (char *) buffer;
    work.mark();
    // here's hoping this is recursive
    INFLUXDB_LOG(doWrite(fptr,towrite,write_block,buffptr),"file action=write,");

    TIME_FUNC(pthread_mutex_unlock(&swap_lock),"mutex action=unlock,");
    *workTime = work.real();
    std::cout << "#Write:Actual write:" << *workTime << std::endl;


}
void doWorkWorker(void *buffer) {

}

typedef struct {
    char *in;
    char *out;
    size_t nelements;
    int intTime;
} thread_args ;

/* this function is run by the second thread */
void *thread_x(void *arg)
{

    thread_args *x_ptr = (thread_args *) arg;
    while (1) {
        // wait for a full buffer
        TIME_FUNC(pthread_mutex_lock (&full_lock),"mutex action=lock,");
        std::cout << "#Worker: releasing full_lock and waiting" << std::endl;
        TIME_FUNC(pthread_cond_wait(&buffer_full,&full_lock),"mutex action=wait,");
        std::cout << "#Worker: acquired full_lock" << std::endl;
        // release wait and release lock
        // do something
        usleep(x_ptr->intTime*1E6);
        INFLUXDB_LOG(memcpy(x_ptr->out, x_ptr->in,x_ptr->nelements*sizeof(float)),"memory,action=copy,");

        TIME_FUNC(pthread_mutex_unlock (&full_lock),"mutex action=unlock,");
        TIME_FUNC(pthread_mutex_lock (&swap_lock),"mutex action=lock,");
        TIME_FUNC(pthread_cond_signal(&buffer_swapped),"mutex action=signal,");
        TIME_FUNC(pthread_mutex_unlock (&swap_lock),"mutex action=unlock,");

    }
}
void transpose(float *in, float *out) {

}


int main(int argc, char *argv[])
{
    // MPI init
    int provided;
    // This is correct initializer

    std::cout << "#Attempting MPI " << std::endl;
    MPI_Init_thread(&argc, &argv,MPI_THREAD_FUNNELED,&provided);
    //
    // This is not correct
    //
    // MPI_Init(&argc, &argv);
    //


    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

    std::cout << "#Rank " << rank << " started" << std::endl;

    ParameterSet parset = getParameterSet(argc, argv);
    ParameterSet subset(parset.makeSubset("mpiperf."));


    std::string filename = subset.getString("filename","data");
    std::string logname = subset.getString("logname","test");

    // initialise the mutex lock
    pthread_mutex_init(&full_lock, NULL);
    pthread_cond_init (&buffer_full, NULL);
    pthread_mutex_init(&swap_lock, NULL);
    pthread_cond_init (&buffer_swapped, NULL);
    // thread attributes
    pthread_t x_thread;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    thread_args work_dat;

    int intTime = subset.getInt32("integrationTime",5);
    int integrations = subset.getInt32("nIntegrations",1);
    int antennas = subset.getInt32("nAntenna",36);
    int channels = subset.getInt32("nChan",2048);
    int beams = subset.getInt32("nFeeds",36);
    int pol = subset.getInt32("nPol",4);
    int maxfilesizeMB = subset.getInt32("maxfilesizeMB",0);

    int baselines = (antennas*(antennas-1)/2);

    size_t nElements = baselines*channels*beams*pol*2;
    size_t sendBufferSize = nElements*sizeof(float);
    size_t recvBufferSize = wsize*sendBufferSize;

    int intPerFile = integrations;

    if (maxfilesizeMB != 0) {
        float temp = recvBufferSize/(1024*1024);
        temp = maxfilesizeMB/temp;
        intPerFile = ceil(temp);
    }


    float *sBuf = (float *) malloc(sendBufferSize);
    float *rBuf = (float *) malloc(recvBufferSize);
    float *oBuf = (float *) malloc(recvBufferSize);

    work_dat.in = (char *) rBuf;
    work_dat.out = (char *) oBuf;
    work_dat.nelements = wsize*nElements;
    work_dat.intTime = intTime;

    int *displs = (int *)malloc(wsize*sizeof(int));
    int *rcounts = (int *)malloc(wsize*sizeof(int));

    for (int i=0; i<wsize; ++i) {
       displs[i] = i*nElements;
       rcounts[i] = nElements;
    }

    //open the logptr
    std::ostringstream oss;
    oss << logname << "_" << rank << ".log";
    logfile.open(oss.str().c_str(), std::ios::out | std::ios::trunc);


    casa::Timer timer;
    casa::Timer total;
    total.mark();
    if (rank == 0) {
        std::cout << "#Gathering and Writing " << integrations << " integrations of " << intTime << " seconds " << std::endl;
        std::cout << "#There are " << wsize << " blocks of " << channels << " channels " << std::endl;
        std::cout << "#With " << antennas << " antennas and " << beams << " beams " << std::endl;
        std::cout << "#For a datasize (in Mbytes) per integration of " << sendBufferSize/(1024*1024) << " per rank and " << recvBufferSize/(1024*1024) << " in total " << std::endl;
        std::cout << "#Datarate in MB/s is " << recvBufferSize/(intTime*1024*1024) << std::endl;
        if (maxfilesizeMB !=0) {
            std::cout << "#Integrations per file " << intPerFile << std::endl;
        }
        // Spawn a thread


        std::cout << "#Spawning a thread" << std::endl;

        if (pthread_create(&x_thread, &attr, thread_x, &work_dat)) {

            fprintf(stderr, "#Error creating thread\n");
            return 1;

        }

    }

    for (int i = 0; i < integrations; ++i) {

        if (i==0 || i%intPerFile == 0) {
            if (fptr != NULL) {
                INFLUXDB_LOG(fclose(fptr),"file,action=close,");
            }
            std::ostringstream oss;
            oss << filename << "_" << i << ".dat";
            fptr = fopen(oss.str().c_str(),"w");
            assert(fptr);
            setvbuf(fptr,NULL,recvBufferSize,_IOFBF);

        }
        timer.mark();
        doWorkWorker(sBuf);

        if (rank == 0) {
            TIME_FUNC(pthread_mutex_lock(&full_lock),"mutex action=lock,");
            std::cout << "#Starting the Gathering" << std::endl;
        }
        INFLUXDB_LOG(MPI_Gatherv((void *) sBuf,nElements,MPI_FLOAT,(void *) rBuf,rcounts,displs,MPI_FLOAT,0,MPI_COMM_WORLD),"mpi,action=gather,");
        MPI_Barrier(MPI_COMM_WORLD);

        // Report progress
        if (rank == 0) {
            const float realtime = timer.real();
            const float perf = static_cast<float>(intTime) / realtime;
            if (perf < 1) {
                std::cout << "#WARNING ";
            }
            std::cout << "#MPI Gather for integration " << i <<
            " in " << realtime << " seconds"
            << " (" << perf << "x requirement)" << std::endl;


            TIME_FUNC(pthread_cond_signal(&buffer_full),"mutex action=signal,");
            TIME_FUNC(pthread_mutex_unlock(&full_lock),"mutex action=unlock,");

            float workTime;


            // aquire lock
            //first get the lock so things stay sync'd

            INFLUXDB_LOG(doWorkRoot(rBuf,recvBufferSize,&workTime,fptr),"file,action=write,");



            // release lock
        }
    }

    // Report totals
    if (rank == 0) {
        const float realtime = total.real();
        const float perf = static_cast<float>(intTime * integrations) / realtime;
        std::cout << "#Received " << integrations << " integrations "
            " in " << realtime << " seconds"
            << " (" << perf << "x requirement)" << std::endl;
    }
    if (fptr != NULL) {
        fclose(fptr);
    }

    if (logfile.is_open()) {
      logfile.close();
    }
    pthread_attr_destroy(&attr);
    // pthread_kill(x_thread,SIGKILL);
    // pthread_join(x_thread, NULL);
    pthread_mutex_destroy(&full_lock);
    pthread_mutex_destroy(&swap_lock);
    pthread_cond_destroy(&buffer_full);
    pthread_cond_destroy(&buffer_swapped);

    free(sBuf);
    free(rBuf);
    free(oBuf);
    free(displs);
    free(rcounts);
    MPI_Finalize();

    // pthread_exit(NULL);
}
