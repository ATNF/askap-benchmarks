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

// ASKAPsoft includes
#include "CommandLineParser.h"
#include "Common/ParameterSet.h"
#include "casacore/casa/OS/Timer.h"

#define BLOCKSIZE 4*1024*1024

// Using
using LOFAR::ParameterSet;

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
void doWorkRoot(void *buffer, size_t buffsize, float *workTime,FILE *fptr) {

    casa::Timer work;
    int rtn=0;
    work.mark();
    size_t towrite=buffsize;
    size_t write_block=BLOCKSIZE;
    char * buffptr= (char *) buffer;
    while (towrite>0) {
        if (fptr != NULL) {
            rtn=fwrite(buffptr,write_block,1,fptr);
            if (rtn!=1) {
                std::cout << "WARNING - failed write" << std::endl;
            }
            else {
                towrite = towrite - write_block;
                buffptr = buffptr+write_block;
            }
        }
        else {
            std::cout << "WARNING - not writing"<< std::endl;
            towrite = 0;
        }
    }

    *workTime = work.real();



}
void doWorkWorker(void *buffer) {

}


int main(int argc, char *argv[])
{
    // MPI init
    MPI_Init(&argc, &argv);
    int rank,wsize;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

    ParameterSet parset = getParameterSet(argc, argv);
    ParameterSet subset(parset.makeSubset("mpiperf."));

    // Replace in the filename the %w pattern with the rank number
    std::string filename = subset.getString("filename");

    // create the output file
    FILE *fptr=NULL;




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

    int *displs = (int *)malloc(wsize*sizeof(int));
    int *rcounts = (int *)malloc(wsize*sizeof(int));

    for (int i=0; i<wsize; ++i) {
       displs[i] = i*nElements;
       rcounts[i] = nElements;
    }

    casa::Timer timer;
    casa::Timer total;
    total.mark();
    if (rank == 0) {
        std::cout << "Gathering and Writing " << integrations << " integrations of " << intTime << " seconds " << std::endl;
        std::cout << "There are " << wsize << " blocks of " << channels << " channels " << std::endl;
        std::cout << "With " << antennas << " antennas and " << beams << " beams " << std::endl;
        std::cout << "For a datasize (in Mbytes) per integration of " << sendBufferSize/(1024*1024) << " per rank and " << recvBufferSize/(1024*1024) << " in total " << std::endl;
        std::cout << "Datarate in MB/s is " << recvBufferSize/(intTime*1024*1024) << std::endl;
        if (maxfilesizeMB !=0) {
            std::cout << "Integrations per file " << intPerFile << std::endl;
        }
    }

    for (int i = 0; i < integrations; ++i) {

        if (i==0 || i%intPerFile == 0) {
            if (fptr != NULL) {
                fclose(fptr);
            }
            std::ostringstream oss;
            oss << filename << "_" << i << ".dat";
            fptr = fopen(oss.str().c_str(),"w");
            assert(fptr);
            setvbuf(fptr,NULL,BLOCKSIZE,_IOFBF);

        }
        timer.mark();
        doWorkWorker(sBuf);
        MPI_Gatherv((void *) sBuf,nElements,MPI_FLOAT,(void *) rBuf,rcounts,displs,MPI_FLOAT,0,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        // Report progress
        if (rank == 0) {
            const float realtime = timer.real();
            const float perf = static_cast<float>(intTime) / realtime;
            if (perf < 1) {
                std::cout << "WARNING ";
            }
            std::cout << "Received integration " << i <<
            " in " << realtime << " seconds"
            << " (" << perf << "x requirement)" << std::endl;
            std::cout << "Doing some work" << std::endl;
            float workTime;
            doWorkRoot(rBuf,recvBufferSize,&workTime,fptr);
            std::cout << "Wrote integration " << i <<  " in "
            << workTime << " seconds" << std::endl;
            float combinedTime = workTime + realtime;
            if (combinedTime < intTime) {
                useconds_t timetosleep = (useconds_t) 1000.0*(intTime-combinedTime);
                usleep(timetosleep);
            }
            else {
                std::cout << "WARNING combined time greater than integration time" << std::endl;
            }
        }
    }

    // Report totals
    if (rank == 0) {
        const float realtime = total.real();
        const float perf = static_cast<float>(intTime * integrations) / realtime;
        std::cout << "Received " << integrations << " integrations "
            " in " << realtime << " seconds"
            << " (" << perf << "x requirement)" << std::endl;
    }
    if (fptr != NULL) {
        fclose(fptr);
    }
    free(sBuf);
    free(rBuf);
    free(displs);
    free(rcounts);
    MPI_Finalize();

    return 0;
}
