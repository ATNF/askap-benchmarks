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
#include <mpi.h>

// ASKAPsoft includes
#include "CommandLineParser.h"
#include "Common/ParameterSet.h"
#include "casacore/casa/OS/Timer.h"


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
void doWorkRoot(void *buffer) {

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

    int intTime = subset.getInt32("integrationTime");
    int integrations = subset.getInt32("nIntegrations");
    int antennas = subset.getInt32("nAntenna");
    int channels = subset.getInt32("nChan");
    int beams = subset.getInt32("nFeeds");
    int pol = subset.getInt32("nPol");

    int baselines = (antennas*pol*(antennas*pol-pol)/2);

    size_t nElements = baselines*channels*beams*2;
    size_t sendBufferSize = nElements*sizeof(float);
    size_t recvBufferSize = wsize*sendBufferSize;

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
    for (int i = 0; i < integrations; ++i) {

        timer.mark();
        doWorkWorker(sBuf);
        MPI_Gatherv((void *) sBuf,nElements,MPI_FLOAT,(void *) rBuf,rcounts,displs,MPI_FLOAT,0,MPI_COMM_WORLD);
        if (rank == 0) {
            doWorkRoot(sBuf);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // Report progress
        if (rank == 0) {
            const float realtime = timer.real();
            const float perf = static_cast<float>(intTime) / realtime;
            std::cout << "Received integration " << i <<
            " in " << realtime << " seconds"
            << " (" << perf << "x requirement)" << std::endl;
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
    free(sBuf);
    free(rBuf);
    free(displs);
    free(rcounts);
    MPI_Finalize();

    return 0;
}
