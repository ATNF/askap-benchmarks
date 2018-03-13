/// @copyright (c) 2007 CSIRO
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
/// @detail
/// This C++ program has been written to demonstrate the convolutional resampling
/// algorithm used in radio interferometry. It should compile with:
///
/// $ g++ -O3 -fstrict-aliasing -fcx-limited-range -Wall -c tConvolveMT.cc
/// $ g++ -O3 -fstrict-aliasing -fcx-limited-range -Wall -c Stopwatch.cc
/// $ g++ -O3 -fstrict-aliasing -fcx-limited-range -Wall -c Benchmark.cc
/// $ g++ -o tConvolveMT tConvolveMT.o Stopwatch.o Benchmark.o -lpthread
///
/// -fstrict-aliasing - tells the compiler that there are no memory locations
///                     accessed through aliases.
/// -fcx-limited-range - states that a range reduction step is not needed when
///                      performing complex division. This is an acceptable
///                      optimization.
///
/// @author Ben Humphreys <ben.humphreys@csiro.au>
/// @author Tim Cornwell  <tim.cornwell@csiro.au>

// Include own header file first
#include "tConvolveMT.h"

/// System includes
#include <iostream>
#include <vector>
#include <pthread.h>
#include <stdlib.h>

/// BLAS includes
#ifdef USEBLAS

#define CAXPY cblas_caxpy
#define CDOTU_SUB cblas_cdotu_sub

#include <mkl_cblas.h>

#endif

/// Local includes
#include "tConvolveMT.h"
#include "Benchmark.h"
#include "Stopwatch.h"

void *gridThread(void *arg)
{
    Benchmark *gp = static_cast<Benchmark*>(arg);
    gp->runGrid();
    return NULL;
}

void *degridThread(void *arg)
{
    Benchmark *gp = static_cast<Benchmark*>(arg);
    gp->runDegrid();
    return NULL;
}

// Main testing routine
int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " < # threads >" << std::endl;
        return 1;
    }

    const int nthreads = atoi(argv[1]);

    std::vector<Benchmark> gp(nthreads);

    for (int i = 0; i < nthreads; ++i) {
        gp[i].init();
    }

    std::vector<pthread_t> gridThreads(nthreads);
    std::vector<pthread_t> degridThreads(nthreads);

    const int sSize = 2 * gp[0].getSupport() + 1;
    double griddings = (double(nSamples * nChan) * double((sSize) * (sSize))) * double(nthreads);

    // Now we can do the timing
    std::cout << "+++++ Forward processing +++++" << std::endl;


    Stopwatch sw;

    sw.start();

    for (int i = 0; i < nthreads; ++i) {
        pthread_create(&gridThreads[i], NULL, gridThread, (void *)&(gp[i]));
    }

    for (int i = 0; i < nthreads; ++i) {
        pthread_join(gridThreads[i], 0);
    }

    double time = sw.stop();

    // Report on timings
    std::cout << "    Time " << time << " (s) " << std::endl;
    std::cout << "    Time per visibility spectral sample " << 1e6*time / double(nSamples*nChan) << " (us) " << std::endl;
    std::cout << "    Time per gridding   " << 1e9*time / (double(nSamples*nChan)* double((sSize)*(sSize))) << " (ns) " << std::endl;
    std::cout << "    Gridding rate   " << (griddings / 1000000) / time << " (million grid points per second)" << std::endl;


    std::cout << "+++++ Reverse processing +++++" << std::endl;

    sw.start();

    for (int i = 0; i < nthreads; ++i) {
        pthread_create(&degridThreads[i], NULL, degridThread, (void *)&(gp[i]));
    }

    for (int i = 0; i < nthreads; ++i) {
        pthread_join(degridThreads[i], 0);
    }

    time = sw.stop();

    // Report on timings
    std::cout << "    Time " << time << " (s) " << std::endl;
    std::cout << "    Time per visibility spectral sample " << 1e6*time / double(nSamples*nChan) << " (us) " << std::endl;
    std::cout << "    Time per degridding " << 1e9*time / (double(nSamples*nChan)* double((sSize)*(sSize))) << " (ns) " << std::endl;
    std::cout << "    Degridding rate " << (griddings / 1000000) / time << " (million grid points per second)" << std::endl;

    std::cout << "Done" << std::endl;

    return 0;
}
