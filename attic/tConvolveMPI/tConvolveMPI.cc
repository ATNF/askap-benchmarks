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
/// mpicxx -O3 -fstrict-aliasing -fcx-limited-range -Wall -c tConvolveMPI.cc
/// mpicxx -O3 -fstrict-aliasing -fcx-limited-range -Wall -c Stopwatch.cc
/// mpicxx -O3 -fstrict-aliasing -fcx-limited-range -Wall -c Benchmark.cc
/// mpicxx -o tConvolveMPI tConvolveMPI.o Stopwatch.o Benchmark.o 
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
#include "tConvolveMPI.h"
#include "Benchmark.h"
#include "Stopwatch.h"
#include "common.h"

// System & MPI includes
#include <mpi.h>

// BLAS includes
#ifdef USEBLAS

#define CAXPY cblas_caxpy
#define CDOTU_SUB cblas_cdotu_sub

#include <mkl_cblas.h>

#endif

// Local includes

// Main testing routine
int main(int argc, char *argv[])
{
    // Initialize MPI
    int rc = MPI_Init(&argc, &argv);

    if (rc != MPI_SUCCESS) {
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Options opt
    getinput(argc, argv, opt);
    // Change these if necessary to adjust run time
    int nSamples = opt.nSamples;
    int wSize = opt.wSize;
    int nChan = opt.nChan;
    Coord cellSize = opt.cellSize;
    const int gSize = opt.gSize;
    const int baseline = opt.baseline; 

    // Setup the benchmark class
    Benchmark bmark;
    bmark.init();

    // Determine how much work will be done across all ranks
    const int sSize = 2 * bmark.getSupport() + 1;
    const double griddings = (double(nSamples * nChan) * double((sSize) * (sSize))) * double(numtasks);

    if (rank == 0) {
        std::cout << "+++++ Forward processing (MPI) +++++" << std::endl;
        std::cout << "    Number of processes: " << numtasks << std::endl;
    }

    Stopwatch sw;
    MPI_Barrier(MPI_COMM_WORLD);
    sw.start();
    bmark.runGrid();
    MPI_Barrier(MPI_COMM_WORLD);
    double time = sw.stop();

    // Report on timings (master reports only)
    if (rank == 0) {
        report_timings(time, opt, sSize, griddings);
        std::cout << "Done" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "+++++ Reverse processing (MPI) +++++" << std::endl;
        std::cout << "    Number of processes: " << numtasks << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    sw.start();
    bmark.runDegrid();
    MPI_Barrier(MPI_COMM_WORLD);
    time = sw.stop();

    // Report on timings (master reports only)
    if (rank == 0) {
        report_timings(time, opt, sSize, griddings);
        std::cout << "Done" << std::endl;
    }

    MPI_Finalize();

    return 0;
}
