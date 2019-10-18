/// @copyright (c) 2019 CSIRO
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
/// pgc++ -O3 -fstrict-aliasing -fcx-limited-range -Wall -c tConvolveACC.cc
/// pgc++ -O3 -fstrict-aliasing -fcx-limited-range -Wall -c Stopwatch.cc
/// pgc++ -O3 -fstrict-aliasing -fcx-limited-range -Wall -c Benchmark.cc
/// pgc++ -o tConvolveACC tConvolveACC.o Stopwatch.o Benchmark.o 
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
#include "tConvolveACC.h"

// System includes
#include <stdlib.h>
#include <iostream>

// BLAS includes
#ifdef USEBLAS

#define CAXPY cblas_caxpy
#define CDOTU_SUB cblas_cdotu_sub

#include <mkl_cblas.h>

#endif

// Local includes
#include "Benchmark.h"
#include "Stopwatch.h"

// Main testing routine
int main(int argc, char *argv[])
{

    // Setup the benchmark class
    Benchmark bmark;

    for (int run=0; run<5; ++run) {

        bmark.setRunType(run);

        std::cout << "+++++ Test "<<bmark.getRunType()<<" +++++" << std::endl;

        bmark.init();

        Stopwatch sw;
        double time1, time2;
 
        // Determine how much work will be done across all ranks
        const double ngridvis = double(bmark.nVisibilitiesGridded());
        const double ngridpix = double(bmark.nPixelsGridded());
        if (ngridpix<0) {
            std::cout << "nPixelsGridded() error: "<<ngridpix << std::endl;
            exit(1);
        }
 
        sw.start();
        bmark.runGrid();
        time1 = sw.stop();

        // Report on timings (master reports only)
        std::cout << "  Forward processing (CPU)" << std::endl;
        std::cout << "    Time " << time1 << " (s) " << std::endl;
        std::cout << "    Time per visibility spectral sample " << 1e6*time1 / ngridvis << " (us) " << std::endl;
        std::cout << "    Time per gridding   " << 1e9*time1 / ngridpix << " (ns) " << std::endl;
        std::cout << "    Gridding rate   " << (ngridvis/1e6)/time1 << " (million vis/sec)" << std::endl;
        std::cout << "    Gridding rate   " << (ngridpix/1e6)/time1 << " (million pix/sec)" << std::endl;
 
        sw.start();
        bmark.runGridACC();
        time2 = sw.stop();
  
        // Report on timings (master reports only)
        std::cout << "  Forward processing (OpenACC)" << std::endl;
        std::cout << "    Time " << time2 << " (s) = CPU / " << time1/time2 << std::endl;
        std::cout << "    Time per visibility spectral sample " << 1e6*time2 / ngridvis << " (us) " << std::endl;
        std::cout << "    Time per gridding   " << 1e9*time2 / ngridpix << " (ns) " << std::endl;
        std::cout << "    Gridding rate   " << (ngridvis/1e6)/time2 << " (million vis/sec)" << std::endl;
        std::cout << "    Gridding rate   " << (ngridpix/1e6)/time2 << " (million pix/sec)" << std::endl;
 
        // Report on accuracy
        std::cout << "  Verifying results..." << std::endl;
        bmark.runGridCheck();
 
        sw.start();
        bmark.runDegrid();
        time1 = sw.stop();
 
        // Report on timings (master reports only)
        std::cout << "  Reverse processing (CPU)" << std::endl;
        std::cout << "    Time " << time1 << " (s) " << std::endl;
        std::cout << "    Time per visibility spectral sample " << 1e6*time1 / ngridvis << " (us) " << std::endl;
        std::cout << "    Time per degridding " << 1e9*time1 / ngridpix << " (ns) " << std::endl;
        std::cout << "    Degridding rate " << (ngridvis/1e6)/time1 << " (million vis/sec)" << std::endl;
        std::cout << "    Degridding rate " << (ngridpix/1e6)/time1 << " (million pix/sec)" << std::endl;
  
        sw.start();
        bmark.runDegridACC();
        time2 = sw.stop();
 
        // Report on timings (master reports only)
        std::cout << "  Reverse processing (OpenACC)" << std::endl;
        std::cout << "    Time " << time2 << " (s) = CPU / " << time1/time2 << std::endl;
        std::cout << "    Time per visibility spectral sample " << 1e6*time2 / ngridvis << " (us) " << std::endl;
        std::cout << "    Time per degridding " << 1e9*time2 / ngridpix << " (ns) " << std::endl;
        std::cout << "    Degridding rate " << (ngridvis/1e6)/time2 << " (million vis/sec)" << std::endl;
        std::cout << "    Degridding rate " << (ngridpix/1e6)/time2 << " (million pix/sec)" << std::endl;
 
        // Report on accuracy
        std::cout << "  Verifying results..." << std::endl;
        bmark.runDegridCheck();

        std::cout << "Done" << std::endl;

    }

    return 0;
}
