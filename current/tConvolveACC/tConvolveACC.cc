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
/// algorithm used in radio interferometry. See Makefile for compilation options 
///
/// @author Ben Humphreys <ben.humphreys@csiro.au>
/// @author Tim Cornwell  <tim.cornwell@csiro.au>
/// @author Daneil Mitchell <daniel.mitchell@csiro.au>

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

    // whether or not to sort visibilities. 0 = no sorting, 1 = sort by w-plane
    bmark.setSort(0);

    // get required gridding rates
    std::vector<float> rates = bmark.requiredRate();

    for (int run=0; run<2; ++run) {

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

        // Report on timings
        std::cout << "  Forward processing (CPU)" << std::endl;
        std::cout << "    Time " << time1 << " (s) " << std::endl;
        std::cout << "    Time per visibility spectral sample " << 1e6*time1 / ngridvis << " (us) " << std::endl;
        std::cout << "    Time per gridding   " << 1e9*time1 / ngridpix << " (ns) " << std::endl;
        std::cout << "    Gridding rate   "<<(ngridvis/1e6)/time1<<" (Mvis/sec)" << std::endl;
        std::cout << "    Gridding rate   "<<(ngridpix/1e6)/time1<<" (Mpix/sec)" << std::endl;
 
        sw.start();
        bmark.runGridACC();
        time2 = sw.stop();
  
        // Report on timings
        std::cout << "  Forward processing (OpenACC)" << std::endl;
        std::cout << "    Time " << time2 << " (s) = CPU / " << time1/time2 << std::endl;
        std::cout << "    Time per visibility spectral sample " << 1e6*time2 / ngridvis << " (us) " << std::endl;
        std::cout << "    Time per gridding   " << 1e9*time2 / ngridpix << " (ns) " << std::endl;
        std::cout << "    Gridding rate   "<<(ngridvis/1e6)/time2<<" (Mvis/sec)" << std::endl;
        std::cout << "    Gridding rate   "<<(ngridpix/1e6)/time2<<" (Mpix/sec)" << std::endl;
        if (run==0) {
            std::cout << "    Continuum gridding performance:   " << (ngridpix/1e6)/time2 << " (Mpix/sec) / "
                      << rates[0]/1e6 << " (Mpix/sec) = " << ngridpix/time2/rates[0]<<"x CPU requirement" << std::endl;
        }
        if (run==1) {
            std::cout << "    Spectral gridding performance:    " << (ngridpix/1e6)/time2 << " (Mpix/sec) / "
                      << rates[1]/1e6 << " (Mpix/sec) = " << ngridpix/time2/rates[1]<<"x CPU requirement" << std::endl;
        }
 
        // Report on accuracy
        std::cout << " t"<<run<<" Verifying results..." << std::endl;
        bmark.runGridCheck();
 
        sw.start();
        bmark.runDegrid();
        time1 = sw.stop();
 
        // Report on timings
        std::cout << "  Reverse processing (CPU)" << std::endl;
        std::cout << "    Time " << time1 << " (s) " << std::endl;
        std::cout << "    Time per visibility spectral sample " << 1e6*time1 / ngridvis << " (us) " << std::endl;
        std::cout << "    Time per degridding " << 1e9*time1 / ngridpix << " (ns) " << std::endl;
        std::cout << "    Degridding rate "<<(ngridvis/1e6)/time1<<" (Mvis/sec)" << std::endl;
        std::cout << "    Degridding rate "<<(ngridpix/1e6)/time1<<" (Mpix/sec)" << std::endl;
  
        sw.start();
        bmark.runDegridACC();
        time2 = sw.stop();
 
        // Report on timings
        std::cout << "  Reverse processing (OpenACC)" << std::endl;
        std::cout << "    Time " << time2 << " (s) = CPU / " << time1/time2 << std::endl;
        std::cout << "    Time per visibility spectral sample " << 1e6*time2 / ngridvis << " (us) " << std::endl;
        std::cout << "    Time per degridding " << 1e9*time2 / ngridpix << " (ns) " << std::endl;
        std::cout << "    Degridding rate "<<(ngridvis/1e6)/time2<<" (Mvis/sec)" << std::endl;
        std::cout << "    Degridding rate "<<(ngridpix/1e6)/time2<<" (Mpix/sec)" << std::endl;
        if (run==0) {
            std::cout << "    Continuum degridding performance:   " << (ngridpix/1e6)/time2 << " (Mpix/sec) / "
                      << rates[0]/1e6 << " (Mpix/sec) = " << ngridpix/time2/rates[0]<<"x CPU requirement" << std::endl;
        }
        if (run==1) {
            std::cout << "    Spectral degridding performance:    " << (ngridpix/1e6)/time2 << " (Mpix/sec) / "
                      << rates[1]/1e6 << " (Mpix/sec) = " << ngridpix/time2/rates[1]<<"x CPU requirement" << std::endl;
        }
 
        // Report on accuracy
        std::cout << " t"<<run<<" Verifying results..." << std::endl;
        bmark.runDegridCheck();

        std::cout << "Done" << std::endl;

    }

    return 0;
}
