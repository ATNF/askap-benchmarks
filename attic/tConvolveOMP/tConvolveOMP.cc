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
/// @author Ben Humphreys <ben.humphreys@csiro.au>
/// @author Tim Cornwell  <tim.cornwell@csiro.au>

// Local includes
#include "../tConvolveCommon/common.h"
#include "Stopwatch.h"

// system includes
#include <cassert>

// OpenMP includes
#include <omp.h>


// BLAS includes
#ifdef USEBLAS

#define CAXPY cblas_caxpy
#define CDOTU_SUB cblas_cdotu_sub

#include <mkl_cblas.h>

#endif

#if defined(GRIDING)  
	#define SERIAL_GRIDING 1
	#define OMP_GRIDING 1
	#define VERIFY_GRIDING 1
#elif defined(SERIAL_GRIDING)
	#define SERIAL_GRIDING 1
#elif defined(OMP_GRIDING)
	#define OMP_GRIDING 1


#elif defined(DEGRIDING)
	#define SERIAL_DEGRIDING 1
	#define OMP_DEGRIDING 1
	#define VERIFY_DEGRIDING 1
#elif defined(SERIAL_DEGRIDING)
	#define SERIAL_DEGRIDING 1
#elif defined(OMP_DEGRIDING)
	#define OMP_DEGRIDING 1
#else
	#define SERIAL_GRIDING 1
	#define OMP_GRIDING 1
	#define VERIFY_GRIDING 1

	#define SERIAL_DEGRIDING 1
	#define OMP_DEGRIDING 1
	#define VERIFY_DEGRIDING 1
#endif  

/////////////////////////////////////////////////////////////////////////////////
// The next two functions are the kernel of the gridding/degridding.
// The data are presented as a vector. Offsets for the convolution function
// and for the grid location are precalculated so that the kernel does
// not need to know anything about world coordinates or the shape of
// the convolution function. The ordering of cOffset and iu, iv is
// random - some presorting might be advantageous.
//
// Perform gridding
//
// data - values to be gridded in a 1D vector
// support - Total width of convolution function=2*support+1
// C - convolution function shape: (2*support+1, 2*support+1, *)
// cOffset - offset into convolution function per data point
// iu, iv - integer locations of grid points
// grid - Output grid: shape (gSize, *)
// gSize - size of one axis of grid
void gridKernel(const std::vector<Value>& data, const int support,
                const std::vector<Value>& C, const std::vector<int>& cOffset,
                const std::vector<int>& iu, const std::vector<int>& iv,
                std::vector<Value>& grid, const int gSize)
{
    const int sSize = 2 * support + 1;

    for (int dind = 0; dind < int(data.size()); ++dind) {
        // The actual grid point
        int gind = iu[dind] + gSize * iv[dind] - support;
        // The Convoluton function point from which we offset
        int cind = cOffset[dind];

        for (int suppv = 0; suppv < sSize; suppv++) {
#ifdef USEBLAS
            CAXPY(sSize, &data[dind], &C[cind], 1, &grid[gind], 1);
#else
            Value* gptr = &grid[gind];
            const Value* cptr = &C[cind];
            const Value d = data[dind];

            for (int suppu = 0; suppu < sSize; suppu++) {
                *(gptr++) += d * (*(cptr++));
            }

#endif
            gind += gSize;
            cind += sSize;
        }
    }
}

int gridKernelOMP(const std::vector<Value>& data, const int support,
        const std::vector<Value>& C, const std::vector<int>& cOffset,
        const std::vector<int>& iu, const std::vector<int>& iv,
        std::vector<Value>& grid, const int gSize)
{
    const int sSize = 2 * support + 1;
    #pragma omp parallel default(shared)
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

        for (int dind = 0; dind < int(data.size()); ++dind) {
            // The actual grid point
            int gind = iu[dind] + gSize * iv[dind] - support;
            // The Convoluton function point from which we offset
            int cind = cOffset[dind];
            int row = iv[dind];
            for (int suppv = 0; suppv < sSize; suppv++) {
                if (row % nthreads == tid) {
#ifdef USEBLAS
                    CAXPY(sSize, &data[dind], &C[cind], 1, &grid[gind], 1);
#else
                    Value* gptr = &grid[gind];
                    const Value* cptr = &C[cind];
                    const Value d = data[dind];

                    for (int suppu = 0; suppu < sSize; suppu++) {
                        *(gptr++) += d * (*(cptr++));
                    }
#endif
                }
                gind += gSize;
                cind += sSize;
                row++;
            }
        }
    } // End omp parallel

    return omp_get_max_threads();
}

// Perform degridding
void degridKernel(const std::vector<Value>& grid, const int gSize, const int support,
                  const std::vector<Value>& C, const std::vector<int>& cOffset,
                  const std::vector<int>& iu, const std::vector<int>& iv,
                  std::vector<Value>& data)
{
    const int sSize = 2 * support + 1;

    for (int dind = 0; dind < int(data.size()); ++dind) {
        data[dind] = 0.0;

        // The actual grid point from which we offset
        int gind = iu[dind] + gSize * iv[dind] - support;
        // The Convoluton function point from which we offset
        int cind = cOffset[dind];

        for (int suppv = 0; suppv < sSize; suppv++) {
#ifdef USEBLAS
            Value dot;
            CDOTU_SUB(sSize, &grid[gind], 1, &C[cind], 1, &dot);
            data[dind] += dot;
#else
            Value* d = &data[dind];
            const Value* gptr = &grid[gind];
            const Value* cptr = &C[cind];

            for (int suppu = 0; suppu < sSize; suppu++) {
                (*d) += (*(gptr++)) * (*(cptr++));
            }

#endif
            gind += gSize;
            cind += sSize;
        }

    }
}

int degridKernelOMP(const std::vector<Value>& grid, const int gSize, const int support,
                    const std::vector<Value>& C, const std::vector<int>& cOffset,
                    const std::vector<int>& iu, const std::vector<int>& iv,
                    std::vector<Value>& data)
{
    const int sSize = 2 * support + 1;

    #pragma omp parallel for  \
        default(shared)   \
        schedule(dynamic, 32)
    for (int dind = 0; dind < int(data.size()); ++dind) {
        data[dind] = 0.0;

        // The actual grid point from which we offset
        int gind = iu[dind] + gSize * iv[dind] - support;
        // The Convoluton function point from which we offset
        int cind = cOffset[dind];

        for (int suppv = 0; suppv < sSize; suppv++) {
#ifdef USEBLAS
            Value dot;
            CDOTU_SUB(sSize, &grid[gind], 1, &C[cind], 1, &dot);
            data[dind] += dot;
#else
            Value* d = &data[dind];
            const Value* gptr = &grid[gind];
            const Value* cptr = &C[cind];

            for (int suppu = 0; suppu < sSize; suppu++) {
                (*d) += (*(gptr++)) * (*(cptr++));
            }

#endif
            gind += gSize;
            cind += sSize;
        }

    }

    return omp_get_max_threads();
}

// Main testing routine
int main(int argc, char* argv[])
{
    Options opt;
    getinput(argc, argv, opt);
    // Change these if necessary to adjust run time
    int nSamples = opt.nSamples;
    int wSize = opt.wSize;
    int nChan = opt.nChan;
    Coord cellSize = opt.cellSize;
    const int gSize = opt.gSize;
    const int baseline = opt.baseline; 

    cout << "nSamples = " << nSamples <<endl;
    // Initialize the data to be gridded
    std::vector<Coord> u(nSamples);
    std::vector<Coord> v(nSamples);
    std::vector<Coord> w(nSamples);
    std::vector<Value> data(nSamples*nChan);
    std::vector<Value> cpuoutdata(nSamples*nChan);
    std::vector<Value> ompoutdata(nSamples*nChan);

    const unsigned int maxint = std::numeric_limits<int>::max();

    for (int i = 0; i < nSamples; i++) {
        u[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;
        v[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;
        w[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;

        for (int chan = 0; chan < nChan; chan++) {
            data[i*nChan+chan] = 1.0;
            cpuoutdata[i*nChan+chan] = 0.0;
            ompoutdata[i*nChan+chan] = 0.0;
        }
    }

    std::vector<Value> grid(gSize*gSize);
    grid.assign(grid.size(), Value(0.0));

    // Measure frequency in inverse wavelengths
    std::vector<Coord> freq(nChan);

    for (int i = 0; i < nChan; i++) {
        freq[i] = (1.4e9 - 2.0e5 * Coord(i) / Coord(nChan)) / 2.998e8;
    }

    // Initialize convolution function and offsets
    std::vector<Value> C;
    int support, overSample;
    std::vector<int> cOffset;
    // Vectors of grid centers
    std::vector<int> iu;
    std::vector<int> iv;
    Coord wCellSize;

    initC(freq, cellSize, baseline, wSize, support, overSample, wCellSize, C);
    initCOffset(u, v, w, freq, cellSize, wCellSize, wSize, gSize, support,
                overSample, cOffset, iu, iv);
    const int sSize = 2 * support + 1;

    const double griddings = (double(nSamples * nChan) * double((sSize) * (sSize)));


    ///////////////////////////////////////////////////////////////////////////
    // DO GRIDDING
    ///////////////////////////////////////////////////////////////////////////
    std::vector<Value> cpugrid(gSize*gSize);
    cpugrid.assign(cpugrid.size(), Value(0.0));
    {
        // Now we can do the timing for the CPU implementation
        cout << "+++++ Forward processing (CPU) +++++" << endl;

        Stopwatch sw;
        sw.start();
        gridKernel(data, support, C, cOffset, iu, iv, cpugrid, gSize);
        double time = sw.stop();
        report_timings(time, opt, sSize, griddings);

        cout << "Done" << endl;
    }

    std::vector<Value> ompgrid(gSize*gSize);
    ompgrid.assign(ompgrid.size(), Value(0.0));
    {
        // Now we can do the timing for the GPU implementation
        cout << "+++++ Forward processing (OpenMP) +++++" << endl;

        // Time is measured inside this function call, unlike the CPU versions
        Stopwatch sw;
        sw.start();
        const int nthreads = gridKernelOMP(data, support, C, cOffset, iu, iv, ompgrid, gSize);
        const double time = sw.stop();
        cout<<" Running with "<< nthreads <<" threads "<<endl;
        report_timings(time, opt, sSize, griddings);

        cout << "Done" << endl;
    }
    verify_result(" Forward Processing ", cpugrid, ompgrid);

    ///////////////////////////////////////////////////////////////////////////
    // DO DEGRIDDING
    ///////////////////////////////////////////////////////////////////////////
    {
        cpugrid.assign(cpugrid.size(), Value(1.0));
        // Now we can do the timing for the CPU implementation
        cout << "+++++ Reverse processing (CPU) +++++" << endl;

        Stopwatch sw;
        sw.start();
        degridKernel(cpugrid, gSize, support, C, cOffset, iu, iv, cpuoutdata);
        const double time = sw.stop();
        report_timings(time, opt, sSize, griddings);

        cout << "Done" << endl;
    }
    {
        ompgrid.assign(ompgrid.size(), Value(1.0));
        // Now we can do the timing for the GPU implementation
        cout << "+++++ Reverse processing (OpenMP) +++++" << endl;

        // Time is measured inside this function call, unlike the CPU versions
        Stopwatch sw;
        sw.start();
        const int nthreads = degridKernelOMP(ompgrid, gSize, support, C, cOffset, iu, iv, ompoutdata);
        const double time = sw.stop();
        cout<<" Running with "<< nthreads <<" threads "<<endl;
        report_timings(time, opt, sSize, griddings);

        cout << "Done" << endl;
    }
    verify_result(" Reverse Processing ", cpuoutdata, ompoutdata);
    return 0;
}
