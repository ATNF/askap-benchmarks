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
/// @author Ben Humphreys   <ben.humphreys@csiro.au>
/// @author Tim Cornwell    <tim.cornwell@csiro.au>
/// @author Daniel Mitchell <daniel.mitchell@csiro.au>

// System includes
#include <iostream>
#include <cmath>
#include <ctime>
#include <complex>
#include <vector>
#include <algorithm>
#include <limits>
#include <cassert>

// OpenACC includes
#include <openacc.h>

// CUDA includes
#ifdef GPU
#include <cufft.h>
#endif

// Local includes
#include "Stopwatch.h"

using std::cout;
using std::endl;
using std::abs;

// Typedefs for easy testing
// Cost of using double for Coord is low, cost for
// double for Real is also low
typedef double Coord;
typedef float Real;
typedef std::complex<Real> Value;

void degridKernelReductionReal(const std::vector<Value>& grid, const int gSize, const int support,
                     const std::vector<Value>& C, const std::vector<int>& cOffset,
                     const std::vector<int>& iu, const std::vector<int>& iv,
                     std::vector<Value>& data)
{
    const int sSize = 2 * support + 1;

    const int d_size = data.size();
    Value *d_data = data.data();
    const Value *d_grid = grid.data();
    const Value *d_C = C.data();
    const int *d_cOffset = cOffset.data();
    const int *d_iu = iu.data();
    const int *d_iv = iv.data();

    int dind;
    #pragma acc parallel loop
    for (dind = 0; dind < d_size; ++dind) {

        // The actual grid point from which we offset
        int gind = d_iu[dind] + gSize * d_iv[dind] - support;
        // The Convoluton function point from which we offset
        int cind = d_cOffset[dind];

        float re = 0.0, im = 0.0;
        #pragma acc loop reduction(+:re,im) collapse(2)
        for (int suppv = 0; suppv < sSize; suppv++) {
            for (int suppu = 0; suppu < sSize; suppu++) {
                re = re + d_grid[gind+suppv*gSize+suppu].real() * d_C[cind+suppv*sSize+suppu].real() -
                          d_grid[gind+suppv*gSize+suppu].imag() * d_C[cind+suppv*sSize+suppu].imag();
                im = im + d_grid[gind+suppv*gSize+suppu].imag() * d_C[cind+suppv*sSize+suppu].real() +
                          d_grid[gind+suppv*gSize+suppu].real() * d_C[cind+suppv*sSize+suppu].imag();
            }
        }
        d_data[dind] = Value(re,im);

    }

}

void degridKernelReductionComplex(const std::vector<Value>& grid, const int gSize, const int support,
                     const std::vector<Value>& C, const std::vector<int>& cOffset,
                     const std::vector<int>& iu, const std::vector<int>& iv,
                     std::vector<Value>& data)
{
    const int sSize = 2 * support + 1;

    const int d_size = data.size();
    Value *d_data = data.data();
    const Value *d_grid = grid.data();
    const Value *d_C = C.data();
    const int *d_cOffset = cOffset.data();
    const int *d_iu = iu.data();
    const int *d_iv = iv.data();

    int dind;
    #pragma acc parallel loop
    for (dind = 0; dind < d_size; ++dind) {

        // The actual grid point from which we offset
        int gind = d_iu[dind] + gSize * d_iv[dind] - support;
        // The Convoluton function point from which we offset
        int cind = d_cOffset[dind];

        float re = 0.0, im = 0.0;
        #pragma acc loop reduction(+:re,im) collapse(2)
        for (int suppv = 0; suppv < sSize; suppv++) {
            for (int suppu = 0; suppu < sSize; suppu++) {
                const Value cval = d_grid[gind+suppv*gSize+suppu] * d_C[cind+suppv*sSize+suppu];
                re = re + cval.real();
                im = im + cval.imag();
            }
        }
        d_data[dind] = Value(re,im);

    }

}

void degridKernelDataLoopReal(const std::vector<Value>& grid, const int gSize, const int support,
                     const std::vector<Value>& C, const std::vector<int>& cOffset,
                     const std::vector<int>& iu, const std::vector<int>& iv,
                     std::vector<Value>& data)
{
    const int sSize = 2 * support + 1;

    const int d_size = data.size();
    Value *d_data = data.data();
    const Value *d_grid = grid.data();
    const Value *d_C = C.data();
    const int *d_cOffset = cOffset.data();
    const int *d_iu = iu.data();
    const int *d_iv = iv.data();

    int dind;
    #pragma acc parallel loop gang vector
    for (dind = 0; dind < d_size; ++dind) {

        // The actual grid point from which we offset
        int gind = d_iu[dind] + gSize * d_iv[dind] - support;
        // The Convoluton function point from which we offset
        int cind = d_cOffset[dind];

        float re = 0.0, im = 0.0;
        for (int suppv = 0; suppv < sSize; suppv++) {
            for (int suppu = 0; suppu < sSize; suppu++) {
                re = re + d_grid[gind+suppv*gSize+suppu].real() * d_C[cind+suppv*sSize+suppu].real() -
                          d_grid[gind+suppv*gSize+suppu].imag() * d_C[cind+suppv*sSize+suppu].imag();
                im = im + d_grid[gind+suppv*gSize+suppu].imag() * d_C[cind+suppv*sSize+suppu].real() +
                          d_grid[gind+suppv*gSize+suppu].real() * d_C[cind+suppv*sSize+suppu].imag();
            }
        }
        d_data[dind] = Value(re,im);

    }

}

void degridKernelDataLoopComplex(const std::vector<Value>& grid, const int gSize, const int support,
                     const std::vector<Value>& C, const std::vector<int>& cOffset,
                     const std::vector<int>& iu, const std::vector<int>& iv,
                     std::vector<Value>& data)
{
    const int sSize = 2 * support + 1;

    const int d_size = data.size();
    Value *d_data = data.data();
    const Value *d_grid = grid.data();
    const Value *d_C = C.data();
    const int *d_cOffset = cOffset.data();
    const int *d_iu = iu.data();
    const int *d_iv = iv.data();

    int dind;
    #pragma acc parallel loop gang vector
    for (dind = 0; dind < d_size; ++dind) {

        // The actual grid point from which we offset
        int gind = d_iu[dind] + gSize * d_iv[dind] - support;
        // The Convoluton function point from which we offset
        int cind = d_cOffset[dind];

        Value cmplx = 0.0;

        for (int suppv = 0; suppv < sSize; suppv++) {
            for (int suppu = 0; suppu < sSize; suppu++) {
                cmplx = cmplx + d_grid[gind+suppv*gSize+suppu] * d_C[cind+suppv*sSize+suppu];
            }
        }
        d_data[dind] = cmplx;

    }

}

/////////////////////////////////////////////////////////////////////////////////
// Initialize W project convolution function
// - This is application specific and should not need any changes.
//
// freq - temporal frequency (inverse wavelengths)
// cellSize - size of one grid cell in wavelengths
// support - Total width of convolution function=2*support+1
// wCellSize - size of one w grid cell in wavelengths
// wSize - Size of lookup table in w
void initC(const std::vector<Coord>& freq, const Coord cellSize,
           const Coord baseline,
           const int wSize, int& support, int& overSample,
           Coord& wCellSize, std::vector<Value>& C)
{
    cout << "Initializing W projection convolution function" << endl;
    // changed 1.5x to 0.9x to reduce kernel memory below 48 KB.
    support = static_cast<int>(1.5 * sqrt(std::abs(baseline) * static_cast<Coord>(cellSize)
                                          * freq[0]) / cellSize);
    overSample = 8;
    cout << "Support = " << support << " pixels" << endl;
    wCellSize = 2 * baseline * freq[0] / wSize;
    cout << "W cellsize = " << wCellSize << " wavelengths" << endl;

    // Convolution function. This should be the convolution of the
    // w projection kernel (the Fresnel term) with the convolution
    // function used in the standard case. The latter is needed to
    // suppress aliasing. In practice, we calculate entire function
    // by Fourier transformation. Here we take an approximation that
    // is good enough.
    const int sSize = 2 * support + 1;

    const int cCenter = (sSize - 1) / 2;

    C.resize(sSize*sSize*overSample*overSample*wSize);
    cout << "Size of convolution function = " << sSize*sSize*overSample
         *overSample*wSize*sizeof(Value) / (1024*1024) << " MB" << std::endl;
    cout << "Shape of convolution function = [" << sSize << ", " << sSize << ", "
             << overSample << ", " << overSample << ", " << wSize << "]" << std::endl;

    for (int k = 0; k < wSize; k++) {
        double w = double(k - wSize / 2);
        double fScale = sqrt(abs(w) * wCellSize * freq[0]) / cellSize;

        for (int osj = 0; osj < overSample; osj++) {
            for (int osi = 0; osi < overSample; osi++) {
                for (int j = 0; j < sSize; j++) {
                    const double j2 = std::pow((double(j - cCenter) + double(osj) / double(overSample)), 2);

                    for (int i = 0; i < sSize; i++) {
                        const double r2 = j2 + std::pow((double(i - cCenter) + double(osi) / double(overSample)), 2);
                        const int cind = i + sSize * (j + sSize * (osi + overSample * (osj + overSample * k)));

                        if (w != 0.0) {
                            C[cind] = static_cast<Value>(std::cos(r2 / (w * fScale)));
                        } else {
                            C[cind] = static_cast<Value>(std::exp(-r2));
                        }
                    }
                }
            }
        }
    }

    // Now normalise the convolution function
    Real sumC = 0.0;

    for (int i = 0; i < sSize*sSize*overSample*overSample*wSize; i++) {
        sumC += abs(C[i]);
    }

    for (int i = 0; i < sSize*sSize*overSample*overSample*wSize; i++) {
        C[i] *= Value(wSize * overSample * overSample / sumC);
    }
}

// Initialize Lookup function
// - This is application specific and should not need any changes.
//
// freq - temporal frequency (inverse wavelengths)
// cellSize - size of one grid cell in wavelengths
// gSize - size of grid in pixels (per axis)
// support - Total width of convolution function=2*support+1
// wCellSize - size of one w grid cell in wavelengths
// wSize - Size of lookup table in w
void initCOffset(const std::vector<Coord>& u, const std::vector<Coord>& v,
                 const std::vector<Coord>& w, const std::vector<Coord>& freq,
                 const Coord cellSize, const Coord wCellSize,
                 const int wSize, const int gSize, const int support, const int overSample,
                 std::vector<int>& cOffset, std::vector<int>& iu,
                 std::vector<int>& iv)
{
    const int nSamples = u.size();
    const int nChan = freq.size();

    const int sSize = 2 * support + 1;

    // Now calculate the offset for each visibility point
    cOffset.resize(nSamples*nChan);
    iu.resize(nSamples*nChan);
    iv.resize(nSamples*nChan);

    for (int i = 0; i < nSamples; i++) {
        for (int chan = 0; chan < nChan; chan++) {

            const int dind = i * nChan + chan;

            const Coord uScaled = freq[chan] * u[i] / cellSize;
            iu[dind] = int(uScaled);

            if (uScaled < Coord(iu[dind])) {
                iu[dind] -= 1;
            }

            const int fracu = int(overSample * (uScaled - Coord(iu[dind])));
            iu[dind] += gSize / 2;

            const Coord vScaled = freq[chan] * v[i] / cellSize;
            iv[dind] = int(vScaled);

            if (vScaled < Coord(iv[dind])) {
                iv[dind] -= 1;
            }

            const int fracv = int(overSample * (vScaled - Coord(iv[dind])));
            iv[dind] += gSize / 2;

            // The beginning of the convolution function for this point
            Coord wScaled = freq[chan] * w[i] / wCellSize;
            int woff = wSize / 2 + int(wScaled);
            cOffset[dind] = sSize * sSize * (fracu + overSample * (fracv + overSample * woff));
        }
    }

}

// Return a pseudo-random integer in the range 0..2147483647
// Based on an algorithm in Kernighan & Ritchie, "The C Programming Language"
static unsigned long next = 1;
int randomInt()
{
    const unsigned int maxint = std::numeric_limits<int>::max();
    next = next * 1103515245 + 12345;
    return ((unsigned int)(next / 65536) % maxint);
}

// Main testing routine
int main()
{
    // Change these if necessary to adjust run time
    const int nSamples = 160000; // Number of data samples
//const int nSamples = 1600; // Number of data samples
    const int wSize = 33; // Number of lookup planes in w projection
    const int nChan = 1; // Number of spectral channels

    // Don't change any of these numbers unless you know what you are doing!
    const int gSize = 4096; // Size of output grid in pixels
    const Coord cellSize = 5.0; // Cellsize of output grid in wavelengths
    const int baseline = 2000; // Maximum baseline in meters

    // Initialize the data to be gridded
    std::vector<Coord> u(nSamples);
    std::vector<Coord> v(nSamples);
    std::vector<Coord> w(nSamples);
    std::vector<Value> data(nSamples*nChan);
    std::vector<Value> outdata0(nSamples*nChan);
    std::vector<Value> outdata1(nSamples*nChan);

    const unsigned int maxint = std::numeric_limits<int>::max();

    for (int i = 0; i < nSamples; i++) {
        u[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;
        v[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;
        w[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;

        for (int chan = 0; chan < nChan; chan++) {
            data[i*nChan+chan] = 1.0;
            outdata0[i*nChan+chan] = 0.0;
            outdata1[i*nChan+chan] = 0.0;
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

    double timeReal, time;

    std::vector<Value> cpugrid(gSize*gSize);

    ///////////////////////////////////////////////////////////////////////////
    // DO DEGRIDDING
    ///////////////////////////////////////////////////////////////////////////
    {
        cpugrid.assign(cpugrid.size(), Value(1.0));
        // Now we can do the timing for the CPU implementation
        cout << "+++++ Reverse processing (CPU with real mult & reduction) +++++" << endl;

        Stopwatch sw;
        sw.start();
        degridKernelReductionReal(cpugrid, gSize, support, C, cOffset, iu, iv, outdata0);
        timeReal = sw.stop();

        // Report on timings
        cout << "    Time " << timeReal << " (s) " << endl;
        cout << "    Time per visibility spectral sample " << 1e6*timeReal / double(data.size()) << " (us) " << endl;
        cout << "    Time per degridding   " << 1e9*timeReal / (double(data.size())* double((sSize)*(sSize))) << " (ns) " << endl;
        cout << "    Degridding rate   " << (griddings / 1000000) / timeReal << " (million grid points per second)" << endl;

        cout << "Done" << endl;
    }

    {
        cpugrid.assign(cpugrid.size(), Value(1.0));
        // Now we can do the timing for the GPU implementation
        cout << "+++++ Reverse processing (OpenACC with complex mult & reduction) +++++" << endl;

        // Time is measured inside this function call, unlike the CPU versions
        Stopwatch sw;
        sw.start();
        degridKernelReductionComplex(cpugrid, gSize, support, C, cOffset, iu, iv, outdata1);
        time = sw.stop();

        // Report on timings
        cout << "    Time " << time << " (s) = real mult / " << timeReal/time << endl;
        cout << "    Time per visibility spectral sample " << 1e6*time / double(data.size()) << " (us) " << endl;
        cout << "    Time per degridding   " << 1e9*time / (double(data.size())* double((sSize)*(sSize))) << " (ns) " << endl;
        cout << "    Degridding rate   " << (griddings / 1000000) / time << " (million grid points per second)" << endl;

        cout << "Done" << endl;
    }

    // Verify degridding results
    cout << "Verifying result...";

    if (outdata0.size() != outdata1.size()) {
        cout << "Fail (Data vector sizes differ)" << std::endl;
        return 1;
    }

    for (unsigned int i = 0; i < outdata0.size(); ++i) {
        if (fabs(outdata0[i].real() - outdata1[i].real()) > 0.00001) {
            cout << "Fail (Expected " << outdata0[i].real() << " got "
                     << outdata1[i].real() << " at index " << i << ")"
                     << std::endl;
            return 1;
        }
    }

    cout << "Pass" << std::endl;

    {
        cpugrid.assign(cpugrid.size(), Value(1.0));
        // Now we can do the timing for the GPU implementation
        cout << "+++++ Reverse processing (OpenACC with real data loop) +++++" << endl;

        // Time is measured inside this function call, unlike the CPU versions
        Stopwatch sw;
        sw.start();
        degridKernelDataLoopReal(cpugrid, gSize, support, C, cOffset, iu, iv, outdata1);
        timeReal = sw.stop();

        // Report on timings
        cout << "    Time " << timeReal << " (s)" << endl;
        cout << "    Time per visibility spectral sample " << 1e6*timeReal / double(data.size()) << " (us) " << endl;
        cout << "    Time per degridding   " << 1e9*timeReal / (double(data.size())* double((sSize)*(sSize))) << " (ns) " << endl;
        cout << "    Degridding rate   " << (griddings / 1000000) / timeReal << " (million grid points per second)" << endl;

        cout << "Done" << endl;
    }

    // Verify degridding results
    cout << "Verifying result...";

    if (outdata0.size() != outdata1.size()) {
        cout << "Fail (Data vector sizes differ)" << std::endl;
        return 1;
    }

    for (unsigned int i = 0; i < outdata0.size(); ++i) {
        if (fabs(outdata0[i].real() - outdata1[i].real()) > 0.00001) {
            cout << "Fail (Expected " << outdata0[i].real() << " got "
                     << outdata1[i].real() << " at index " << i << ")"
                     << std::endl;
            return 1;
        }
    }

    cout << "Pass" << std::endl;

    {
        cpugrid.assign(cpugrid.size(), Value(1.0));
        // Now we can do the timing for the GPU implementation
        cout << "+++++ Reverse processing (OpenACC with complex data loop) +++++" << endl;

        // Time is measured inside this function call, unlike the CPU versions
        Stopwatch sw;
        sw.start();
        degridKernelDataLoopComplex(cpugrid, gSize, support, C, cOffset, iu, iv, outdata1);
        time = sw.stop();

        // Report on timings
        cout << "    Time " << time << " (s) = real mult / " << timeReal/time << endl;
        cout << "    Time per visibility spectral sample " << 1e6*time / double(data.size()) << " (us) " << endl;
        cout << "    Time per degridding   " << 1e9*time / (double(data.size())* double((sSize)*(sSize))) << " (ns) " << endl;
        cout << "    Degridding rate   " << (griddings / 1000000) / time << " (million grid points per second)" << endl;

        cout << "Done" << endl;
    }

    // Verify degridding results
    cout << "Verifying result...";

    if (outdata0.size() != outdata1.size()) {
        cout << "Fail (Data vector sizes differ)" << std::endl;
        return 1;
    }

    for (unsigned int i = 0; i < outdata0.size(); ++i) {
        if (fabs(outdata0[i].real() - outdata1[i].real()) > 0.00001) {
            cout << "Fail (Expected " << outdata0[i].real() << " got "
                     << outdata1[i].real() << " at index " << i << ")"
                     << std::endl;
            return 1;
        }
    }

    cout << "Pass" << std::endl;

    return 0;
}
