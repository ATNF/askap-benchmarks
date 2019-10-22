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

// Include own header file first
#include "Benchmark.h"

// System includes
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

// BLAS includes
#ifdef USEBLAS

#define CAXPY cblas_caxpy
#define CDOTU_SUB cblas_cdotu_sub

#include <mkl_cblas.h>

#endif

Benchmark::Benchmark()
        : next(1)
{
}

// Return a pseudo-random integer in the rand 0..2147483647
// Based on an algorithm in Kernighan & Ritchie, "The C Programming Language"
int Benchmark::randomInt()
{
    const unsigned int maxint = std::numeric_limits<int>::max();
    next = next * 1103515245 + 12345;
    return ((unsigned int)(next / 65536) % maxint);
}

void Benchmark::init()
{
    // Initialize the data to be gridded
    u.resize(nSamples);
    v.resize(nSamples);
    w.resize(nSamples);
    samples.resize(nSamples*nChan);
    outdata.resize(nSamples*nChan);

    const unsigned int maxint = std::numeric_limits<int>::max();

    for (int i = 0; i < nSamples; i++) {
        u[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;
        v[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;
        w[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;

        for (int chan = 0; chan < nChan; chan++) {
            samples[i*nChan+chan].data = 1.0;
            outdata[i*nChan+chan] = 0.0;
        }
    }

    grid.resize(gSize*gSize);
    grid.assign(grid.size(), Value(0.0));

    // Measure frequency in inverse wavelengths
    std::vector<Coord> freq(nChan);

    for (int i = 0; i < nChan; i++) {
        freq[i] = (1.4e9 - 2.0e5 * Coord(i) / Coord(nChan)) / 2.998e8;
    }

    // Initialize convolution function and offsets
    initC(freq, cellSize, baseline, wSize, m_support, overSample, wCellSize, C);
    initCOffset(u, v, w, freq, cellSize, wCellSize, wSize, gSize, m_support,
                overSample);
}

void Benchmark::runGrid()
{
    gridKernel(m_support, C, grid, gSize);
}

void Benchmark::runDegrid()
{
    degridKernel(grid, gSize, m_support, C, outdata);
}

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
void Benchmark::gridKernel(const int support,
                           const std::vector<Value>& C,
                           std::vector<Value>& grid, const int gSize)
{
    const int sSize = 2 * support + 1;

    for (int dind = 0; dind < int(samples.size()); ++dind) {
        // The actual grid point from which we offset
        int gind = samples[dind].iu + gSize * samples[dind].iv - support;

        // The Convoluton function point from which we offset
        int cind = samples[dind].cOffset;

        for (int suppv = 0; suppv < sSize; suppv++) {
#ifdef USEBLAS
            CAXPY(sSize, &(samples[dind].data), &C[cind], 1, &grid[gind], 1);
#else
            Value* gptr = &grid[gind];
            const Value* cptr = &C[cind];
            const Value d = samples[dind].data;

            for (int suppu = 0; suppu < sSize; suppu++) {
                *(gptr++) += d * (*(cptr++));
            }

#endif
            gind += gSize;
            cind += sSize;
        }
    }
}

// Perform degridding
void Benchmark::degridKernel(const std::vector<Value>& grid, const int gSize, const int support,
                             const std::vector<Value>& C,
                             std::vector<Value>& data)
{
    const int sSize = 2 * support + 1;

    for (int dind = 0; dind < int(data.size()); ++dind) {

        data[dind] = 0.0;

        // The actual grid point from which we offset
        int gind = samples[dind].iu + gSize * samples[dind].iv - support;

        // The Convoluton function point from which we offset
        int cind = samples[dind].cOffset;

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

/////////////////////////////////////////////////////////////////////////////////
// Initialize W project convolution function
// - This is application specific and should not need any changes.
//
// freq - temporal frequency (inverse wavelengths)
// cellSize - size of one grid cell in wavelengths
// support - Total width of convolution function=2*support+1
// wCellSize - size of one w grid cell in wavelengths
// wSize - Size of lookup table in w
void Benchmark::initC(const std::vector<Coord>& freq,
                      const Coord cellSize, const Coord baseline,
                      const int wSize, int& support, int& overSample,
                      Coord& wCellSize, std::vector<Value>& C)
{
    std::cout << "Initializing W projection convolution function" << std::endl;
    support = static_cast<int>(1.5 * sqrt(std::abs(baseline) * static_cast<Coord>(cellSize)
                                          * freq[0]) / cellSize);
    overSample = 8;
    std::cout << "Support = " << support << " pixels" << std::endl;
    wCellSize = 2 * baseline * freq[0] / wSize;
    std::cout << "W cellsize = " << wCellSize << " wavelengths" << std::endl;

    // Convolution function. This should be the convolution of the
    // w projection kernel (the Fresnel term) with the convolution
    // function used in the standard case. The latter is needed to
    // suppress aliasing. In practice, we calculate entire function
    // by Fourier transformation. Here we take an approximation that
    // is good enough.
    const int sSize = 2 * support + 1;

    const int cCenter = (sSize - 1) / 2;

    C.resize(sSize*sSize*overSample*overSample*wSize);
    std::cout << "Size of convolution function = " << sSize*sSize*overSample
              *overSample*wSize*sizeof(Value) / (1024*1024) << " MB" << std::endl;
    std::cout << "Shape of convolution function = [" << sSize << ", " << sSize << ", "
                  << overSample << ", " << overSample << ", " << wSize << "]" << std::endl;

    for (int k = 0; k < wSize; k++) {
        double w = double(k - wSize / 2);
        double fScale = sqrt(std::abs(w) * wCellSize * freq[0]) / cellSize;

        for (int osj = 0; osj < overSample; osj++) {
            for (int osi = 0; osi < overSample; osi++) {
                for (int j = 0; j < sSize; j++) {
                    double j2 = std::pow((double(j - cCenter) + double(osj) / double(overSample)), 2);

                    for (int i = 0; i < sSize; i++) {
                        double r2 = j2 + std::pow((double(i - cCenter) + double(osi) / double(overSample)), 2);
                        long int cind = i + sSize * (j + sSize * (osi + overSample * (osj + overSample * k)));

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
        sumC += std::abs(C[i]);
    }

    for (int i = 0; i < sSize*sSize*overSample*overSample*wSize; i++) {
        C[i] *= Value(wSize * overSample * overSample / sumC);
    }
}

// Initialize Lookup function
// - This is application specific and should not need any changes.
//
// nSamples - number of visibility samples
// freq - temporal frequency (inverse wavelengths)
// cellSize - size of one grid cell in wavelengths
// gSize - size of grid in pixels (per axis)
// support - Total width of convolution function=2*support+1
// wCellSize - size of one w grid cell in wavelengths
// wSize - Size of lookup table in w
void Benchmark::initCOffset(const std::vector<Coord>& u, const std::vector<Coord>& v,
                            const std::vector<Coord>& w, const std::vector<Coord>& freq,
                            const Coord cellSize, const Coord wCellSize,
                            const int wSize, const int gSize, const int support, const int overSample)
{
    const int nSamples = u.size();
    const int nChan = freq.size();

    const int sSize = 2 * support + 1;

    // Now calculate the offset for each visibility point
    for (int i = 0; i < nSamples; i++) {
        for (int chan = 0; chan < nChan; chan++) {

            int dind = i * nChan + chan;

            Coord uScaled = freq[chan] * u[i] / cellSize;
            samples[dind].iu = int(uScaled);

            if (uScaled < Coord(samples[dind].iu)) {
                samples[dind].iu -= 1;
            }

            int fracu = int(overSample * (uScaled - Coord(samples[dind].iu)));
            samples[dind].iu += gSize / 2;

            Coord vScaled = freq[chan] * v[i] / cellSize;
            samples[dind].iv = int(vScaled);

            if (vScaled < Coord(samples[dind].iv)) {
                samples[dind].iv -= 1;
            }

            int fracv = int(overSample * (vScaled - Coord(samples[dind].iv)));
            samples[dind].iv += gSize / 2;

            // The beginning of the convolution function for this point
            Coord wScaled = freq[chan] * w[i] / wCellSize;
            int woff = wSize / 2 + int(wScaled);
            samples[dind].cOffset = sSize * sSize * (fracu + overSample * (fracv + overSample * woff));
        }
    }
}

int Benchmark::getSupport()
{
    return m_support;
};

