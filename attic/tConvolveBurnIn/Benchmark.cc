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
#include <mpi.h>

// BLAS includes
#ifdef USEBLAS

#define CAXPY cblas_caxpy
#define CDOTU_SUB cblas_cdotu_sub

#include <mkl_cblas.h>

#endif

// Local includes
#include "Stopwatch.h"

Benchmark::Benchmark(int rank, int numtasks)
        : m_rank(rank), m_numtasks(numtasks), m_next(1)
{
}

// Return a pseudo-random integer in the range 0..2147483647
// Based on an algorithm in Kernighan & Ritchie, "The C Programming Language"
int Benchmark::randomInt()
{
    const unsigned int maxint = std::numeric_limits<int>::max();
    m_next = m_next * 1103515245 + 12345;
    return ((unsigned int)(m_next / 65536) % maxint);
}

void Benchmark::init()
{
    // Initialize the data to be gridded
    u.resize(nSamples);
    v.resize(nSamples);
    w.resize(nSamples);
    m_samples.resize(nSamples*nChan);
    m_outdata.resize(nSamples*nChan);

    const unsigned int maxint = std::numeric_limits<int>::max();

    for (int i = 0; i < nSamples; i++) {
        u[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;
        v[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;
        w[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;

        for (int chan = 0; chan < nChan; chan++) {
            m_samples[i*nChan+chan].data = 1.0;
            m_outdata[i*nChan+chan] = 0.0;
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
    initC(freq, cellSize, wSize, m_support, m_overSample, m_wCellSize, C);
    initCOffset(u, v, w, freq, cellSize, m_wCellSize, wSize, gSize,
                m_support, m_overSample);
}

bool Benchmark::runGrid(double& time)
{
    Stopwatch sw;
    sw.start();
    gridKernel(m_support, C, grid, gSize);
    time = sw.stop();
    return shareAndCompare(grid);
}

bool Benchmark::runDegrid(double& time)
{
    Stopwatch sw;
    sw.start();
    degridKernel(grid, gSize, m_support, C, m_outdata);
    time = sw.stop();
    return shareAndCompare(m_outdata);
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

    for (int dind = 0; dind < int(m_samples.size()); ++dind) {
        // The actual grid point from which we offset
        int gind = m_samples[dind].iu + gSize * m_samples[dind].iv - support;

        // The Convoluton function point from which we offset
        int cind = m_samples[dind].cOffset;

        for (int suppv = 0; suppv < sSize; suppv++) {
#ifdef USEBLAS
            CAXPY(sSize, &(m_samples[dind].data), &C[cind], 1, &grid[gind], 1);
#else
            Value* gptr = &grid[gind];
            const Value* cptr = &C[cind];
            const Value d = m_samples[dind].data;

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
void Benchmark::degridKernel(const std::vector<Value>& grid,
                             const int gSize, const int support,
                             const std::vector<Value>& C,
                             std::vector<Value>& data)
{
    const int sSize = 2 * support + 1;

    for (int dind = 0; dind < int(data.size()); ++dind) {

        data[dind] = 0.0;

        // The actual grid point from which we offset
        int gind = m_samples[dind].iu + gSize * m_samples[dind].iv - support;

        // The Convoluton function point from which we offset
        int cind = m_samples[dind].cOffset;

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
// wSize - Size of lookup table in w
// support - Total width of convolution function=2*support+1
// wCellSize - size of one w grid cell in wavelengths
void Benchmark::initC(const std::vector<Coord>& freq,
                      const Coord cellSize, const int wSize,
                      int& support, int& overSample,
                      Coord& wCellSize, std::vector<Value>& C)
{
    if (m_rank == 0) {
        std::cout << "Initializing W projection convolution function" << std::endl;
    }
    support = static_cast<int>(1.5 * sqrt(std::abs(baseline) * static_cast<Coord>(cellSize)
                                          * freq[0]) / cellSize);

    wCellSize = 2 * baseline * freq[0] / wSize;

    // Convolution function. This should be the convolution of the
    // w projection kernel (the Fresnel term) with the convolution
    // function used in the standard case. The latter is needed to
    // suppress aliasing. In practice, we calculate entire function
    // by Fourier transformation. Here we take an approximation that
    // is good enough.
    const int sSize = 2 * support + 1;

    const int cCenter = (sSize - 1) / 2;

    overSample = 8;
    C.resize(sSize*sSize*overSample*overSample*wSize);

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

    if (m_rank == 0) {
        std::cout << "Support = " << support << " pixels" << std::endl;
        std::cout << "W cellsize = " << wCellSize << " wavelengths" << std::endl;
        std::cout << "Size of convolution function = " << sSize*sSize*overSample
            *overSample*wSize*sizeof(Value) / (1024*1024) << " MB" << std::endl;
        std::cout << "Shape of convolution function = [" << sSize << ", " << sSize << ", "
            << overSample << ", " << overSample << ", " << wSize << "]" << std::endl;
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
void Benchmark::initCOffset(const std::vector<Coord>& u, const std::vector<Coord>& v,
                            const std::vector<Coord>& w, const std::vector<Coord>& freq,
                            const Coord cellSize, const Coord wCellSize,
                            const int wSize, const int gSize, const int support,
                            const int overSample)
{
    const int nSamples = u.size();
    const int nChan = freq.size();

    const int sSize = 2 * support + 1;

    // Now calculate the offset for each visibility point
    for (int i = 0; i < nSamples; i++) {
        for (int chan = 0; chan < nChan; chan++) {

            int dind = i * nChan + chan;

            Coord uScaled = freq[chan] * u[i] / cellSize;
            m_samples[dind].iu = int(uScaled);

            if (uScaled < Coord(m_samples[dind].iu)) {
                m_samples[dind].iu -= 1;
            }

            int fracu = int(overSample * (uScaled - Coord(m_samples[dind].iu)));
            m_samples[dind].iu += gSize / 2;

            Coord vScaled = freq[chan] * v[i] / cellSize;
            m_samples[dind].iv = int(vScaled);

            if (vScaled < Coord(m_samples[dind].iv)) {
                m_samples[dind].iv -= 1;
            }

            int fracv = int(overSample * (vScaled - Coord(m_samples[dind].iv)));
            m_samples[dind].iv += gSize / 2;

            // The beginning of the convolution function for this point
            Coord wScaled = freq[chan] * w[i] / wCellSize;
            int woff = wSize / 2 + int(wScaled);
            m_samples[dind].cOffset = sSize * sSize * (fracu + overSample * (fracv + overSample * woff));
        }
    }
}

bool Benchmark::shareAndCompare(std::vector<Value>& data)
{
    const int dest = (m_rank + (m_numtasks / 2)) % m_numtasks;
    std::vector<Value> other(data.size());

    // Do async send/recv
    const int ioCount = 2;
    MPI_Request reqs[ioCount];
    MPI_Status status[ioCount];

    int error = MPI_Isend(&data[0], data.size() * sizeof(Value), MPI_BYTE, dest, 0, MPI_COMM_WORLD, &reqs[0]);
    checkError(error, "MPI_ISend");
    error = MPI_Irecv(&other[0], data.size() * sizeof(Value), MPI_BYTE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &reqs[1]);
    checkError(error, "MPI_Irecv");

    error = MPI_Waitall(ioCount, reqs, status);
    checkError(error, "MPI_Waitall");
    const int source = status[1].MPI_SOURCE;

    // Compare arrays
    const float fltEpsilon = std::numeric_limits<float>::epsilon();
    for (size_t i = 0; i < data.size(); ++i) {
        if (fabs(data[i].real() - other[i].real()) > fltEpsilon) {
            std::cout << "Error: Ranks " << m_rank << " and " << source <<
                " disagree. (Expected " << data[i].real() << " got "
                << other[i].real() << " at index " << i << ")" << std::endl;
            return false;
        }
    }

    return true;
}

void Benchmark::checkError(const int error, const std::string& location)
{
    if (error == MPI_SUCCESS) {
        return;
    }

    char estring[MPI_MAX_ERROR_STRING];
    int eclass;
    int len;

    MPI_Error_class(error, &eclass);
    MPI_Error_string(error, estring, &len);
    std::cout << "Error: " << location << " failed with " << eclass << ": "
        << estring << std::endl;
}
