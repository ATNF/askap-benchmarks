/// @copyright (c) 2007, 2019 CSIRO
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
/// @author Daneil Mitchell <daniel.mitchell@csiro.au>

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

// Return a pseudo-random integer in the range 0..2147483647
// Based on an algorithm in Kernighan & Ritchie, "The C Programming Language"
int Benchmark::randomInt()
{
    const unsigned int maxint = std::numeric_limits<int>::max();
    next = next * 1103515245 + 12345;
    return ((unsigned int)(next / 65536) % maxint);
}

void Benchmark::init()
{

    // Initialize constants
    const Coord obslen = 12.;               // Observation length in hours
    const Coord scanlen = 5.;               // Observation scan length in seconds
    const int nScans = obslen*3600./scanlen;

    nChan = 1;                              // Number of spectral channels
    //baseline = set this later;            // Maximum baseline in meters

    const int apertureDiam = 12.;           // diameter of aperture (dish or station) in meters
    const Real maxFreqHz = 1.0e9;           // maximum frequency in Hz
    const Real lambda = 2.998e8/maxFreqHz;  // minimum wavelength in meters

    Real imgOS;                             // synthesised beam oversampling factor
    Real imgExt;                            // image extension factor: 1~FWHM, 2~first null, 4~second null

    int wkernel = 0;                        // just a trigger to print more info when w-kernels are used
    Real wmax, fov;
    // for du and dw to have a similar effect on DR, arXiv:1207.586 gives dw ~ du * 2 / FoV
    // or equivalenty, both should have their natural resolution divided by the same oversampling factor
    // or equivalenty, both should have the same number of pixels in the cached gridding cube
    // du_os = du / os
    //       = 1/FoV / os
    // dw_os = 2*wmax / (Nwplanes = os * kernel width)
    //       = 2*wmax / (os * wmax*FoV**2)
    //       = 2/FoV / os / FoV
    //       = du_os * 2 / FoV

    if (runType == 0) {
        // grid with variable kernel sizes (continuum imaging)
        baseline = 6440.;
        wkernel = 1;
        imgOS = 4.0;
        imgExt = 3;
        overSample = 4;
        wmax = baseline/lambda;
        fov = lambda/apertureDiam * imgExt;
        const Real wPart = wmax*fov*fov;
        const Real aPart = 7.;
        m_support = int(ceil(sqrt( aPart*aPart + wPart*wPart )))/2;
        wSize = ceil(overSample * wPart);
        wSize += (wSize+1)%2; // make odd
        wCellSize = 2*wmax / (wSize-1);
    } else if (runType == 1) {
        // grid with variable kernel sizes (spectral line imaging)
        baseline = 2000.;
        wkernel = 1;
        imgOS = 3.1;
        imgExt = 2;
        overSample = 8;
        wmax = baseline/lambda;
        fov = lambda/apertureDiam * imgExt;
        const Real wPart = wmax*fov*fov;
        const Real aPart = 7.;
        m_support = int(ceil(sqrt( aPart*aPart + wPart*wPart )))/2;
        wSize = ceil(overSample * wPart);
        wSize += (wSize+1)%2; // make odd
        wCellSize = 2*wmax / (wSize-1);
    } else if (runType == 2) {
        // nearest-neighbour gridding
        baseline = 6440.;
        imgOS = 2.0;
        imgExt = 1.0;
        overSample = 1;
        m_support = 0;
        wSize = 1;
        wCellSize = 0.0;
    } else if (runType == 3) {
        // grid with small kernels (7x7)
        baseline = 6440.;
        imgOS = 4.0;
        imgExt = 1.923;
        overSample = 128;
        m_support = 3;
        wSize = 1;
        wCellSize = 0.0;
    } else if (runType == 4) {
        // grid with large kernels (87x87)
        baseline = 6440.;
        imgOS = 4.0;
        imgExt = 1.923;
        overSample = 8;
        m_support = 43;
        wSize = 1;
        wCellSize = 0.0;
    } else {
        std::cout << "Unsupported imaging type" << std::endl;
        exit(1);
    }

    // Size of output grid in pixels
    gSize = ceil(baseline/apertureDiam * imgOS * imgExt);
    // Cellsize of output grid in wavelengths
    uvCellSize = baseline/lambda * imgOS / Real(gSize);

    if (mpirank == 0) {
        std::cout << "  Maximum frequency = " << maxFreqHz/1e6 << " MHz (lambda = "<<lambda<<" m)" << std::endl;
        std::cout << "  Grid size = " << gSize << " pixels ("<<1./uvCellSize*180/3.141593<<" deg)" <<
                     " uv cell size = " << uvCellSize << " wavelengths" << std::endl;
        if (wkernel) {
            std::cout << "  wmax: "<<wmax << " => 1/2 w theta^2 = " << m_support <<
                         ". num planes = os.w.theta^2 = "<< overSample*m_support << std::endl;
        }
    }

    const unsigned int maxint = std::numeric_limits<int>::max();

    // observation coordinates (26.6970° S, 116.6311° E)
    // set dec to obs lat and ha to +/- 6 hours
    Coord lat = -26.6970 * 3.141593/180.0;
    Coord dec = lat;
    const Coord cdec = cos(dec);
    const Coord sdec = sin(dec);

    const int nAntennas = 36;
    const int nBaselinesMax = (nAntennas*(nAntennas-1))/2;

    static const Coord east[]  = {  -42.43847222,   -15.46047222,    -6.48847222,   -51.41747222,
                                   -116.43047222,    93.22152778,   200.42152778,   -80.24847222,
                                   -286.64847222,  -138.75447222,   225.51252778,   353.48652778,
                                    396.28152778,   -67.29847222,  -782.13847222,  -678.55347222,
                                   -539.25647222,  -149.22347222,   175.37552778,   463.88152778,
                                    643.72952778,   803.40152778,   -43.14647222,    36.03352778,
                                   -656.05547222,  -435.77447222, -1112.94147222,   207.32652778,
                                    523.49152778,  1186.61752778,  2178.51052778,  2982.48652778,
                                    -17.41247222, -3017.44747222, -2213.43647222,   -19.20647222};
    static const Coord north[] = { -105.22933333,  -118.24033333,   -97.73933333,   -70.22133333,
                                    -73.72633333,     6.77066667,  -215.20933333,  -343.73933333,
                                      5.32066667,   174.26666667,   235.30966667,   164.24666667,
                                   -469.23433333,  -565.23833333,  -263.22233333,   260.21066667,
                                    417.28966667,   270.27966667,   376.29066667,   209.79766667,
                                    216.77666667,   230.75266667,  -762.23633333, -1083.75233333,
                                    548.27766667,   562.27366667,   835.74566667,  1093.28166667,
                                    647.98066667,   693.23866667,   887.75566667, -2612.27533333,
                                  -2916.19433333, -2112.20733333,   887.76166667,  3084.83866667};
    std::vector<Coord> E (east, east + sizeof(east) / sizeof(east[0]) );
    std::vector<Coord> N (north, north + sizeof(north) / sizeof(north[0]) );
    std::vector<Coord> X(nAntennas), Y(nAntennas), Z(nAntennas);
    std::vector<Coord> BX(nBaselinesMax), BY(nBaselinesMax), BZ(nBaselinesMax);

    for (int i = 0; i < nAntennas; i++) {
        X[i] = -N[i]*sin(lat);
        Y[i] =  E[i];
        Z[i] =  N[i]*cos(lat);
    }
    int bl = 0;
    for (int i = 0; i < nAntennas-1; i++) {
        for (int j = i+1; j < nAntennas; j++) {
            Coord dX = X[i] - X[j];
            Coord dY = Y[i] - Y[j];
            Coord dZ = Z[i] - Z[j];
            if (dX*dX + dY*dY + dZ*dZ > baseline*baseline) continue;
            BX[bl] = dX;
            BY[bl] = dY;
            BZ[bl] = dZ;
            bl++;
        }
    }

    const int nBaselines = bl;
    nSamples = nScans*nBaselines;           // Number of data samples per channel, polarisation & beam

    // Initialize the data to be gridded
    u.resize(nSamples);
    v.resize(nSamples);
    w.resize(nSamples);
    iu.resize(nSamples*nChan);
    iv.resize(nSamples*nChan);
    wPlane.resize(nSamples*nChan);
    cOffset.resize(nSamples*nChan);
    data.resize(nSamples*nChan);
    outdata1.resize(nSamples*nChan);
    outdata2.resize(nSamples*nChan);

    cOffset0.resize(wSize);
    sSize.resize(wSize);
    numPerPlane.resize(wSize);
    for (int woff = 0; woff < wSize; woff++) {
        numPerPlane[woff] = 0;
    }

    for (int i = 0; i < nSamples; i++) {
        const int bl = nBaselines * (Coord(randomInt()) / Coord(maxint));
        const Coord ha = obslen * 3.141593/12.0 * ((Coord(randomInt()) / Coord(maxint)) - 0.5);
        const Coord cha = cos(ha);
        const Coord sha = sin(ha);
        u[i] =       sha*BX[bl] +      cha*BY[bl];
        v[i] = -sdec*cha*BX[bl] + sdec*sha*BY[bl] + cdec*BZ[bl];
        w[i] =  cdec*cha*BX[bl] - cdec*sha*BY[bl] + sdec*BZ[bl];

        for (int chan = 0; chan < nChan; chan++) {
            data[i*nChan+chan] = 1.0;
            outdata1[i*nChan+chan] = 0.0;
            outdata2[i*nChan+chan] = 0.0;
        }
    }

    grid1.resize(gSize*gSize);
    grid1.assign(grid1.size(), Value(0.0));
    //grid2.resize(gSize*gSize);
    //grid2.assign(grid2.size(), Value(0.0));

    // Measurement frequency in inverse wavelengths
    std::vector<Coord> wavenumber(nChan);
    for (int i = 0; i < nChan; i++) {
        wavenumber[i] = (maxFreqHz - 2.0e5 * Coord(i) / Coord(nChan)) / 2.998e8;
    }

    // Initialize convolution function and offsets
    initC(uvCellSize, wSize, m_support, overSample, wCellSize, C);
    initCOffset(u, v, w, wavenumber, uvCellSize, wCellSize, wSize, gSize, overSample);

    if ( (doSort==1) && (wSize>1) ) {
        // sort based on w-plane but without consideration of order within
        //  - want threads to have equal kernel size
        //  - also align by uv offset?
        //  - also align by uv region?
        const std::vector<int> iu_tmp(iu);
        const std::vector<int> iv_tmp(iv);
        const std::vector<int> wPlane_tmp(wPlane);
        const std::vector<int> cOffset_tmp(cOffset);

        std::vector<int> sortedIndex(wSize,0);
        for (int woff = 1; woff < wSize; woff++) {
            sortedIndex[woff] = sortedIndex[woff-1] + numPerPlane[woff-1];
        }
        for (int i = 0; i < int(data.size()); i++) {
            const int j = sortedIndex[wPlane_tmp[i]];
            sortedIndex[wPlane_tmp[i]]++;
            iu[j] = iu_tmp[i];
            iv[j] = iv_tmp[i];
            wPlane[j] = wPlane_tmp[i];
            cOffset[j] = cOffset_tmp[i];
        }

    }

}

void Benchmark::runGrid()
{
    gridKernel(C, grid1, gSize);
}

void Benchmark::runDegrid()
{
    degridKernel(grid1, gSize, C, outdata1);
}

/*
void Benchmark::runGridCheck()
{
    double sum1 = 0.0;
    double sum2 = 0.0;
    for (int i = 0; i < int(grid1.size()); i++) {
        sum1 += abs(grid1[i]);
        sum2 += abs(grid2[i]);
        if (abs(grid1[i] - grid2[i])/abs(grid1[i]) > 1e-4) {
            std::cout << "    Check failed" << std::endl;
            std::cout << "     - grid["<<i<<"] = "<< grid1[i]<<" != "<<grid2[i] << std::endl;
            return;
        }
    }
    if ( sum1 > 0 && sum2 > 0 ) {
        std::cout << "    Check passed" << std::endl;
    } else {
        std::cout << "    Check failed" << std::endl;
        std::cout << "     - null test: sum of absolute pixel values = " << sum1 << ", " << sum2 << std::endl;
    }
}

void Benchmark::runDegridCheck()
{
    double sum1 = 0.0;
    double sum2 = 0.0;
    for (int i = 0; i < int(outdata1.size()); i++) {
        sum1 += abs(outdata1[i]);
        sum2 += abs(outdata2[i]);
        if (abs(outdata1[i] - outdata2[i])/abs(outdata1[i]) > 1e-4) {
            std::cout << "    Check failed" << std::endl;
            std::cout << "     - outdata["<<i<<"] = "<< outdata1[i]<<" != "<<outdata2[i] << std::endl;
            return;
        }
    }
    if ( sum1 > 0 && sum2 > 0 ) {
        std::cout << "    Check passed" << std::endl;
    } else {
        std::cout << "    Check failed" << std::endl;
        std::cout << "     - null test: sum of absolute data values = " << sum1 << ", " << sum2 << std::endl;
    }
}
*/

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
void Benchmark::gridKernel(const std::vector<Value>& C,
                           std::vector<Value>& grid,
                           const int gSize)
{
    for (int dind = 0; dind < int(data.size()); ++dind) {

        // Kernel info
        const int wind = wPlane[dind];
        const int support = sSize[wind]/2;

        // The actual grid point from which we offset
        int gind = iu[dind] + gSize * iv[dind] - support;

        // The Convoluton function point from which we offset
        int cind = cOffset[dind];

        const Real dre = data[dind].real();
        const Real dim = data[dind].imag();

        for (int suppv = 0; suppv < sSize[wind]; suppv++) {
#ifdef USEBLAS
            // replace the following with saxpy calls...
            CAXPY(sSize[wind], &data[dind], &C[cind], 1, &grid[gind], 1);
#else
            Value* gptr = &grid[gind];
            const Value* cptr = &C[cind];

            for (int suppu = 0; suppu < sSize[wind]; suppu++) {
                Real *gptr_re = (Real *)gptr;
                const Real *cptr_re = (Real *)cptr;
                gptr_re[0] += dre * cptr_re[0] - dim * cptr_re[1];
                gptr_re[1] += dim * cptr_re[0] + dre * cptr_re[1];
                gptr++;
                cptr++;
            }
#endif
            gind += gSize;
            cind += sSize[wind];
        }
    }
}

// Perform degridding
void Benchmark::degridKernel(const std::vector<Value>& grid,
                             const int gSize,
                             const std::vector<Value>& C,
                             std::vector<Value>& data)
{
    for (int dind = 0; dind < int(data.size()); ++dind) {

        // Kernel info
        const int wind = wPlane[dind];
        const int support = sSize[wind]/2;

        // The actual grid point from which we offset
        int gind = iu[dind] + gSize * iv[dind] - support;

        // The Convoluton function point from which we offset
        int cind = cOffset[dind];

        Real re = 0.0, im = 0.0;
        for (int suppv = 0; suppv < sSize[wind]; suppv++) {
#ifdef USEBLAS
            // replace the following with sdotu calls...
            Value dot;
            CDOTU_SUB(sSize[wind], &grid[gind], 1, &C[cind], 1, &dot);
            data[dind] += dot;
#else
            const Value* gptr = &grid[gind];
            const Value* cptr = &C[cind];

            for (int suppu = 0; suppu < sSize[wind]; suppu++) {
                const Real *gptr_re = (Real *)gptr;
                const Real *cptr_re = (Real *)cptr;
                re += gptr_re[0] * cptr_re[0] - gptr_re[1] * cptr_re[1];
                im += gptr_re[1] * cptr_re[0] + gptr_re[0] * cptr_re[1];
                gptr++;
                cptr++;
            }
#endif
            gind += gSize;
            cind += sSize[wind];
        }

        data[dind] = Value(re,im);

    }
}

/////////////////////////////////////////////////////////////////////////////////
// Initialize W project convolution function
// - This is application specific and should not need any changes.
//
// wavenumber - temporal frequency (inverse wavelengths)
// uvCellSize - size of one grid cell in wavelengths
// wSize - Size of lookup table in w
// support - Total width of convolution function=2*support+1
// wCellSize - size of one w grid cell in wavelengths
void Benchmark::initC(const Coord uvCellSize, const int wSize,
                      int& support, int& overSample,
                      Coord& wCellSize, std::vector<Value>& C)
{
    //std::cout << "Initializing W projection convolution function" << std::endl;

    // Convolution function. This should be the convolution of the
    // w projection kernel (the Fresnel term) with the convolution
    // function used in the standard case. The latter is needed to
    // suppress aliasing. In practice, we calculate entire function
    // by Fourier transformation. Here we take an approximation that
    // is good enough.
    const int sSizeMax = 2 * support + 1;
    if (wSize<1) {
        std::cout << "initC: require at least 1 plane but wSize" << wSize << std::endl;
    }
    else if (wSize==1) {
        sSize[0] = sSizeMax;
    }

    if (mpirank == 0) {
        std::cout << "  Maximum support = " << support <<
                     " pixels ("<<sSizeMax<<"x"<<sSizeMax<<" kernels)" << std::endl;
        if (wSize>1) {
            std::cout << "  w cellsize = " << wCellSize << " wavelengths" << std::endl;
        }
    }

    int sSizeMin = sSizeMax;
    int offsetCount = 0;
    for (int k = 0; k < wSize; k++) {
        const int wind = double(k - wSize/2);
        const double w = wind * wCellSize;
        double fScale = 0.0;
        if (wind != 0) {
            fScale = uvCellSize*uvCellSize / w;
        }

        cOffset0[k] = offsetCount;
        if (wSize > 1) {
            const Real wPart = w/uvCellSize/uvCellSize;
            const Real aPart = 7.;
            sSize[k] = ceil( sqrt( aPart*aPart + wPart*wPart ) );
            sSize[k] += (sSize[k]+1)%2; // make it odd
        }

        if (sSize[k] < sSizeMin) sSizeMin = sSize[k];

        C.resize(offsetCount + sSize[k]*sSize[k] * overSample*overSample);

        const int cCenter = sSize[k]/2;

        double sumC = 0.0;

        for (int osj = 0; osj < overSample; osj++) {
            for (int osi = 0; osi < overSample; osi++) {
                long int osOffset = sSize[k]*sSize[k] * (osi + overSample*osj) + offsetCount;
                for (int j = 0; j < sSize[k]; j++) {
                    double j2 = std::pow((double(j - cCenter) + double(osj) / double(overSample)), 2);

                    for (int i = 0; i < sSize[k]; i++) {
                        long int cind = i + sSize[k]*j + osOffset;
                        double r2 = j2 + std::pow((double(i - cCenter) + double(osi) / double(overSample)), 2);

                        C[cind] = static_cast<Value>(std::exp(-r2));

                        // for large w the corners where r2 > sSize can lead to w>uv
                        if ((wind != 0) && (r2<sSize[k]/2)) {
                            const Real n = sqrt(1.-r2*fScale/w);
                            const Real phase = -2.*3.141593 * (r2*fScale + w*(n-1.));
                            C[cind] *= static_cast<Value>(n/w) * Value(std::sin(phase),-std::cos(phase));
                        }

                        sumC += std::abs(C[cind]);

                    }
                }
            }
        }

        // Normalise the convolution function
        const Value normC = Value(overSample * overSample / sumC);
        for (int i = 0; i < sSize[k]*sSize[k]*overSample*overSample; i++) {
            C[i+offsetCount] *= normC;
        }

        offsetCount += sSize[k]*sSize[k] * overSample*overSample;

    }

    if (mpirank == 0) {
        float size = offsetCount*sizeof(Value);
        std::string units = " B";
        if ( ceil(log10(size)) > 9 ) {
            size /= 1024*1024*1024;
            units = " GB";
        } else if ( ceil(log10(size)) > 6 ) {
            size /= 1024*1024;
            units = " MB";
        } else if ( ceil(log10(size)) > 3 ) {
            size /= 1024;
            units = " kB";
        }
        if (wSize==1) {
            std::cout << "  Shape of convolution function = [" << sSize[0] << ", " << sSize[0] << ", " <<
                      overSample << ", " << overSample << ", " << wSize << "] = " << size << units << std::endl;
        }
        else {
            std::cout << "  Shape of convolution function = [width, width, " <<
                      overSample << ", " << overSample << ", " << wSize << "] = " << size << units << std::endl;
            std::cout << "   - maximum width = " << sSizeMax << std::endl;
            std::cout << "   - minimum width = " << sSizeMin << std::endl;
        }
    }

}

// Initialize Lookup function
// - This is application specific and should not need any changes.
//
// wavenumber - temporal frequency (inverse wavelengths)
// uvCellSize - size of one grid cell in wavelengths
// gSize - size of grid in pixels (per axis)
// support - Total width of convolution function=2*support+1
// wCellSize - size of one w grid cell in wavelengths
// wSize - Size of lookup table in w
void Benchmark::initCOffset(const std::vector<Coord>& u, const std::vector<Coord>& v,
                            const std::vector<Coord>& w, const std::vector<Coord>& wavenumber,
                            const Coord uvCellSize, const Coord wCellSize,
                            const int wSize, const int gSize, const int overSample)
{
    const int nSamples = u.size();
    const int nChan = wavenumber.size();

    double wmin = +1e12;
    double wmax = -1e12;
    double wave = 0.0;
    double wrms = 0.0;
    // Now calculate the offset for each visibility point
    for (int i = 0; i < nSamples; i++) {
        for (int chan = 0; chan < nChan; chan++) {

            int dind = i * nChan + chan;

            Coord uScaled = wavenumber[chan] * u[i] / uvCellSize;
            iu[dind] = int(uScaled);

            if (uScaled < Coord(iu[dind])) {
                iu[dind] -= 1;
            }

            int fracu = int(overSample * (uScaled - Coord(iu[dind])));
            iu[dind] += gSize / 2;

            Coord vScaled = wavenumber[chan] * v[i] / uvCellSize;
            iv[dind] = int(vScaled);

            if (vScaled < Coord(iv[dind])) {
                iv[dind] -= 1;
            }

            int fracv = int(overSample * (vScaled - Coord(iv[dind])));
            iv[dind] += gSize / 2;

            // The beginning of the convolution function for this point
            int woff = 0;
            if (wCellSize > 0.0) {
                Coord wScaled = wavenumber[chan] * w[i] / wCellSize;
                woff = wSize / 2 + int(wScaled);
            }
            wPlane[dind] = woff;
            cOffset[dind] = sSize[woff]*sSize[woff] * (fracu + overSample*fracv) + cOffset0[woff];
            numPerPlane[woff]++;

            if (w[i]*wavenumber[chan] < wmin) wmin = w[i]*wavenumber[chan];
            if (w[i]*wavenumber[chan] > wmax) wmax = w[i]*wavenumber[chan];
            wave += w[i]*wavenumber[chan];
            wrms += (w[i]*wavenumber[chan]) * (w[i]*wavenumber[chan]);

        }
    }

    if (mpirank == 0) {

        long numGriddedVis = 0;
        long numGriddedPixels = 0;
        for (int woff = 0; woff < wSize; woff++) {
            numGriddedVis += numPerPlane[woff];
            numGriddedPixels += long(numPerPlane[woff]) * long(sSize[woff]*sSize[woff]);
        }

        if (wSize>1) {
            std::cout << "   - average width = " << ceil(sqrt(double(numGriddedPixels)/double(numGriddedVis))) <<
                         ": sqrt( sum(Nkernelpix/wplane x Nvis/wplane) / Nvis )" << std::endl;
        }

        std::cout << "  number of gridded visibilities: "<<numGriddedVis<<
                     ", number of gridded pixels: "<<numGriddedPixels << std::endl;

        wave /= double(nSamples*nChan);
        wrms = sqrt( wrms / double(nSamples*nChan) );
        std::cout << "  w = [" <<wmin<<","<<wmax<< "], ave = "<<wave<<", RMS = "<<wrms << std::endl;

        //for (int woff = 0; woff < wSize; woff++) {
        //    const Real planew = (-(wSize/2) + woff) * wCellSize;
        //    std::cout << "   - w="<<planew<<", kernel="<<sSize[woff]<<"x"<<sSize[woff]<<
        //                 ", Nvis="<<numPerPlane[woff] << std::endl;
        //}
    }

}

long Benchmark::nPixelsGridded()
{

    long numGriddedVis = 0;
    long numGriddedPixels = 0;
    for (int woff = 0; woff < wSize; woff++) {
        numGriddedVis += numPerPlane[woff];
        numGriddedPixels += long(numPerPlane[woff]) * long(sSize[woff]*sSize[woff]);
    }

    if (numGriddedVis != nVisibilitiesGridded()) {
        std::cout << "Visibility count error: "<<numGriddedVis<<" != "<<nVisibilitiesGridded() << std::endl;
        return(-1);
    }

    return numGriddedPixels;

}

std::vector<float> Benchmark::requiredRate()
{

    std::vector<float> rates(2);

    // calculate gridding rate for continuum imaging. Assume 1 process per beam and frequency
    double tmax = 5250.;                // maximum allowed time (seconds)
    long Nvis = (36*35)/2*12*3600./5.;  // number of visibilities (use actual rather than nVisibilitiesGridded)
    long Ncycles = 10;                  // total number of major cycles
    long Npercycle = 3;                 // number of griddings per cycle (grid,degrid,psf)
    long NTT = 3;                       // number of Taylor terms gridded
    long Npol = 1;                      // number of polarisations gridded
    long Nchanperproc = 1;              // number of griddings per cycle (grid,degrid,psf)

    rates[0] = float(Nvis * Ncycles * Npercycle * NTT * Npol * Nchanperproc / tmax);
    std::cout << "continuum gridding rate for " << Nvis * Ncycles * Npercycle * NTT * Npol * Nchanperproc <<
                 " vis gridded is " << rates[0]/1e6 << " Mvis/sec" << std::endl;

    // calculate gridding rate for spectral-line imaging
    tmax = 5500.;
    Nvis = (30*29)/2*12*3600./5.;
    Ncycles = 4;
    Npercycle = 3;
    NTT = 1;
    Npol = 1;
    Nchanperproc = 44; // 18,144 chan * 36 beams / 15,000 compute units (cores) = 44 chan per cores

    rates[1] = float(Nvis * Ncycles * Npercycle * NTT * Npol * Nchanperproc / tmax);

    std::cout << "spectral gridding rate for " << Nvis * Ncycles * Npercycle * NTT * Npol * Nchanperproc <<
                 " vis gridded is " << rates[1]/1e6 << " Mvis/sec" << std::endl;

    return rates;

}

