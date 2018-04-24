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
/// @author Daniel Mitchell <tim.cornwell@csiro.au>

// System includes
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <complex>
#include <vector>
#include <algorithm>
#include <limits>
#include <cassert>

#include <fftw3.h>

// OpenACC includes
#include <openacc.h>

// CUDA includes
#ifdef GPU
#include <cufft.h>
#endif

// Local includes
#include "Stopwatch.h"

#if defined(VERIFY)
	#define RUN_CPU 1
	#define RUN_VERIFY 1
#elif defined(SINGLECPU)
	#define RUN_CPU 1
#endif

using std::cout;
using std::endl;
using std::abs;

// Typedefs for easy testing
// Cost of using double for Coord is low, cost for
// double for float is also low
typedef double Coord;


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
void gridKernel(const std::vector<std::complex<float> >& data, const int support,
                const std::vector<std::complex<float> >& C, const std::vector<int>& cOffset,
                const std::vector<int>& iu, const std::vector<int>& iv,
                std::vector<std::complex<float> >& grid, const int gSize, const bool isPSF)
{
    const int sSize = 2 * support + 1;

    std::complex<float> *d_grid = grid.data();
    const std::complex<float> *d_data = data.data();
    const std::complex<float> *d_C = C.data();

    float dre, dim;
    if ( isPSF ) {
        dre = 1.0;
        dim = 0.0;
    }

    for (int dind = 0; dind < int(data.size()); ++dind) {
        // The actual grid point
        int gind = iu[dind] + gSize * iv[dind] - support;
        // The Convoluton function point from which we offset
        int cind = cOffset[dind];
        int suppu, suppv;
        if ( !isPSF ) {
            dre = d_data[dind].real();
            dim = d_data[dind].imag();
        }

        for (suppv = 0; suppv < sSize; suppv++) {
            for (suppu = 0; suppu < sSize; suppu++) {
                float *dref = (float *)&d_grid[gind+suppv*gSize+suppu];
                const int supp = cind + suppv*sSize+suppu;
                const float reval = dre * d_C[supp].real() - dim * d_C[supp].imag();
                const float imval = dim * d_C[supp].real() + dre * d_C[supp].imag();
                dref[0] = dref[0] + reval;
                dref[1] = dref[1] + imval;
            }
        }

    }
}

void gridKernelACC(const std::vector<std::complex<float> >& data, const int support,
        const std::vector<std::complex<float> >& C, const std::vector<int>& cOffset,
        const std::vector<int>& iu, const std::vector<int>& iv,
        std::vector<std::complex<float> >& grid, const int gSize, const bool isPSF)
{
    const int sSize = 2 * support + 1;

    // std::complex<float> = std::complex<float> = std::complex<float>
    //float *d_grid = (float *)grid.data();
    std::complex<float> *d_grid = grid.data();
    const int d_size = data.size();
    const std::complex<float> *d_data = data.data();
    const std::complex<float> *d_C = C.data();
    const int c_size = C.size();
    const int *d_cOffset = cOffset.data();
    const int *d_iu = iu.data();
    const int *d_iv = iv.data();

    //int dind;
    // Both of the following approaches are the same when running on multicore CPUs.
    // Using "gang vector" here without the inner pragma below sets 1 vis per vector element (c.f. CUDA thread)
    //#pragma acc parallel loop gang vector
    // Using "gang" here with the inner pragma below sets 1 vis per gang, and 1 pixel per vec element

        //int suppu, suppv;

    float dre, dim;
    //std::complex<float> dval;
    if ( isPSF ) {
        dre = 1.0;
        dim = 0.0;
        //dval = 1.0;
    }

#ifdef GPU

    // tile is giving an ~25% improvement over the collapse(2) loop version on initial P100 tests, but is fragile.
    //  - if the total number of cores is above 1386 or so our of 2048 in our P100 tests, verification fails.
    //  - letting the compiler choose using tile(*,*,*) should work but isn't. Will be fixed in the PGI release after
    //    next (currently using pgc++ 18.3-0).
    // wait(1): wait until async(1) is finished...
    #pragma acc parallel loop tile(77,6,3) \
            present(d_grid[0:gSize*gSize],d_data[0:d_size],d_C[0:c_size], \
                    d_cOffset[0:d_size],d_iu[0:d_size],d_iv[0:d_size]) wait(1)
    for (int dind = 0; dind < d_size; ++dind) {
        for (int suppv = 0; suppv < sSize; suppv++) {
            for (int suppu = 0; suppu < sSize; suppu++) {

        int cind = d_cOffset[dind];
        //const float *c_C = (float *)&d_C[cind];
        //#pragma acc cache(c_C[0:2*sSize*sSize])

        // The actual grid point
        int gind = d_iu[dind] + gSize * d_iv[dind] - support;
        // The Convoluton function point from which we offset
        if ( !isPSF ) {
            dre = d_data[dind].real();
            dim = d_data[dind].imag();
            //dval = d_data[dind];
        }

        //#pragma acc loop collapse(2)
        //for (int suppv = 0; suppv < sSize; suppv++) {
        //    for (int suppu = 0; suppu < sSize; suppu++) {
                float *dref = (float *)&d_grid[gind+suppv*gSize+suppu];
                const int supp = cind + suppv*sSize+suppu;
                const float reval = dre * d_C[supp].real() - dim * d_C[supp].imag();
                const float imval = dim * d_C[supp].real() + dre * d_C[supp].imag();
                // note the real mults above are only really needed on the CPUs...
                //const std::complex<float> cval = dval * d_C[cind+suppv*sSize+suppu];
                #pragma acc atomic update
                dref[0] = dref[0] + reval;
                //dref[0] = dref[0] + cval.real();
                #pragma acc atomic update
                dref[1] = dref[1] + imval;
                //dref[1] = dref[1] + cval.imag();
            }
        }
    }
#else
    for (int dind = 0; dind < d_size; ++dind) {
        int cind = d_cOffset[dind];
        //const float *c_C = (float *)&d_C[cind];
        //#pragma acc cache(c_C[0:2*sSize*sSize])

        // The actual grid point
        int gind = d_iu[dind] + gSize * d_iv[dind] - support;
        // The Convoluton function point from which we offset
        int suppu, suppv;
        if ( !isPSF ) {
            dre = d_data[dind].real();
            dim = d_data[dind].imag();
        }

        #pragma acc parallel loop gang vector collapse(2)
        for (suppv = 0; suppv < sSize; suppv++) {
            for (suppu = 0; suppu < sSize; suppu++) {
                float *dref = (float *)&d_grid[gind+suppv*gSize+suppu];
                //const int suppre = 2 * (suppv*sSize+suppu);
                const int supp = cind + suppv*sSize + suppu;
                const float reval = dre * d_C[supp].real() - dim * d_C[supp].imag();
                const float imval = dim * d_C[supp].real() + dre * d_C[supp].imag();
                dref[0] = dref[0] + reval;
                dref[1] = dref[1] + imval;
            }
        }
    }
#endif

}

// Simple degridding to set visibilities
void degridNN(const std::vector<std::complex<float> >& grid, const int gSize,
              const std::vector<int>& iu, const std::vector<int>& iv,
              std::vector<std::complex<float> >& data)
{
    std::complex<float> *d_data = data.data();
    const std::complex<float> *d_grid = grid.data();
    for (int dind = 0; dind < int(data.size()); ++dind) {
        d_data[dind] = d_grid[iu[dind] + gSize * iv[dind]];
    }
}

// Perform degridding
void degridKernel(const std::vector<std::complex<float> >& grid, const int gSize, const int support,
                  const std::vector<std::complex<float> >& C, const std::vector<int>& cOffset,
                  const std::vector<int>& iu, const std::vector<int>& iv,
                  std::vector<std::complex<float> >& data)
{
    const int sSize = 2 * support + 1;

    std::complex<float> *d_data = data.data();
    const std::complex<float> *d_grid = grid.data();
    const std::complex<float> *d_C = C.data();

    for (int dind = 0; dind < int(data.size()); ++dind) {

        // The actual grid point from which we offset
        int gind = iu[dind] + gSize * iv[dind] - support;
        // The Convoluton function point from which we offset
        const int cind = cOffset[dind];

        float re = 0.0, im = 0.0;
        for (int suppv = 0; suppv < sSize; suppv++) {
            for (int suppu = 0; suppu < sSize; suppu++) {
                re = re + d_grid[gind+suppv*gSize+suppu].real() * d_C[cind+suppv*sSize+suppu].real() -
                          d_grid[gind+suppv*gSize+suppu].imag() * d_C[cind+suppv*sSize+suppu].imag();
                im = im + d_grid[gind+suppv*gSize+suppu].imag() * d_C[cind+suppv*sSize+suppu].real() +
                          d_grid[gind+suppv*gSize+suppu].real() * d_C[cind+suppv*sSize+suppu].imag();
            }
        }
        d_data[dind] = std::complex<float>(re,im);

    }
}

void degridKernelACC(const std::vector<std::complex<float> >& grid, const int gSize, const int support,
                     const std::vector<std::complex<float> >& C, const std::vector<int>& cOffset,
                     const std::vector<int>& iu, const std::vector<int>& iv,
                     std::vector<std::complex<float> >& data)
{
    const int sSize = 2 * support + 1;

    const int d_size = data.size();
    std::complex<float> *d_data = data.data();
    const std::complex<float> *d_grid = grid.data();
    const int c_size = C.size();
    const std::complex<float> *d_C = C.data();
    const int *d_cOffset = cOffset.data();
    const int *d_iu = iu.data();
    const int *d_iv = iv.data();

    int dind;

    #pragma acc parallel loop present(d_grid[0:gSize*gSize],d_data[0:d_size],d_C[0:c_size], \
                                      d_cOffset[0:d_size],d_iu[0:d_size],d_iv[0:d_size])
    for (dind = 0; dind < d_size; ++dind) {

        // The actual grid point from which we offset
        int gind = d_iu[dind] + gSize * d_iv[dind] - support;
        // The Convoluton function point from which we offset
        int cind = d_cOffset[dind];
        float re = 0.0, im = 0.0;

/*
#ifdef GPU
        #pragma acc loop reduction(+:re,im) collapse(2)
        for (int suppv = 0; suppv < sSize; suppv++) {
            for (int suppu = 0; suppu < sSize; suppu++) {
                const std::complex<float> cval = d_grid[gind+suppv*gSize+suppu] * d_C[cind+suppv*sSize+suppu];
                re = re + cval.real();
                im = im + cval.imag();
            }
        }
#else
*/
        for (int suppv = 0; suppv < sSize; suppv++) {
            for (int suppu = 0; suppu < sSize; suppu++) {
                re = re + d_grid[gind+suppv*gSize+suppu].real() * d_C[cind+suppv*sSize+suppu].real() -
                          d_grid[gind+suppv*gSize+suppu].imag() * d_C[cind+suppv*sSize+suppu].imag();
                im = im + d_grid[gind+suppv*gSize+suppu].imag() * d_C[cind+suppv*sSize+suppu].real() +
                          d_grid[gind+suppv*gSize+suppu].real() * d_C[cind+suppv*sSize+suppu].imag();
            }
        }
//#endif

        d_data[dind] = std::complex<float>(re,im);

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
           Coord& wCellSize, std::vector<std::complex<float> >& C)
{
    cout << "Initializing W projection convolution function" << endl;
    // DAM -- I don't really understand the following equation. baseline*freq is the array size in wavelengths,
    // but I don't know why the sqrt is used and why there is a multiplication with cellSize rather than a division.
    // In the paper referred to in ../README.md they suggest using rms(w)*FoV for the width (in wavelengths), which
    // would lead to something more like:
    // support = max( 3, ceil( 0.5 * scale*baseline*freq[0] / (cellSize*cellSize) ) )
    // where "scale" reduces the maximum baseline length to the RMS (1/sqrt(3) for uniformaly distributed
    // visibilities, 1/(2+log10(n)/2) or so for n baselines with a Gaussian radial profile).
    support = static_cast<int>(1.5 * sqrt(std::abs(baseline) * static_cast<Coord>(cellSize)
                                          * freq[0]) / cellSize);

    cout << "FoV = " << 180./3.14159265/cellSize << " deg" << endl;

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
         *overSample*wSize*sizeof(std::complex<float>) / (1024*1024) << " MB" << std::endl;
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
if ( i==0 && j==0 ) C[cind] = 1.0;
/*
                        if (w != 0.0) {
                            C[cind] = static_cast<std::complex<float> >(std::cos(r2 / (w * fScale)));
                        } else {
                            C[cind] = static_cast<std::complex<float> >(std::exp(-r2));
                        }
*/

                    }
                }
            }
        }
    }

    // Now normalise the convolution function
    float sumC = 0.0;

    for (int i = 0; i < sSize*sSize*overSample*overSample*wSize; i++) {
        sumC += abs(C[i]);
    }

    for (int i = 0; i < sSize*sSize*overSample*overSample*wSize; i++) {
        C[i] *= std::complex<float>(wSize * overSample * overSample / sumC);
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

// Generate and execute an FFTW plan.
int fftExec(std::vector<std::complex<float> >& grid, const int gSize, const bool forward)
{
    const size_t nPixels = grid.size();
    if (nPixels != gSize*gSize) {
        cout << "bad vector size" << endl;
        return 1;
    }
    if (gSize%2 == 1) {
        cout << "fftExec does not currently support odd sized arrays (fix fftshfit)" << endl;
        return 1;
    }

    std::complex<float> *dataPtr = grid.data();

    // rotate input because the origin for FFTW is at 0, not n/2 (i.e. fftshfit)
    std::vector<std::complex<float> > rotgrid(nPixels);
    std::complex<float> *buffer = rotgrid.data();

    fftwf_plan plan;
    plan = fftwf_plan_dft_2d( gSize, gSize, (fftwf_complex*)buffer, (fftwf_complex*)buffer,
                              (forward) ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE );

    for (int col = 0; col < gSize; col++) {
        const int colin = col * gSize;
        const int colout = ( ( col + gSize/2 ) % gSize ) * gSize;
        //std::rotate_copy(dataPtr + colin,
        //                 dataPtr + colin + (gSize / 2),
        //                 dataPtr + colin + gSize,
        //                 reinterpret_cast<std::complex<float> *>(buffer + colout));
        for (int row = 0; row < gSize/2; row++) {
            buffer[colout + row] = dataPtr[colin + gSize/2 + row];
            buffer[colout + gSize/2 + row] = dataPtr[colin + row];
        }
    }

    fftwf_execute(plan);

    // rotate back
    for (int col = 0; col < gSize; col++) {
        const int colin = col * gSize;
        const int colout = ( ( col + gSize/2 ) % gSize ) * gSize;
        //std::rotate_copy(reinterpret_cast<std::complex<float> *>(buffer) + colin,
        //                 reinterpret_cast<std::complex<float> *>(buffer) + colin + (gSize / 2),
        //                 reinterpret_cast<std::complex<float> *>(buffer) + colin + gSize,
        //                 dataPtr + colout);
        for (int row = 0; row < gSize/2; row++) {
            dataPtr[colout + row] = buffer[colin + gSize/2 + row];
            dataPtr[colout + gSize/2 + row] = buffer[colin + row];
        }
    }

    // Delete the plan and temporary buffer
    fftwf_destroy_plan(plan);
    //fftwf_free(buffer);

    return 0;

}

// Generate and execute a CUFFT plan.
int fftExecGPU(std::vector<std::complex<float> >& grid, const int gSize, const bool forward)
{
    #ifdef GPU
    const size_t nPixels = grid.size();
    if (nPixels != gSize*gSize) {
        cout << "bad vector size" << endl;
        return 1;
    }
    if (gSize%2 == 1) {
        cout << "fftExecGPU does not currently support odd sized arrays (fix fftshfit)" << endl;
        return 1;
    }

    cufftHandle plan;

    if ( cufftPlan2d( &plan, gSize, gSize, CUFFT_C2C ) != CUFFT_SUCCESS ) {
        cout << "CUFFT error: Plan creation failed" << endl;
        return 1;
    }

    std::complex<float> *dataPtr = grid.data();

    // use a buffer to rotate the pixels for the fft
    std::vector<std::complex<float> > rotgrid(nPixels);
    std::complex<float> *buffer = rotgrid.data();

    // buffer only ever needs to be on the device. So create it then ignore the cpu version
    #pragma acc enter data create(buffer[0:gSize*gSize])

    // rotate input because the origin for CUFFT is at 0, not n/2 (i.e. fftshfit)
    #pragma acc parallel loop collapse(2) present(dataPtr[0:gSize*gSize],buffer[0:gSize*gSize])
    for (int col = 0; col < gSize; col++) {
        for (int row = 0; row < gSize/2; row++) {
            const int colin = col * gSize;
            const int colout = ( ( col + gSize/2 ) % gSize ) * gSize;
            buffer[colout + row] = dataPtr[colin + gSize/2 + row];
            buffer[colout + gSize/2 + row] = dataPtr[colin + row];
        }
    }

    cufftResult fftErr;
    #pragma acc host_data use_device(buffer)
    {
        fftErr = cufftExecC2C(plan, (cufftComplex*)buffer, (cufftComplex*)buffer, (forward) ? CUFFT_FORWARD : CUFFT_INVERSE);
    }
    if ( fftErr != CUFFT_SUCCESS ) {
        cout << "CUFFT error: Forward FFT failed" << endl;
        return 1;
    }
 
    // rotate back
    #pragma acc parallel loop collapse(2) present(dataPtr[0:gSize*gSize],buffer[0:gSize*gSize])
    for (int col = 0; col < gSize; col++) {
        for (int row = 0; row < gSize/2; row++) {
            const int colin = col * gSize;
            const int colout = ( ( col + gSize/2 ) % gSize ) * gSize;
            dataPtr[colout + row] = buffer[colin + gSize/2 + row];
            dataPtr[colout + gSize/2 + row] = buffer[colin + row];
        }
    }

    #pragma acc exit data delete(buffer[0:gSize*gSize])

    // Delete the plan
    cufftDestroy(plan);
    #endif

    return 0;

}

// Generate and execute an FFTW plan.
void fftFix(std::vector<std::complex<float> >& grid, const float scale)
{
    const size_t nPixels = grid.size();
    for (size_t i = 0; i < nPixels; i++) {
        grid[i] = grid[i].real() * scale;
    }
}

// Generate and execute an FFTW plan.
void fftFixGPU(std::vector<std::complex<float> >& grid, const float scale)
{
    #ifdef GPU
    const size_t nPixels = grid.size();
    std::complex<float> *dataPtr = grid.data();
    #pragma acc parallel loop present(dataPtr[0:nPixels])
    for (size_t i = 0; i < nPixels; i++) {
        dataPtr[i] = dataPtr[i].real() * scale;
    }
    #endif
}

// currently only used during varification, so remove otherwise to suppress compiler warnings...
#ifdef RUN_VERIFY
static bool abs_compare(std::complex<float> a, std::complex<float> b)
{
    return (std::abs(a) < std::abs(b));
}
#endif

// Return a pseudo-random integer in the range 0..2147483647
// Based on an algorithm in Kernighan & Ritchie, "The C Programming Language"
static unsigned long next = 1;
int randomInt()
{
    const unsigned int maxint = std::numeric_limits<int>::max();
    next = next * 1103515245 + 12345;
    return ((unsigned int)(next / 65536) % maxint);
}

// ------------------------------------------------------------------------- //
// Hogbom stuff

void writeImage(const std::string& filename, std::vector<std::complex<float> >& image)
{
    std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
    std::vector<float> realpart(image.size());
    for (int i=0; i<image.size(); i++) realpart[i] = image[i].real();
    file.write(reinterpret_cast<char *>(&realpart[0]), realpart.size() * sizeof(float));
    file.close();
}

struct Position {
    Position(int _x, int _y) : x(_x), y(_y) { };
    int x;
    int y;
};

Position idxToPos(const int idx, const size_t width)
{
    const int y = idx / width;
    const int x = idx % width;
    return Position(x, y);
}

size_t posToIdx(const size_t width, const Position& pos)
{
    return (pos.y * width) + pos.x;
}

void findPeak(const std::vector<std::complex<float> >& image,
              float& maxVal, size_t& maxPos)
{
    maxVal = 0.0;
    maxPos = 0;
    const size_t size = image.size();
    for (size_t i = 0; i < size; ++i) {
        if (abs(image[i].real()) > abs(maxVal)) {
            maxVal = image[i].real();
            maxPos = i;
        }
    }
}

void findPeakACC(const std::complex<float> *data,
              float& maxVal, size_t& maxPos, const size_t size)
{
    size_t tmpPos=0;
    float threadAbsMaxVal = 0.0;

    #pragma acc parallel loop reduction(max:threadAbsMaxVal) gang vector present(data[0:size])
    for (size_t i = 0; i < size; ++i) {
        threadAbsMaxVal = fmaxf( threadAbsMaxVal, abs(data[i].real()) );
    }
    #pragma acc parallel loop gang vector present(data[0:size]) copyout(tmpPos)
    for (size_t i = 0; i < size; ++i) {
        if (abs(data[i].real()) == threadAbsMaxVal) tmpPos = i;
    }

    maxVal = threadAbsMaxVal;
    maxPos = tmpPos;

}

void subtractPsf(const std::vector<std::complex<float> >& psf,
        const size_t psfWidth,
        std::vector<std::complex<float> >& residual,
        const size_t residualWidth,
        const size_t peakPos, const size_t psfPeakPos,
        const float absPeakVal,
        const float gain)
{
    const int rx = idxToPos(peakPos, residualWidth).x;
    const int ry = idxToPos(peakPos, residualWidth).y;

    const int px = idxToPos(psfPeakPos, psfWidth).x;
    const int py = idxToPos(psfPeakPos, psfWidth).y;

    const int diffx = rx - px;
    const int diffy = ry - px;

    const int startx = std::max(0, rx - px);
    const int starty = std::max(0, ry - py);

    const int stopx = std::min(residualWidth - 1, rx + (psfWidth - px - 1));
    const int stopy = std::min(residualWidth - 1, ry + (psfWidth - py - 1));

    for (int y = starty; y <= stopy; ++y) {
        for (int x = startx; x <= stopx; ++x) {
            residual[posToIdx(residualWidth, Position(x, y))] -= gain * absPeakVal
                * psf[posToIdx(psfWidth, Position(x - diffx, y - diffy))].real();
        }
    }
}

void subtractPsfACC(const std::complex<float> *psfdata,
        const size_t psfWidth,
        std::complex<float> *resdata,
        const size_t residualWidth,
        const size_t peakPos, const size_t psfPeakPos,
        const float absPeakVal,
        const float gain)
{
    const int rx = idxToPos(peakPos, residualWidth).x;
    const int ry = idxToPos(peakPos, residualWidth).y;

    const int px = idxToPos(psfPeakPos, psfWidth).x;
    const int py = idxToPos(psfPeakPos, psfWidth).y;

    const int diffx = rx - px;
    const int diffy = ry - px;

    const int startx = std::max(0, rx - px);
    const int starty = std::max(0, ry - py);

    const int stopx = std::min(residualWidth - 1, rx + (psfWidth - px - 1));
    const int stopy = std::min(residualWidth - 1, ry + (psfWidth - py - 1));

    #pragma acc parallel loop collapse(2) gang vector present(resdata[0:residualWidth*residualWidth],psfdata[0:psfWidth*psfWidth])
    for (int y = starty; y <= stopy; ++y) {
        for (int x = startx; x <= stopx; ++x) {
            resdata[posToIdx(residualWidth, Position(x, y))] -= gain * absPeakVal
                * psfdata[posToIdx(psfWidth, Position(x - diffx, y - diffy))].real();
        }
    }
}

void deconvolve(std::vector<std::complex<float> >& residual,
                const size_t dirtyWidth,
                const std::vector<std::complex<float> >& psf,
                const size_t psfWidth,
                std::vector<std::complex<float> >& model,
                const int g_niters)
{

    const float g_gain = 0.1;
    // disable this to keep things more readily comparable
    //float g_threshold;

    // Find the peak of the PSF
    float psfPeakVal = 0.0;
    size_t psfPeakPos = 0;
    findPeak(psf, psfPeakVal, psfPeakPos);
    cout << "    PSF peak (cpu): " << "Maximum = " << psfPeakVal << " at location "
         << idxToPos(psfPeakPos, psfWidth).x << "," << idxToPos(psfPeakPos, psfWidth).y << endl;

    for (unsigned int i = 0; i < g_niters; ++i) {
        // Find the peak in the residual image
        float absPeakVal = 0.0;
        size_t absPeakPos = 0;
        findPeak(residual, absPeakVal, absPeakPos);
        if (i==0) {
            cout << "    dirty peak (cpu): " << "Maximum = " << absPeakVal << " at location "
                 << idxToPos(absPeakPos, dirtyWidth).x << "," << idxToPos(absPeakPos, dirtyWidth).y << endl;
        }
        // Check if threshold has been reached
        //if (i==0) g_threshold = 1e-3 * abs(absPeakVal);
        //if (abs(absPeakVal) < g_threshold) {
        //    cout << "Reached stopping threshold" << endl;
        //    break;
        //}

        // Add to model
        model[absPeakPos] += absPeakVal * g_gain;

        // Subtract the PSF from the residual image
        subtractPsf(psf, psfWidth, residual, dirtyWidth, absPeakPos, psfPeakPos, absPeakVal, g_gain);
    }
}

void deconvolveACC(std::vector<std::complex<float> >& residual,
                const size_t dirtyWidth,
                const std::vector<std::complex<float> >& psf,
                const size_t psfWidth,
                std::vector<std::complex<float> >& model,
                const int g_niters)
{

    const float g_gain = 0.1;
    // disable this to keep things more readily comparable
    //float g_threshold;

    // referece the basic data arrays for use in the parallel loop
    const std::complex<float> *psfdata = psf.data();
    std::complex<float> *resdata = residual.data();
    const size_t psfsize = psf.size();
    const size_t ressize = residual.size();

    // Find the peak of the PSF
    float psfPeakVal = 0.0;
    size_t psfPeakPos = 0;
    findPeakACC(psfdata, psfPeakVal, psfPeakPos, psfsize);
    cout << "    PSF peak (acc): " << "Maximum = " << psfPeakVal << " at location "
         << idxToPos(psfPeakPos, psfWidth).x << "," << idxToPos(psfPeakPos, psfWidth).y << endl;

    for (unsigned int i = 0; i < g_niters; ++i) {
        // Find the peak in the residual image
        float absPeakVal = 0.0;
        size_t absPeakPos = 0;
        findPeakACC(resdata, absPeakVal, absPeakPos, ressize);
        if (i==0) {
            cout << "    dirty peak (acc): " << "Maximum = " << absPeakVal << " at location "
                 << idxToPos(absPeakPos, dirtyWidth).x << "," << idxToPos(absPeakPos, dirtyWidth).y << endl;
        }

        // Check if threshold has been reached
        //if (i==0) g_threshold = 1e-3 * abs(absPeakVal);
        //if (abs(absPeakVal) < g_threshold) {
        //    cout << "Reached stopping threshold" << endl;
        //    break;
        //}

        // Add to model
        model[absPeakPos] += absPeakVal * g_gain;

        // Subtract the PSF from the residual image
        subtractPsfACC(psfdata, psfWidth, resdata, dirtyWidth, absPeakPos, psfPeakPos, absPeakVal, g_gain);
    }
}

void usage() {
    cout << "usage: tMajorACC [-h] [option]" << endl;
    cout << "-n num\t change the number of data samples to num." << endl;
    cout << "-w num\t change the number of lookup planes in w projection to num." << endl;
    cout << "-c num\t change the number of spectral channels to num." << endl;
    cout << "-f val\t reduce the field of view by a factor of val (=> reduce the kernel size)." << endl;
}

// ------------------------------------------------------------------------- //
// Main testing routine
int main(int argc, char* argv[])
{
    // Change these if necessary to adjust run time
    int nSamples = 160000; // Number of data samples
    int wSize = 33; // Number of lookup planes in w projection
    int nChan = 1; // Number of spectral channels
    Coord cellSize = 5.0; // Cellsize of output grid in wavelengths

    if (argc > 1){
        for (int i=0; i < argc; i++){
            if (argv[i][0] == '-') {
                if (argv[i][1] == 'n'){
                    nSamples = atoi(argv[i+1]);
                }
                else if (argv[i][1] == 'w'){
                    wSize = atoi(argv[i+1]);
                }
                else if (argv[i][1] == 'c'){
                    nChan = atoi(argv[i+1]);
                }
                else if (argv[i][1] == 'f') {
                    cellSize *= atof(argv[i+1]);
                    i++;
                }
                else {
                    usage();
                    return 1;
                }
            }
        }
    }
    cout << "nSamples = " << nSamples <<endl;
    cout << "nChan = " << nChan <<endl;
    cout << "wSize = " << wSize <<endl;
    cout << "cellSize = " << cellSize <<endl;

    // Don't change any of these numbers unless you know what you are doing!
    const int gSize = 4096; // Size of output grid in pixels
    const int baseline = 2000; // Maximum baseline in meters

    const int nMajor = 5; // Number of major cycle iterations
    const int nMinor = 100; // Number of minor cycle iterations

    const unsigned int maxint = std::numeric_limits<int>::max();

    // Initialize the uvw data 
    std::vector<Coord> u(nSamples);
    std::vector<Coord> v(nSamples);
    std::vector<Coord> w(nSamples);
    for (int i = 0; i < nSamples; i++) {
        u[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;
        v[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;
        w[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;
    }

    std::vector<std::complex<float> > grid(gSize*gSize);
    grid.assign(grid.size(), std::complex<float>(0.0));

    // Measure frequency in inverse wavelengths
    std::vector<Coord> freq(nChan);

    for (int i = 0; i < nChan; i++) {
        freq[i] = (1.4e9 - 2.0e5 * Coord(i) / Coord(nChan)) / 2.998e8;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Initialize convolution function and offsets
    ///////////////////////////////////////////////////////////////////////////

    std::vector<std::complex<float> > C;
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

    // asynchronously copy coords to the device while we are doing other initialisation
    int *iu_d = iu.data();
    int *iv_d = iv.data();
    int *cOffset_d = cOffset.data();
    std::complex<float> *C_d = C.data();
    #pragma acc enter data copyin(C_d[0:sSize*sSize*overSample*overSample*wSize], cOffset_d[0:nSamples*nChan], \
                                  iu_d[0:nSamples*nChan], iv_d[0:nSamples*nChan]) async(1)

    const double griddings = (double(nSamples * nChan) * double((sSize) * (sSize)));

    ///////////////////////////////////////////////////////////////////////////
    // Generate initial sky model and visibilties
    ///////////////////////////////////////////////////////////////////////////

    // make an image of point sources (the true sky)
    const int nSources = 100;
    std::vector<std::complex<float> > trueGrid(gSize*gSize);
    trueGrid.assign(trueGrid.size(), std::complex<float>(0.0));
    // record a few test positions for later
    for (int i = 0; i < nSources; i++) {
        int l = gSize * Coord(randomInt()) / Coord(maxint);
        int m = gSize * Coord(randomInt()) / Coord(maxint);
        trueGrid[m*gSize+l] += std::complex<float>(randomInt()) / std::complex<float>(maxint);
    }

    // FFT to the true uv grid
    if ( fftExec(trueGrid, gSize, true) != 0 ) {
        cout << "fftExec error" << endl;
        return -1;
    }

    // degrid true visibiltiies from sky true model

    // generate data on GPU, but in the CPU object so that we have to send it to the GPU for profiling
    std::vector<std::complex<float> > visData(nSamples*nChan);
    std::complex<float> *visData_d = visData.data();
    std::complex<float> *trueGrid_d = trueGrid.data();
    #pragma acc enter data create(visData_d[0:nSamples*nChan]) copyin(trueGrid_d[0:gSize*gSize])
    degridKernelACC(trueGrid, gSize, support, C, cOffset, iu, iv, visData);
    #pragma acc exit data delete(trueGrid_d[0:gSize*gSize]) async(2)
    // pull the data back to the CPU and delete/deallocate the GPU copy
    #pragma acc exit data copyout(visData_d[0:nSamples*nChan])

#ifdef RUN_CPU
    // make a single-core cpu copy
    std::vector<std::complex<float> > cpuData(visData);
    std::vector<std::complex<float> > cpuModel(nSamples*nChan);
    for (int i = 0; i < nSamples*nChan; i++) {
        cpuModel[i] = 0.0;
    }
    // set main single-core cpu scratch arrays
    std::vector<std::complex<float> > cpuPsfGrid(gSize*gSize);
    std::vector<std::complex<float> > cpuImgGrid(gSize*gSize);
#endif

    // make an acc copy and send initial visibility data to the device
    std::vector<std::complex<float> > accData(visData);
    std::vector<std::complex<float> > accModel(nSamples*nChan);
    std::complex<float> *accData_d = accData.data();
    #pragma acc enter data copyin(accData_d[0:nSamples*nChan]) async(1)
    std::complex<float> *accModel_d = accModel.data();
    #pragma acc enter data create(accModel_d[0:nSamples*nChan])
    #pragma acc parallel loop present(accModel_d[0:nSamples*nChan])
    for (int i = 0; i < nSamples*nChan; i++) {
        accModel_d[i] = 0.0;
    }
    // set main acc scratch arrays
    std::vector<std::complex<float> > accPsfGrid(gSize*gSize);
    std::vector<std::complex<float> > accImgGrid(gSize*gSize);
    std::complex<float> *accPsfGrid_d = accPsfGrid.data();
    std::complex<float> *accImgGrid_d = accImgGrid.data();
    #pragma acc enter data create(accPsfGrid_d[0:gSize*gSize], accImgGrid_d[0:gSize*gSize])

    // initialise timers
#ifdef RUN_CPU
    double psfCpuTimer = 0.0;
    double imgCpuTimer = 0.0;
    double ifftCpuTimer = 0.0;
    double HogbomCpuTimer = 0.0;
    double fftCpuTimer = 0.0;
    double degridCpuTimer = 0.0;
#endif
    double psfAccTimer = 0.0;
    double imgAccTimer = 0.0;
    double ifftAccTimer = 0.0;
    double HogbomAccTimer = 0.0;
    double fftAccTimer = 0.0;
    double degridAccTimer = 0.0;
#ifdef RUN_VERIFY
    std::vector<std::complex<float> > cpuuvPsf(gSize*gSize);
    std::vector<std::complex<float> > cpuuvGrid(gSize*gSize);
    std::vector<std::complex<float> > cpulmPsf(gSize*gSize);
    std::vector<std::complex<float> > cpulmGrid(gSize*gSize);
    std::vector<std::complex<float> > cpulmRes(gSize*gSize);
    std::vector<std::complex<float> > accuvPsf(gSize*gSize);
    std::vector<std::complex<float> > accuvGrid(gSize*gSize);
    std::vector<std::complex<float> > acclmPsf(gSize*gSize);
    std::vector<std::complex<float> > acclmGrid(gSize*gSize);
    std::vector<std::complex<float> > acclmRes(gSize*gSize);
    float psfScale = 1.0;
#endif

    cout << endl;

    for ( int it_major=0; it_major<nMajor; ++it_major ) {

        cout << "cycle " << it_major << endl;

#ifdef RUN_CPU

        ///////////////////////////////////////////////////////////////////////////
        // CPU single core
        ///////////////////////////////////////////////////////////////////////////

        Stopwatch sw_cpu;
        sw_cpu.start();

        //-----------------------------------------------------------------------//
        // DO GRIDDING
        if (it_major == 0)
        {
            Stopwatch sw;
            sw.start();
            cpuPsfGrid.assign(cpuPsfGrid.size(), std::complex<float>(0.0));
            gridKernel(cpuData, support, C, cOffset, iu, iv, cpuPsfGrid, gSize, true);
            psfCpuTimer += sw.stop();
#ifdef RUN_VERIFY
            // Save copies for varification
            cpuuvPsf = cpuPsfGrid;
#endif
        }
        {
            Stopwatch sw;
            sw.start();
            cpuImgGrid.assign(cpuImgGrid.size(), std::complex<float>(0.0));
            gridKernel(cpuData, support, C, cOffset, iu, iv, cpuImgGrid, gSize, false);
            imgCpuTimer += sw.stop();
#ifdef RUN_VERIFY
            // Save copies for varification
            cpuuvGrid = cpuImgGrid;
#endif
        }

        //-----------------------------------------------------------------------//
        // Form dirty image and run the minor cycle

        // FFT gridded data to form psf image
        if (it_major == 0)
        {
            if ( fftExec(cpuPsfGrid, gSize, false) != 0 ) {
                cout << "inverse fftExec error" << endl;
                return -1;
            }
            fftFix(cpuPsfGrid, 1.0/float(cpuData.size()));
#ifdef RUN_VERIFY
            // Save copies for varification
            cpulmPsf = cpuPsfGrid;
#endif
        }
 
        // FFT gridded data to form dirty image
        {
            Stopwatch sw;
            sw.start();
            if ( fftExec(cpuImgGrid, gSize, false) != 0 ) {
                cout << "inverse fftExec error" << endl;
                return -1;
            }
            fftFix(cpuImgGrid, 1.0/float(cpuData.size()));
            ifftCpuTimer += sw.stop();
#ifdef RUN_VERIFY
            // Save copies for varification
            cpulmGrid = cpuImgGrid;
#endif
        }

        //-------------------------------------------------------------------//
        // Do Hogbom CLEAN

        std::vector<std::complex<float> > cpuModelGrid(cpuImgGrid.size());
        cpuModelGrid.assign(cpuModelGrid.size(), std::complex<float>(0.0));
        {
            Stopwatch sw;
            sw.start();
            deconvolve(cpuImgGrid, gSize, cpuPsfGrid, gSize, cpuModelGrid, nMinor);
            HogbomCpuTimer += sw.stop();
#ifdef RUN_VERIFY
            // Save a copy for varification
            cpulmRes = cpuImgGrid;
#endif
        }

        // Set the main scratch grid to the model (just use the model?)
        cpuImgGrid = cpuModelGrid;

        //-------------------------------------------------------------------//
        // FFT deconvolved model image for degridding
        {
            Stopwatch sw;
            sw.start();
            // this should be the model, not cpuImgGrid
            if ( fftExec(cpuImgGrid, gSize, true) != 0 ) {
                cout << "forward fftExec error" << endl;
                return -1;
            }
            fftCpuTimer += sw.stop();
        }

        //-----------------------------------------------------------------------//
        // DO DEGRIDDING
        {
            Stopwatch sw;
            sw.start();
            degridKernel(cpuImgGrid, gSize, support, C, cOffset, iu, iv, cpuModel);
            degridCpuTimer += sw.stop();
        }

        double cpu_time = sw_cpu.stop();
        cout << "    time " << cpu_time << " (s)" << endl;

#endif

        ///////////////////////////////////////////////////////////////////////////
        // OpenACC
        ///////////////////////////////////////////////////////////////////////////

        Stopwatch sw_acc;
        sw_acc.start();

        //-----------------------------------------------------------------------//
        // DO GRIDDING
        if (it_major == 0)
        {
            // Time is measured inside this function call, unlike the CPU versions
            Stopwatch sw;
            sw.start();
            #pragma acc parallel loop present(accPsfGrid_d[0:gSize*gSize])
            for (unsigned int i = 0; i < gSize*gSize; ++i) {
                accPsfGrid_d[i] = 0.0;
            }
            gridKernelACC(accData, support, C, cOffset, iu, iv, accPsfGrid, gSize, true);
            psfAccTimer += sw.stop();
#ifdef RUN_VERIFY
            // Save copies for varification
            #pragma acc update host(accPsfGrid_d[0:gSize*gSize])
            accuvPsf = accPsfGrid;
#endif
        }
        {
            // Time is measured inside this function call, unlike the CPU versions
            Stopwatch sw;
            sw.start();
            #pragma acc parallel loop present(accImgGrid_d[0:gSize*gSize])
            for (unsigned int i = 0; i < gSize*gSize; ++i) {
                accImgGrid_d[i] = 0.0;
            }
            gridKernelACC(accData, support, C, cOffset, iu, iv, accImgGrid, gSize, false);
            imgAccTimer += sw.stop();
#ifdef RUN_VERIFY
            // Save copies for varification
            #pragma acc update host(accImgGrid_d[0:gSize*gSize])
            accuvGrid = accImgGrid;
#endif
        }

        //-----------------------------------------------------------------------//
        // Form dirty image and run the minor cycle

        // FFT gridded data to form psf image
        if (it_major == 0)
        {
            #ifdef GPU
            // Use CUFFT
            if ( fftExecGPU(accPsfGrid, gSize, false) != 0 ) {
                cout << "inverse fftExecGPU error" << endl;
                return -1;
            }
            fftFixGPU(accPsfGrid, 1.0/float(accData.size()));
            #else
            // Use FFTW
            //  - note that we should really enable multithreaded FFTW in this situation
            if ( fftExec(accPsfGrid, gSize, false) != 0 ) {
                cout << "inverse fftExec error" << endl;
                return -1;
            }
            fftFix(accPsfGrid, 1.0/float(accData.size()));
            #endif
#ifdef RUN_VERIFY
            // Save copies for varification
            #pragma acc update host(accPsfGrid_d[0:gSize*gSize])
            acclmPsf = accPsfGrid;
#endif
        }

        // FFT gridded data to form dirty image
        {
            Stopwatch sw;
            sw.start();
            #ifdef GPU
            // Use CUFFT
            if ( fftExecGPU(accImgGrid, gSize, false) != 0 ) {
                cout << "inverse fftExecGPU error" << endl;
                return -1;
            }
            fftFixGPU(accImgGrid, 1.0/float(accData.size()));
            #else
            // Use FFTW
            //  - note that we should really enable multithreaded FFTW in this situation
            if ( fftExec(accImgGrid, gSize, false) != 0 ) {
                cout << "inverse fftExec error" << endl;
                return -1;
            }
            fftFix(accImgGrid, 1.0/float(accData.size()));
            #endif
            ifftAccTimer += sw.stop();
#ifdef RUN_VERIFY
            // Save copies for varification
            #pragma acc update host(accImgGrid_d[0:gSize*gSize])
            acclmGrid = accImgGrid;
#endif
        }

        //-------------------------------------------------------------------//
        // Do Hogbom CLEAN

        std::vector<std::complex<float> > accModelGrid(accPsfGrid.size());
        accModelGrid.assign(accModelGrid.size(), std::complex<float>(0.0));

        {
            Stopwatch sw;
            sw.start();
            deconvolveACC(accImgGrid, gSize, accPsfGrid, gSize, accModelGrid, nMinor);
            HogbomAccTimer += sw.stop();
        }

#ifdef RUN_VERIFY
        // Save a copy for varification
        #pragma acc update host(accImgGrid_d[0:gSize*gSize])
        acclmRes = accImgGrid;
#endif

        accImgGrid = accModelGrid;
        #pragma acc update device(accImgGrid_d[0:gSize*gSize])

        //-------------------------------------------------------------------//
        // FFT deconvolved model image for degridding
        {
            Stopwatch sw;
            sw.start();
            #ifdef GPU
            // Use CUFFT
            if ( fftExecGPU(accImgGrid, gSize, true) != 0 ) {
                cout << "forward fftExecGPU error" << endl;
                return -1;
            }
            #else
            // Use FFTW
            //  - note that we should really enable multithreaded FFTW in this situation
            if ( fftExec(accImgGrid, gSize, true) != 0 ) {
                cout << "forward fftExec error" << endl;
                return -1;
            }
            #endif
            fftAccTimer += sw.stop();
        }

        //-------------------------------------------------------------------//
        // DO DEGRIDDING
        {
            // Time is measured inside this function call, unlike the CPU versions
            Stopwatch sw;
            sw.start();
            degridKernelACC(accImgGrid, gSize, support, C, cOffset, iu, iv, accModel);
            degridAccTimer += sw.stop();
        }

        //-------------------------------------------------------------------//
        // Copy GPU data back to CPU

        //#pragma acc exit data copyout(accImgGrid_d[0:gSize*gSize],accModel_d[0:nSamples*nChan]) 
        #pragma acc update host(accImgGrid_d[0:gSize*gSize],accModel_d[0:nSamples*nChan])

        double acc_time = sw_acc.stop();
        cout << "    time " << acc_time << " (s)" << endl;

#ifdef RUN_VERIFY

        ///////////////////////////////////////////////////////////////////////
        // Verify results
        ///////////////////////////////////////////////////////////////////////
        cout << "    verifying:";

        // store the location and value of the maximum PSF pixel to normalise everything by
        std::vector<std::complex<float> >::iterator maxLoc;
        int maxPixel;
        if (it_major == 0)
        {
            maxLoc = std::max_element(cpulmPsf.begin(), cpulmPsf.end(), abs_compare);
            maxPixel = std::distance(cpulmPsf.begin(), maxLoc);
            psfScale = 1.0 / fabs(cpulmPsf[maxPixel].real());
        }

        // set a threshold factor
        const float thresh = 1e-5;

        //-------------------------------------------------------------------//
        // Verify gridding results
        if (it_major == 0)
        {
            cout << " psfgrid";
         
            if (cpuuvPsf.size() != accuvPsf.size()) {
                cout << endl;
                cout << "Fail (PSF grid sizes differ)" << endl;
                return 1;
            }

            maxLoc = std::max_element(cpuuvPsf.begin(), cpuuvPsf.end(), abs_compare);
            maxPixel = std::distance(cpuuvPsf.begin(), maxLoc);

            for (unsigned int i = 0; i < cpuuvPsf.size(); ++i) {
                if (fabs(cpuuvPsf[i].real() - accuvPsf[i].real()) / fabs(cpuuvPsf[maxPixel].real()) > thresh) {
                    cout << "Fail (Expected " << cpuuvPsf[i].real() << " got "
                             << accuvPsf[i].real() << " at index " << i << ")"
                             << endl;
                    return 1;
                }
            }
        }

        cout << " grid";

        if (cpuuvGrid.size() != accuvGrid.size()) {
            cout << endl;
            cout << "Fail (Dirty image grid sizes differ)" << endl;
            return 1;
        }

        maxLoc = std::max_element(cpuuvGrid.begin(), cpuuvGrid.end(), abs_compare);
        maxPixel = std::distance(cpuuvGrid.begin(), maxLoc);

        for (unsigned int i = 0; i < cpuuvGrid.size(); ++i) {
            if (fabs(cpuuvGrid[i].real() - accuvGrid[i].real()) / fabs(cpuuvGrid[maxPixel].real()) > thresh) {
                cout << endl;
                cout << "Fail (Expected " << cpuuvGrid[i].real() << " got "
                         << accuvGrid[i].real() << " at index " << i << ")"
                         << endl;
                return 1;
            }
        }

        //-------------------------------------------------------------------//
        // Verify Inverse FFT results
        cout << " ifft";

        if (cpulmPsf.size() != acclmPsf.size()) {
            cout << endl;
            cout << "Fail (PSF grid sizes differ)" << endl;
            return 1;
        }

        for (unsigned int i = 0; i < cpulmPsf.size(); ++i) {
            if (fabs(cpulmPsf[i].real() - acclmPsf[i].real()) * psfScale > thresh) {
                cout << endl;
                cout << "Fail for PSF (Expected " << cpulmPsf[i].real() << " got "
                         << acclmPsf[i].real() << " at index " << i << ")"
                         << endl;
                return 1;
            }
        }

        if (cpulmGrid.size() != acclmGrid.size()) {
            cout << endl;
            cout << "Fail (Dirty image grid sizes differ)" << endl;
            return 1;
        }

        for (unsigned int i = 0; i < cpulmGrid.size(); ++i) {
            if (fabs(cpulmGrid[i].real() - acclmGrid[i].real()) * psfScale > thresh) {
                cout << endl;
                cout << "Fail for dirty image (Expected " << cpulmGrid[i].real() << " got "
                         << acclmGrid[i].real() << " at index " << i << ")"
                         << endl;
                return 1;
            }
        }

        //-------------------------------------------------------------------//
        // Verify Hogbom clean results
        cout << " clean";

        if (cpulmRes.size() != acclmRes.size()) {
            cout << endl;
            cout << "Fail (Residual grid sizes differ)" << endl;
            return 1;
        }

        for (unsigned int i = 0; i < cpulmRes.size(); ++i) {
            if (fabs(cpulmRes[i].real() - acclmRes[i].real()) * psfScale > thresh) {
                cout << endl;
                cout << "Fail for residual (Expected " << cpulmRes[i].real() << " got "
                         << acclmRes[i].real() << " at index " << i << ")"
                         << endl;
                return 1;
            }
        }

        if (cpuModelGrid.size() != accModelGrid.size()) {
            cout << endl;
            cout << "Fail (Model grid sizes differ)" << endl;
            return 1;
        }

        for (unsigned int i = 0; i < cpuModelGrid.size(); ++i) {
            if (fabs(cpuModelGrid[i].real() - accModelGrid[i].real()) * psfScale > thresh) {
                cout << endl;
                cout << "Fail for model (Expected " << cpuModelGrid[i].real() << " got "
                         << accModelGrid[i].real() << " at index " << i << ")"
                         << endl;
                return 1;
            }
        }

        //-------------------------------------------------------------------//
        // Verify Forward FFT results
        cout << " fft";

        if (cpuImgGrid.size() != accImgGrid.size()) {
            cout << endl;
            cout << "Fail (Grid sizes differ)" << endl;
            return 1;
        }

        for (unsigned int i = 0; i < cpuImgGrid.size(); ++i) {
            if (fabs(cpuImgGrid[i].real() - accImgGrid[i].real()) * psfScale > thresh) {
                cout << endl;
                cout << "Fail (Expected " << cpuImgGrid[i].real() << " got "
                         << accImgGrid[i].real() << " at index " << i << ")"
                         << endl;
                return 1;
            }
        }

        //-------------------------------------------------------------------//
        // degridding results
        cout << " degrid";

        if (cpuModel.size() != accModel.size()) {
            cout << endl;
            cout << "Fail (Data vector sizes differ)" << std::endl;
            return 1;
        }

        for (unsigned int i = 0; i < cpuModel.size(); ++i) {
            if (fabs(cpuModel[i].real() - accModel[i].real()) * psfScale > thresh) {
                cout << endl;
                cout << "Fail (Expected " << cpuModel[i].real() << " got "
                         << accModel[i].real() << " at index " << i << ")"
                         << std::endl;
                return 1;
            }
        }

        cout << ": pass" << endl;

#endif

        ///////////////////////////////////////////////////////////////////////
        // Update data for next major cycle
        ///////////////////////////////////////////////////////////////////////

        //-------------------------------------------------------------------//
        // update visibility data 

#ifdef RUN_CPU
        // subtract the model vis and cycle back
        for (unsigned int i = 0; i < nSamples*nChan; ++i) {
            cpuData[i] = cpuData[i] - cpuModel[i];
        }
#endif

        #pragma acc parallel loop present(accData_d[0:nSamples*nChan],accModel_d[0:nSamples*nChan])
        for (unsigned int i = 0; i < nSamples*nChan; ++i) {
            accData_d[i] = accData_d[i] - accModel_d[i];
        }

    } // it_major

    ///////////////////////////////////////////////////////////////////////////
    // Timing results
    ///////////////////////////////////////////////////////////////////////////

    double time;

#ifdef RUN_CPU
    cout << endl << "+++++ CPU single core times +++++" << endl << endl;
    time = psfCpuTimer; // Only done once, during the first major cycle
    cout << "Gridding PSF" << endl;
    cout << "    Time per major cycle " << time << " (s) " << endl;
    cout << "    Time per visibility sample " << 1e6*time / double(cpuData.size()) << " (us) " << endl;
    cout << "    Time per gridding   " << 1e9*time / double(cpuData.size()*sSize*sSize) << " (ns) " << endl;
    cout << "    Gridding rate   " << griddings/1e6/time << " (million grid points per second)" << endl;
    time = imgCpuTimer/double(nMajor);
    cout << "Gridding data" << endl;
    cout << "    Time per major cycle " << time << " (s) " << endl;
    cout << "    Time per visibility sample " << 1e6*time / double(cpuData.size()) << " (us) " << endl;
    cout << "    Time per gridding   " << 1e9*time / double(cpuData.size()*sSize*sSize) << " (ns) " << endl;
    cout << "    Gridding rate   " << griddings/1e6/time << " (million grid points per second)" << endl;
    time = ifftCpuTimer/double(nMajor);
    cout << "Inverse FFTs" << endl;
    cout << "    Time per major cycle " << time << " (s) " << endl;
    time = HogbomCpuTimer/double(nMajor);
    cout << "Hogbom clean" << endl;
    cout << "    Time per major cycle " << time << " (s) " << endl;
    cout << "    Time per minor cycle " << time / nMinor * 1000 << " (ms)" << endl;
    cout << "    Cleaning rate  " << nMinor / time << " (iterations per second)" << endl;
    time = fftCpuTimer/double(nMajor);
    cout << "Forward FFT" << endl;
    cout << "    Time per major cycle " << time << " (s) " << endl;
    time = degridCpuTimer/double(nMajor);
    cout << "Degridding data" << endl;
    cout << "    Time per major cycle " << time << " (s) " << endl;
    cout << "    Time per visibility sample " << 1e6*time / double(cpuData.size()) << " (us) " << endl;
    cout << "    Time per degridding   " << 1e9*time / double(cpuData.size()*sSize*sSize) << " (ns) " << endl;
    cout << "    Degridding rate   " << griddings/1e6/time << " (million grid points per second)" << endl;
#endif
    cout << endl << "+++++ OpenACC times +++++" << endl << endl;
    time = psfAccTimer; // Only done once, during the first major cycle
    cout << "Gridding PSF" << endl;
    cout << "    Time per major cycle " << time << " (s) " << endl;
    cout << "    Time per visibility sample " << 1e6*time / double(accData.size()) << " (us) " << endl;
    cout << "    Time per gridding   " << 1e9*time / double(accData.size()*sSize*sSize) << " (ns) " << endl;
    cout << "    Gridding rate   " << griddings/1e6/time << " (million grid points per second)" << endl;
    time = imgAccTimer/double(nMajor);
    cout << "Gridding data" << endl;
    cout << "    Time per major cycle " << time << " (s) " << endl;
    cout << "    Time per visibility sample " << 1e6*time / double(accData.size()) << " (us) " << endl;
    cout << "    Time per gridding   " << 1e9*time / double(accData.size()*sSize*sSize) << " (ns) " << endl;
    cout << "    Gridding rate   " << griddings/1e6/time << " (million grid points per second)" << endl;
    time = ifftAccTimer/double(nMajor);
    cout << "Inverse FFTs" << endl;
    cout << "    Time per major cycle " << time << " (s) " << endl;
    time = HogbomAccTimer/double(nMajor);
    cout << "Hogbom clean" << endl;
    cout << "    Time per major cycle " << time << " (s) " << endl;
    cout << "    Time per minor cycle " << time / nMinor * 1000 << " (ms)" << endl;
    cout << "    Cleaning rate  " << nMinor / time << " (iterations per second)" << endl;
    time = fftAccTimer/double(nMajor);
    cout << "Forward FFT" << endl;
    cout << "    Time per major cycle " << time << " (s) " << endl;
    time = degridAccTimer/double(nMajor);
    cout << "Degridding data" << endl;
    cout << "    Time per major cycle " << time << " (s) " << endl;
    cout << "    Time per visibility sample " << 1e6*time / double(accData.size()) << " (us) " << endl;
    cout << "    Time per degridding   " << 1e9*time / double(accData.size()*sSize*sSize) << " (ns) " << endl;
    cout << "    Degridding rate   " << griddings/1e6/time << " (million grid points per second)" << endl;

    cout << endl;

    //writeImage("dirty_cpu.img", cpulmPsf);
    //writeImage("psf_cpu.img", cpulmGrid);

    return 0;
}
