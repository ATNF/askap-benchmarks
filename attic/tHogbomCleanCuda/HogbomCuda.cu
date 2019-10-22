/// @copyright (c) 2011 CSIRO
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

// Include own header file first
#include "HogbomCuda.h"

// System includes
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstddef>
#include <stdio.h>

// Local includes
#include "Parameters.h"

using namespace std;

// Some constants for findPeak
const int findPeakNBlocks = 4;
const int findPeakWidth = 1024;

struct Peak {
    size_t pos;
    float val;
};

struct Position {
    __host__ __device__
    Position(int _x, int _y) : x(_x), y(_y) { };
    int x;
    int y;
};

__host__
static void checkerror(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

__host__ __device__
static Position idxToPos(const size_t idx, const int width)
{
    const int y = idx / width;
    const int x = idx % width;
    return Position(x, y);
}

__host__ __device__
static size_t posToIdx(const int width, const Position& pos)
{
    return (pos.y * width) + pos.x;
}

__global__
void d_findPeak(const float* image, size_t size, Peak* absPeak)
{
    __shared__ float maxVal[findPeakWidth];
    __shared__ size_t maxPos[findPeakWidth];
    const int column = threadIdx.x + (blockIdx.x * blockDim.x);
    maxVal[threadIdx.x] = 0.0;
    maxPos[threadIdx.x] = 0;

    for (int idx = column; idx < size; idx += 4096) {
        if (abs(image[idx]) > abs(maxVal[threadIdx.x])) {
            maxVal[threadIdx.x] = image[idx];
            maxPos[threadIdx.x] = idx;
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        absPeak[blockIdx.x].val = 0.0;
        absPeak[blockIdx.x].pos = 0;
        for (int i = 0; i < findPeakWidth; ++i) {
            if (abs(maxVal[i]) > abs(absPeak[blockIdx.x].val)) {
                absPeak[blockIdx.x].val = maxVal[i];
                absPeak[blockIdx.x].pos = maxPos[i];
            }
        }
    }
}

__host__
static Peak findPeak(const float* d_image, size_t size)
{
    const int nBlocks = findPeakNBlocks;
    Peak peaks[nBlocks];

    // Initialise a peaks array on the device. Each thread block will return
    // a peak. Note:  the d_peaks array is not initialized (hence avoiding the
    // memcpy), it is up to the device function to do that
    cudaError_t err;
    Peak* d_peak;
    err = cudaMalloc((void **) &d_peak, nBlocks * sizeof(Peak));
    checkerror(err);

    // Find peak
    d_findPeak<<<nBlocks, findPeakWidth>>>(d_image, size, d_peak);
    err = cudaGetLastError();
    checkerror(err);

    // Get the peaks array back from the device
    err = cudaMemcpy(&peaks, d_peak, nBlocks * sizeof(Peak), cudaMemcpyDeviceToHost);
    checkerror(err);
    err = cudaDeviceSynchronize();
    checkerror(err);
    cudaFree(d_peak);

    // Each thread block returned a peak, find the absolute maximum
    Peak p;
    p.val = 0;
    p.pos = 0;
    for (int i = 0; i < nBlocks; ++i) {
        if (abs(peaks[i].val) > abs(p.val)) {
            p.val = peaks[i].val;
            p.pos = peaks[i].pos;

        }
    }

    return p;
}

__global__
void d_subtractPSF(const float* d_psf,
    const int psfWidth,
    float* d_residual,
    const int residualWidth,
    const int startx, const int starty,
    int const stopx, const int stopy,
    const int diffx, const int diffy,
    const float absPeakVal, const float gain)
{   
    const int x =  startx + threadIdx.x + (blockIdx.x * blockDim.x);
    const int y =  starty + threadIdx.y + (blockIdx.y * blockDim.y);

    // Because thread blocks are of size 16, and the workload is not always
    // a multiple of 16, need to ensure only those threads whos responsibility
    // lies in the work area actually do work
    if (x <= stopx && y <= stopy) {
        d_residual[posToIdx(residualWidth, Position(x, y))] -= gain * absPeakVal
            * d_psf[posToIdx(psfWidth, Position(x - diffx, y - diffy))];
    }
}

__host__
static void subtractPSF(const float* d_psf, const int psfWidth,
        float* d_residual, const int residualWidth,
        const size_t peakPos, const size_t psfPeakPos,
        const float absPeakVal, const float gain)
{
    const int blockDim = 16;

    const int rx = idxToPos(peakPos, residualWidth).x;
    const int ry = idxToPos(peakPos, residualWidth).y;

    const int px = idxToPos(psfPeakPos, psfWidth).x;
    const int py = idxToPos(psfPeakPos, psfWidth).y;

    const int diffx = rx - px;
    const int diffy = ry - px;

    const int startx = max(0, rx - px);
    const int starty = max(0, ry - py);

    const int stopx = min(residualWidth - 1, rx + (psfWidth - px - 1));
    const int stopy = min(residualWidth - 1, ry + (psfWidth - py - 1));

    // Note: Both start* and stop* locations are inclusive.
    const int blocksx = ceil((stopx-startx+1.0) / static_cast<float>(blockDim));
    const int blocksy = ceil((stopy-starty+1.0) / static_cast<float>(blockDim));

    dim3 numBlocks(blocksx, blocksy);
    dim3 threadsPerBlock(blockDim, blockDim);
    d_subtractPSF<<<numBlocks,threadsPerBlock>>>(d_psf, psfWidth, d_residual, residualWidth,
            startx, starty, stopx, stopy, diffx, diffy, absPeakVal, gain);
    cudaError_t err = cudaGetLastError();
    checkerror(err);
}

__host__
HogbomCuda::HogbomCuda()
{
}

__host__
HogbomCuda::~HogbomCuda()
{
}

__host__
void HogbomCuda::deconvolve(const vector<float>& dirty,
        const size_t dirtyWidth,
        const vector<float>& psf,
        const size_t psfWidth,
        vector<float>& model,
        vector<float>& residual)
{
    reportDevice();
    residual = dirty;

    // Allocate device memory
    float* d_dirty;
    float* d_psf;
    float* d_residual;

    cudaError_t err;
    err = cudaMalloc((void **) &d_dirty, dirty.size() * sizeof(float));
    checkerror(err);
    err = cudaMalloc((void **) &d_psf, psf.size() * sizeof(float));
    checkerror(err);
    err = cudaMalloc((void **) &d_residual, residual.size() * sizeof(float));
    checkerror(err);

    // Copy host vectors to device arrays
    err = cudaMemcpy(d_dirty, &dirty[0], dirty.size() * sizeof(float), cudaMemcpyHostToDevice);
    checkerror(err);
    err = cudaMemcpy(d_psf, &psf[0], psf.size() * sizeof(float), cudaMemcpyHostToDevice);
    checkerror(err);
    err = cudaMemcpy(d_residual, &residual[0], residual.size() * sizeof(float), cudaMemcpyHostToDevice);
    checkerror(err);

    // Find peak of PSF
    Peak psfPeak = findPeak(d_psf, psf.size());

    cout << "Found peak of PSF: " << "Maximum = " << psfPeak.val 
        << " at location " << idxToPos(psfPeak.pos, psfWidth).x << ","
        << idxToPos(psfPeak.pos, psfWidth).y << endl;
    assert(psfPeak.pos <= psf.size());

    for (unsigned int i = 0; i < g_niters; ++i) {
        // Find peak in the residual image
        Peak peak = findPeak(d_residual, residual.size());

        assert(peak.pos <= residual.size());
        //cout << "Iteration: " << i + 1 << " - Maximum = " << peak.val
        //    << " at location " << idxToPos(peak.pos, dirtyWidth).x << ","
        //    << idxToPos(peak.pos, dirtyWidth).y << endl;


        // Check if threshold has been reached
        if (abs(peak.val) < g_threshold) {
            cout << "Reached stopping threshold" << endl;
            break;
        }

        // Subtract the PSF from the residual image (this function will launch
        // an kernel asynchronously, need to sync later
        subtractPSF(d_psf, psfWidth, d_residual, dirtyWidth, peak.pos, psfPeak.pos, peak.val, g_gain);

        // Add to model
        model[peak.pos] += peak.val * g_gain;

        // Wait for the PSF subtraction to finish
        err = cudaDeviceSynchronize();
        checkerror(err);
    }

    // Copy device arrays back into the host vector
    err = cudaMemcpy(&residual[0], d_residual, residual.size() * sizeof(float), cudaMemcpyDeviceToHost);
    checkerror(err);
    err = cudaDeviceSynchronize();
    checkerror(err);

    // Free device memory
    cudaFree(d_dirty);
    cudaFree(d_psf);
    cudaFree(d_residual);
}

__host__
void HogbomCuda::reportDevice(void)
{
    // Report the type of device being used
    int device;
    cudaDeviceProp devprop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devprop, device);
    std::cout << "    Using CUDA Device " << device << ": "
        << devprop.name << std::endl;
}
