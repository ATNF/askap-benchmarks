/// @copyright (c) 2011 CSIRO
/// Australia Telescope National Facility (ATNF)
/// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
/// PO Box 76, Epping NSW 1710, Australia
/// atnf-enquiries@csiro.au
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

#include <cub.cuh>

using namespace std;

// Some constants for findPeak
int findPeakNBlocks = 26;
static const int findPeakWidth = 1024;

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

__host__ __device__ inline
static Position idxToPos(const size_t idx, const int width)
{
    const int y = idx / width;
    const int x = idx % width;
    return Position(x, y);
}

__host__ __device__ inline
static size_t posToIdx(const int width, const Position& pos)
{
    return (pos.y * width) + pos.x;
}

// For CUB
struct MaxOp
{
    __host__ __device__ inline
    Peak operator()(const Peak &a, const Peak &b)
    {
        return (abs(b.val) > abs(a.val)) ? b : a;
    }
};

__global__ 
void d_findPeak(Peak *peaks, const float* __restrict__ image, int size)
{
    Peak threadMax = {0.0, 0};   
        
    // parallel raking reduction (independent threads)
    for (int i = findPeakWidth * blockIdx.x + threadIdx.x; 
        i < size; 
        i += gridDim.x * findPeakWidth) {
        if (abs(image[i]) > abs(threadMax.val)) {
            threadMax.val = image[i];
            threadMax.pos = i;
        }
    }

    // Use CUB to find the max for each thread block.
    typedef cub::BlockReduce<Peak, findPeakWidth> BlockMax;
    __shared__ typename BlockMax::TempStorage temp_storage;
    threadMax = BlockMax(temp_storage).Reduce(threadMax, MaxOp());

    if (threadIdx.x == 0) peaks[blockIdx.x] = threadMax;
}

__host__
static Peak findPeak(Peak *d_peaks, const float* d_image, size_t size)
{
    // Find peak
    d_findPeak<<<findPeakNBlocks, findPeakWidth>>>(d_peaks, d_image, size);   
    
    // Get the peaks array back from the device
    Peak peaks[findPeakNBlocks];
    cudaError_t err = cudaMemcpy(peaks, d_peaks, findPeakNBlocks * sizeof(Peak), cudaMemcpyDeviceToHost);
    checkerror(err);
    
    Peak p = peaks[0];
    // serial final reduction
    for (int i = 1; i < findPeakNBlocks; ++i) {
        if (abs(peaks[i].val) > abs(p.val))
            p = peaks[i];
    }

    return p;
}

__global__
void d_subtractPSF(const float* __restrict__ d_psf,
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

    // Because workload is not always a multiple of thread block size, 
    // need to ensure only threads in the work area actually do work
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
    // The x,y coordinate of the peak in the residual image
    const int rx = idxToPos(peakPos, residualWidth).x;
    const int ry = idxToPos(peakPos, residualWidth).y;

    // The x,y coordinate for the peak of the PSF (usually the centre)
    const int px = idxToPos(psfPeakPos, psfWidth).x;
    const int py = idxToPos(psfPeakPos, psfWidth).y;

    // The PSF needs to be overlayed on the residual image at the position
    // where the peaks align. This is the offset between the above two points
    const int diffx = rx - px;
    const int diffy = ry - py;

    // The top-left-corner of the region of the residual to subtract from.
    // This will either be the top right corner of the PSF too, or on an edge
    // in the case the PSF spills outside of the residual image
    const int startx = max(0, rx - px);
    const int starty = max(0, ry - py);

    // This is the bottom-right corner of the region of the residual to
    // subtract from.
    const int stopx = min(residualWidth - 1, rx + (psfWidth - px - 1));
    const int stopy = min(residualWidth - 1, ry + (psfWidth - py - 1));

    const dim3 blockDim(32, 4);

    // Note: Both start* and stop* locations are inclusive.
    const int blocksx = ceil((stopx-startx+1.0f) / static_cast<float>(blockDim.x));
    const int blocksy = ceil((stopy-starty+1.0f) / static_cast<float>(blockDim.y));

    dim3 gridDim(blocksx, blocksy);

    d_subtractPSF<<<gridDim, blockDim>>>(d_psf, psfWidth, d_residual, residualWidth,
        startx, starty, stopx, stopy, diffx, diffy, absPeakVal, gain);
    cudaError_t err = cudaGetLastError();
    checkerror(err);
}

__host__
HogbomCuda::HogbomCuda(size_t psfSize, size_t residualSize)
{
    reportDevice();

    cudaError_t err;
    err = cudaMalloc((void **) &d_psf, psfSize * sizeof(float));
    checkerror(err);
    err = cudaMalloc((void **) &d_residual, residualSize * sizeof(float));
    checkerror(err);
    err = cudaMalloc((void **) &d_peaks, findPeakNBlocks * sizeof(Peak));
    checkerror(err);
}

__host__
HogbomCuda::~HogbomCuda()
{
    // Free device memory
    cudaFree(d_psf);
    cudaFree(d_residual);
    cudaFree(d_peaks);
}

__host__
void HogbomCuda::deconvolve(const vector<float>& dirty,
        const size_t dirtyWidth,
        const vector<float>& psf,
        const size_t psfWidth,
        vector<float>& model,
        vector<float>& residual)
{
    cudaError_t err;

    // Copy host vectors to device arrays
    err = cudaMemcpy(d_psf, &psf[0], psf.size() * sizeof(float), cudaMemcpyHostToDevice);
    checkerror(err);
    err = cudaMemcpy(d_residual, &dirty[0], residual.size() * sizeof(float), cudaMemcpyHostToDevice);
    checkerror(err);

    // Find peak of PSF
    Peak psfPeak = findPeak(d_peaks, d_psf, psf.size());

    cout << "Found peak of PSF: " << "Maximum = " << psfPeak.val 
        << " at location " << idxToPos(psfPeak.pos, psfWidth).x << ","
        << idxToPos(psfPeak.pos, psfWidth).y << endl;
    assert(psfPeak.pos <= psf.size());

    for (unsigned int i = 0; i < g_niters; ++i) {
        // Find peak in the residual image
        Peak peak = findPeak(d_peaks, d_residual, residual.size());

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
    }

    // Copy device array back into the host vector
    err = cudaMemcpy(&residual[0], d_residual, residual.size() * sizeof(float), cudaMemcpyDeviceToHost);
    checkerror(err);
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

    // Allocate 2 blocks per multiprocessor
    findPeakNBlocks = 2 * devprop.multiProcessorCount;
}
