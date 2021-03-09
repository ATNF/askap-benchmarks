#include "hip/hip_runtime.h"
#include "hip/hip_complex.h"
// @copyright (c) 2009 CSIRO
// Australia Telescope National Facility (ATNF)
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// PO Box 76, Epping NSW 1710, Australia
// atnf-enquiries@csiro.au
//
// This file is part of the ASKAP software distribution.
//
// The ASKAP software distribution is free software: you can redistribute it
// and/or modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the License,
// or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
//
// @author Ben Humphreys <ben.humphreys@csiro.au>
// @author Tim Cornwell  <tim.cornwell@csiro.au>
//
// Acknowledgement:
// Mark Harris of NVIDIA contributed the below version of the degridding code.
// This was a significant speedup from the original version.

// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Local includes
#include "CudaGridKernel.h"

// Check and report last error
__host__ __inline__ void checkError(void)
{
        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
                printf("CUDA Error: %s\n", hipGetErrorString(err));
        }
}

// Perform Gridding (Device Function)
// Each thread handles a different grid point
__global__ void d_gridKernel(const Complex *data, const int support,
		const Complex *C, const int *cOffset,
		const int *iu, const int *iv,
		Complex *grid, const int gSize, const int dind)
{
	// The actual starting grid point
	__shared__ int s_gind;
	// The Convoluton function point from which we offset
	__shared__ int s_cind;

	// Calculate the data index offset for this block
	const int l_dind = dind + blockIdx.y;

	// A copy of the vis data so all threads can read it from shared
	// memory rather than all reading from device memory.
	__shared__ Complex l_data;

	if (threadIdx.x == 0) {
		s_gind = iu[l_dind] + gSize * iv[l_dind] - support;
		s_cind = cOffset[l_dind];
		l_data = data[l_dind];
	}
	__syncthreads();

	// Make a local copy from shared memory
	int gind = s_gind;
	int cind = s_cind;

	// blockIdx.x gives the support location in the v direction
	int sSize = 2 * support + 1;
    gind += gSize * blockIdx.x;
    cind += sSize * blockIdx.x;

    // threadIdx.x gives the support location in the u dirction
    grid[gind+threadIdx.x] = hipCfmaf(l_data, C[cind+threadIdx.x], grid[gind+threadIdx.x]);
}

// Calculates, for a given dind, how many of the next samples can
// be gridded without an overlap occuring
__host__ __inline__ int gridStep(const int *h_iu, const int *h_iv,
        const int dSize, const int dind, const int sSize)
{
    const int maxsamples = 32;  // Maximum number of samples to grid
    for (int step = 1; step <= maxsamples; step++) {
        for (int check = (step - 1); check >= 0; check--) {
            if (!((dind+step) < dSize && (
                            abs(h_iu[dind+step] - h_iu[dind+check]) > sSize ||
                            abs(h_iv[dind+step] - h_iv[dind+check]) > sSize))) {
                return step;
            }
        }
    }

    return maxsamples;
}

// Perform Gridding (Host Function)
__host__ void cuda_gridKernel(const Complex  *data, const int dSize, const int support,
		const Complex *C, const int *cOffset,
		const int *iu, const int *iv,
		Complex *grid, const int gSize,
		const int *h_iu, const int *h_iv)
{
    hipFuncSetCacheConfig(reinterpret_cast<const void*>(d_gridKernel), hipFuncCachePreferL1);

	const int sSize=2*support+1;
	int step = 1;

	// This loop begs some explanation. It steps through each spectral
	// sample either one at a time or two at a time. It will do multiple
	// samples if the regions involved do not overlap. If they do, only the
    // non-overlapping samples are gridded.
	//
	// Gridding multiple points is better because giving the GPU more
    // work to do allows it to hide memory latency better. The call to
    // d_gridKernel() is asynchronous so subsequent calls to gridStep()
    // overlap with the actual gridding.
    int count = 0;
    for (int dind = 0; dind < dSize; dind += step) {
        step = gridStep(h_iu, h_iv, dSize, dind, sSize);
        dim3 gridDim(sSize, step);
        hipLaunchKernelGGL(d_gridKernel, dim3(gridDim), dim3(sSize ), 0, 0, data, support,
                C, cOffset, iu, iv, grid, gSize, dind);
        checkError();
        count++;
    }
    printf("    Used %d kernel launches\n", count);
}

template <int support>
__device__ Complex sumReduceWarpComplex(Complex val)
{
    const int offset = 2*support;
    volatile __shared__ float vals[offset*2];

    int i = threadIdx.x;
    int lane = i & 31;
    vals[i]           = val.x;
    vals[i+offset] = val.y;

    float v = val.x;
    if (lane >= 16)
    {
        i += offset;
        v = val.y;
    }

    vals[i] = v = v + vals[i + 16];
    vals[i] = v = v + vals[i +  8];
    vals[i] = v = v + vals[i +  4];
    vals[i] = v = v + vals[i +  2];
    vals[i] = v = v + vals[i +  1];

    return make_hipFloatComplex(vals[threadIdx.x], vals[threadIdx.x + offset]);
}

// Perform De-Gridding (Device Function)
template <int support>
//__launch_bounds__(2*support+1, 8)
__global__ void d_degridKernel(const Complex *grid, const int gSize,
                const Complex *C, const int *cOffset,
                const int *iu, const int *iv,
                Complex  *data, const int dind)
{
    const int l_dind = dind + blockIdx.x;

    // The actual starting grid point
    __shared__ int s_gind;
    // The Convoluton function point from which we offset
    __shared__ int s_cind;

    if (threadIdx.x == 0) {
        s_gind = iu[l_dind] + gSize * iv[l_dind] - support;
        s_cind = cOffset[l_dind];
    }
    __syncthreads();

    Complex original = data[l_dind];

    const int sSize = 2 * support + 1;

    //#pragma unroll 
    // row gives the support location in the v direction
    for (int row = 0; row < sSize; ++row)
    {
        // Make a local copy from shared memory
        int gind = s_gind + gSize * row;
        int cind = s_cind + sSize * row;

        Complex sum = hipCmulf(grid[gind+threadIdx.x], C[cind+threadIdx.x]);

        // compute warp sums
        int i = threadIdx.x;
        if (i < sSize -1)
            sum = sumReduceWarpComplex<support>(sum);

        const int numWarps = (2 * support) / 32;
        __shared__ Complex s_data[numWarps + 1];

        int warp = i / 32;
        int lane = threadIdx.x & 31;

        if (lane == 0)
            s_data[warp] = sum;

        __syncthreads();

        // combine warp sums 
        if (i == 0)
        {
            //#pragma unroll
            for (int w = 1; w < numWarps+1; w++)
                sum = hipCaddf(sum, s_data[w]);

            original = hipCaddf(original, sum);
        }
    }

    if (threadIdx.x == 0)
        data[l_dind] = original;
}

// Perform De-Gridding (Host Function)
__host__ void cuda_degridKernel(const Complex *grid, const int gSize, const int support,
        const Complex *C, const int *cOffset,
        const int *iu, const int *iv,
        Complex  *data, const int dSize)
{
    const int sSize = 2 * support + 1;

    // Compute grid dimensions based on number of multiprocessors to ensure 
    // as much balance as possible
    int device;
    hipGetDevice(&device);
    hipDeviceProp_t devprop;
    hipGetDeviceProperties(&devprop, device);

    int count = 0;
    int dimGrid = 1024 * devprop.multiProcessorCount; // is starting size, will be reduced as required
    for (int dind = 0; dind < dSize; dind += dimGrid) {
        if ((dSize - dind) < dimGrid) {
            // If there are less than dimGrid elements left,
            // just do the remaining
            dimGrid = dSize - dind;
        }

        count++;
        switch (support)
        {
            case 64:
                hipLaunchKernelGGL(HIP_KERNEL_NAME(d_degridKernel<64>), dim3(dimGrid), dim3(sSize ), 0, 0, grid, gSize,
                        C, cOffset, iu, iv, data, dind);
                break;
            default:
                assert(0);
        }
        checkError();
    }
    printf("    Used %d kernel launches\n", count);
}
