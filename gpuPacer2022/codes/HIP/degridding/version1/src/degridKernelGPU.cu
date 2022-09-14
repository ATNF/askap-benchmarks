#include <hip/hip_runtime.h>
#include "degridKernelGPU.h"

#define FULL_MASK 0xffffffff

#ifdef __NVCC__
#define WARPSIZE 32
#else
#define WARPSIZE 64
#endif

// launch_bounds__(2*support+1, 8)
template <int support>
__global__
void devDegridKernel(
    const Complex* grid,
    const int GSIZE,
    const Complex* C,
    const int* cOffset,
    const int* iu,
    const int* iv,
    Complex* data,
    const int dind)
{

    const int dindLocal = dind + blockIdx.x;

    // The actual starting grid point
    __shared__ int gindShared;

    // The Convolution function point from which we offset
    __shared__ int cindShared;

    if (threadIdx.x == 0)
    {
        gindShared = iu[dindLocal] + GSIZE * iv[dindLocal] - support;
        cindShared = cOffset[dindLocal];
    }
    __syncthreads();

    Complex original = data[dindLocal];

    const int SSIZE = 2 * support + 1;

//#pragma unroll 8
    // row gives the support location in the v-direction
    for (int row = 0; row < SSIZE; ++row)
    {
        // Make a local copy from shared memory
        int gind = gindShared + GSIZE * row;
        int cind = cindShared + SSIZE * row;

        Complex sum = hipCmulf(grid[gind + threadIdx.x], C[cind + threadIdx.x]);

        // compute warp sums
        int i = threadIdx.x;
        if (i < SSIZE)
        {
          for (int offset = WARPSIZE/2; offset > 0; offset /=2 )
          {
#ifdef __NVCC__		  
            sum.x += __shfl_down_sync(FULL_MASK,sum.x,offset,WARPSIZE);
            sum.y += __shfl_down_sync(FULL_MASK,sum.y,offset,WARPSIZE);
#else
            sum.x += __shfl_down(sum.x,offset,WARPSIZE);
            sum.y += __shfl_down(sum.y,offset,WARPSIZE);
#endif	    
          }
          
        }

        const int NUMWARPS = (2 * support) / WARPSIZE;
        __shared__ Complex dataShared[NUMWARPS + 1];

        int warp = i / WARPSIZE;
        int lane = threadIdx.x & (WARPSIZE-1);

        if (lane == 0)
        {
            dataShared[warp] = sum;
        }

        __syncthreads();

        // combine warp sums 
        if (i == 0)
        {
//#pragma unroll
            for (int w = 1; w < NUMWARPS + 1; w++)
            {
                sum = hipCaddf(sum, dataShared[w]);
            }

            original = hipCaddf(original, sum);
        }
    }
    if (threadIdx.x == 0)
    {
        data[dindLocal] = original;
    }
}

template
__global__
void devDegridKernel<64>(
    const Complex* grid,
    const int GSIZE,
    const Complex* C,
    const int* cOffset,
    const int* iu,
    const int* iv,
    Complex* data,
    const int dind);

//template
//__device__
//Complex sumReduceWarpComplex<64>(Complex val);
