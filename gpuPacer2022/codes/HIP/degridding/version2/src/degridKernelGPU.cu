#include "hip/hip_runtime.h"
#include "degridKernelGPU.h"


#define FULL_MASK 0xffffffff
#define WARPSIZE 32

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
    int dindLocal = dind + blockIdx.x;
    int gindStart = iu[dindLocal] + GSIZE*iv[dindLocal] - support;
    int cindStart = cOffset[dindLocal];
    int SSIZE = 2*support+1;
    int suppu = threadIdx.x;

    Complex dOrig = data[dindLocal];
    // suppv loop
    for (int suppv = 0; suppv < SSIZE; ++suppv)
    {
      int gind = gindStart + GSIZE * suppv;
      int cind = cindStart + SSIZE * suppv;
      Complex sum = hipCmulf(grid[gind + suppu], C[cind + suppu]);

      __syncthreads();
      // Reduce within each warp
      if (suppu < SSIZE)
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

      // Gather warp sums into shared memory
      const int NUMWARPS = (2*support+1) / WARPSIZE + 1;
      __shared__ Complex dataShared[NUMWARPS];

      int warp = suppu / WARPSIZE;
      int lane = threadIdx.x & (WARPSIZE-1); // the lead thread in the warp

      if (lane == 0)
      {
          dataShared[warp] = sum;
      }

      __syncthreads();
      // combine warp sums using a single thread in this block
      if (suppu == 0)
      {
        for (int w = 1; w < NUMWARPS; w++)
        {
            sum = hipCaddf(sum, dataShared[w]);
        }

        dOrig = hipCaddf(dOrig, sum);
      }
    }
    if (suppu == 0)
    {
      data[dindLocal] = dOrig;
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
