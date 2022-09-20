#include "degridKernelGPU.cuh"
#include <stdio.h>

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
    const int i)
{
    const int dind = i + blockIdx.x;

    int suppU = threadIdx.x;
    int suppV = threadIdx.y;
    int tID = blockDim.x * suppV + suppU;

    const int SSIZE = 2 * support + 1;

    __shared__ float sdata_re[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float sdata_im[BLOCK_SIZE * BLOCK_SIZE];
 
    sdata_re[tID] = 0.0;
    sdata_im[tID] = 0.0;

    int gindShared = iu[dind] + GSIZE * iv[dind] - support;
    int cindShared = cOffset[dind];

    // Block-stride loading
    while (suppV < SSIZE)
    {
        while (suppU < SSIZE)
        {
            int gind = gindShared + GSIZE * (suppV);
            int cind = cindShared + SSIZE * (suppV);

            // copy the local convolution product to shared memory
            sdata_re[tID] += grid[gind + suppU].x * C[cind + suppU].x - grid[gind + suppU].y * C[cind + suppU].y;
            sdata_im[tID] += grid[gind + suppU].x * C[cind + suppU].y + grid[gind + suppU].y * C[cind + suppU].x;
      
            suppU += blockDim.x;
        }
        
        suppU = threadIdx.x;
        suppV += blockDim.y;
        
    }
    
    for (unsigned int s = (BLOCK_SIZE * BLOCK_SIZE) / 2; s > 0; s >>= 1)
    {
        __syncthreads();
        if (tID < s)
        {
            sdata_re[tID] += sdata_re[tID + s];
            sdata_im[tID] += sdata_im[tID + s];
        }
        
    }
    

    if (tID == 0)
    {
        data[dind].x = sdata_re[tID];
        data[dind].y = sdata_im[tID];
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
