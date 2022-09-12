#include "degridKernelGPU.h"

template <int support>
__device__ Complex sumReduceWarpComplex(Complex val)
{
    const int offset = 2 * support;
    volatile __shared__ float vals[offset * 2];

    int i = threadIdx.x;
    int lane = i & 31;
    vals[i] = val.x;
    vals[i + offset] = val.y;

    float v = val.x;
    if (lane >= 16)
    {
        i += offset;
        v = val.y;
    }

    vals[i] = v = v + vals[i + 16];
    vals[i] = v = v + vals[i + 8];
    vals[i] = v = v + vals[i + 4];
    vals[i] = v = v + vals[i + 2];
    vals[i] = v = v + vals[i + 1];

    return make_hipComplex(vals[threadIdx.x], vals[threadIdx.x + offset]);
}

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
    //printf("Hi there!\n");
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

#pragma unroll 8
    // row gives the support location in the v-direction
    for (int row = 0; row < SSIZE; ++row)
    {
        // Make a local copy from shared memory
        int gind = gindShared + GSIZE * row;
        int cind = cindShared + SSIZE * row;

        Complex sum = hipCmulf(grid[gind + threadIdx.x], C[cind + threadIdx.x]);

        // compute warp sums
        int i = threadIdx.x;
        if (i < SSIZE - 1)
        {
            sum = sumReduceWarpComplex<support>(sum);
        }

        const int NUMWARPS = (2 * support) / 32;
        __shared__ Complex dataShared[NUMWARPS + 1];

        int warp = i / 32;
        int lane = threadIdx.x & 31;

        if (lane == 0)
        {
            dataShared[warp] = sum;
        }

        __syncthreads();

        // combine warp sums 
        if (i == 0)
        {
#pragma unroll
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

template
__device__
Complex sumReduceWarpComplex<64>(Complex val);
