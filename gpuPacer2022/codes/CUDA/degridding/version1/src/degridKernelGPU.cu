#include "degridKernelGPU.cuh"
#include <cassert>

// need a max size for the static shared memory vectors. Note that there are two of them.
#define MAX_SSIZE 256

__global__
void devDegridKernel(
    const Complex* grid,
    const int GSIZE,
    const Complex* C,
    const int support,
    const int* cOffset,
    const int* iu,
    const int* iv,
    Complex* data,
    const int dind)
{

    const int bind = blockIdx.x;
    const int tind = threadIdx.x;

    const int dindLocal = dind + bind;

    const int SSIZE = 2 * support + 1;
    assert(SSIZE == blockDim.x);

    // The actual starting grid point
    __shared__ int gindShared;

    // The Convolution function point from which we offset
    __shared__ int cindShared;

    // Shared memory buffer for the conv pixels in this block (data point)
    __shared__ float sdata_re[MAX_SSIZE];
    __shared__ float sdata_im[MAX_SSIZE];

    if (tind == 0)
    {
        gindShared = iu[dindLocal] + GSIZE * iv[dindLocal] - support;
        cindShared = cOffset[dindLocal];
    }
    __syncthreads();

    Complex original = data[dindLocal];

    for (int row = 0; row < SSIZE; ++row)
    {
        // Make a local copy from shared memory
        int gind = gindShared + GSIZE * row;
        int cind = cindShared + SSIZE * row;

        if (tind < SSIZE)
        {
            const Complex cpix = cuCmulf(grid[gind + tind], C[cind + tind]);
            sdata_re[tind] = cpix.x;
            sdata_im[tind] = cpix.y;
            __syncthreads();

            // Reduction suggestions at https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

            // Note: This hasn't gone as far as separate warp reductions. The whole thing is reduced together.
            //       Performance may be improved if separate warp reductions are used and loops are unrolled.

            // reductions 3 and 4 only work if support is a power of 2
            // 2 and 3 seem to be the fastest from a small amount of testing. 1 is about twice as slow
            const int reduction = 3;

            switch (reduction)
            {
            case 0: // this is the original case, which has been deleted (had a bug and hasn't been redone yet)
                // assert(0);
                break;
            case 1: // Reduction #1
                for(unsigned int s=1; s < SSIZE; s *= 2)
                {
                    if ((tind % (2*s) == 0) && (tind + s < SSIZE))
                    {
                        //sdata[tind] = hipCaddf(sdata[tind], sdata[tind + s]);
                        sdata_re[tind] += sdata_re[tind + s];
                        sdata_im[tind] += sdata_im[tind + s];
                    }
                    __syncthreads();
                }
                break;
            case 2: // Reduction #2
                for(unsigned int s=1; s < SSIZE; s *= 2)
                {
                    int index = 2 * s * tind;
                    if (index + s < SSIZE) {
                        //sdata[tind] = hipCaddf(sdata[tind], sdata[tind + s]);
                        sdata_re[index] += sdata_re[index + s];
                        sdata_im[index] += sdata_im[index + s];
                    }
                    __syncthreads();
                }
                break;
            case 3: // Reduction #3
                for(unsigned int s=SSIZE/2; s > 0; s /= 2)
                {
                    if ((tind < s) && (tind + s < SSIZE)) {
                        //sdata[tind] = hipCaddf(sdata[tind], sdata[tind + s]);
                        sdata_re[tind] += sdata_re[tind + s];
                        sdata_im[tind] += sdata_im[tind + s];
                    }
                    __syncthreads();
                }
                // because SSIZE is odd, reduction #3 misses the last thread
                if (tind == 0)
                {
                    sdata_re[tind] += sdata_re[SSIZE-1];
                    sdata_im[tind] += sdata_im[SSIZE-1];
                }
                __syncthreads();
                break;
            case 4: // deal with imaginary parts if tind > s
                for(unsigned int s=SSIZE/2; s > 0; s /= 2)
                {
                    // reduce the real part with threads 0:SSIZE/2
                    if ((tind < s) && (tind + s < SSIZE)) {
                        sdata_re[tind] += sdata_re[tind + s];
                    }
                    // reduce the imaginary part with threads SSIZE/2:SSIZE
                    if ((tind > SSIZE-1-s) && (tind - s >= 0)) {
                        sdata_im[tind] += sdata_im[tind - s];
                    }
                    __syncthreads();
                }
                // because SSIZE is odd, the real accumulation ends in the first thread but misses the last thread
                // while the imaginary accumulation ends in the last thread but misses the first thread.
                // So add the last to the first before moving on
                if (tind == 0)
                {
                    sdata_re[tind] += sdata_re[SSIZE-1];
                    sdata_im[tind] += sdata_im[SSIZE-1];
                }
                __syncthreads();
                break;
            default:
                // assert(0);
                break;
            }

        }

        if (tind == 0)
        {
            original = cuCaddf(original, make_cuComplex(sdata_re[tind], sdata_im[tind]));
        }
    }

    if (tind == 0)
    {
        data[dindLocal] = original;
    }

}
