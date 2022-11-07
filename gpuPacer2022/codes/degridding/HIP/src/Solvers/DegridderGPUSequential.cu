#include "DegridderGPUSequential.h"

using std::cout;
using std::endl;
using std::vector;
using std::complex;

#define MAX_SSIZE 256

__global__
void devDegridKernelSequential(
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
            const Complex cpix = hipCmulf(grid[gind + tind], C[cind + tind]);
            sdata_re[tind] = cpix.x;
            sdata_im[tind] = cpix.y;
            __syncthreads();

            for (unsigned int s = SSIZE / 2; s > 0; s /= 2)
            {
                if ((tind < s) && (tind + s < SSIZE)) 
                {
                    //sdata[tind] = hipCaddf(sdata[tind], sdata[tind + s]);
                    sdata_re[tind] += sdata_re[tind + s];
                    sdata_im[tind] += sdata_im[tind + s];
                }
                __syncthreads();
            }
            // because SSIZE is odd, reduction #3 misses the last thread
            if (tind == 0)
            {
                sdata_re[tind] += sdata_re[SSIZE - 1];
                sdata_im[tind] += sdata_im[SSIZE - 1];
            }
            __syncthreads();

        }

        if (tind == 0)
        {
            original = hipCaddf(original, make_hipComplex(sdata_re[tind], sdata_im[tind]));
        }
    }

    if (tind == 0)
    {
        data[dindLocal] = original;
    }

}

template<typename T2>
void DegridderGPUSequential<T2>::deviceAllocations()
{
    // Allocate device vectors
    hipMalloc(&dData, SIZE_DATA);
    hipMalloc(&dGrid, SIZE_GRID);
    hipMalloc(&dC, SIZE_C);
    hipMalloc(&dCOffset, SIZE_COFFSET);
    hipMalloc(&dIU, SIZE_IU);
    hipMalloc(&dIV, SIZE_IV);
    gpuCheckErrors("hipMalloc failure");
}

template<typename T2>
void DegridderGPUSequential<T2>::copyH2D()
{
    hipMemcpy(dData, data.data(), SIZE_DATA, hipMemcpyHostToDevice);
    hipMemcpy(dGrid, grid.data(), SIZE_GRID, hipMemcpyHostToDevice);
    hipMemcpy(dC, C.data(), SIZE_C, hipMemcpyHostToDevice);
    hipMemcpy(dCOffset, cOffset.data(), SIZE_COFFSET, hipMemcpyHostToDevice);
    hipMemcpy(dIU, iu.data(), SIZE_IU, hipMemcpyHostToDevice);
    hipMemcpy(dIV, iv.data(), SIZE_IV, hipMemcpyHostToDevice);
    gpuCheckErrors("hipMemcpy H2D failure");
}

template<typename T2>
DegridderGPUSequential<T2>::~DegridderGPUSequential()
{
    // Deallocate device vectors
    hipFree(dData);
    hipFree(dGrid);
    hipFree(dC);
    hipFree(dCOffset);
    hipFree(dIU);
    hipFree(dIV);
    gpuCheckErrors("hipFree failure");
}

template <typename T2>
void DegridderGPUSequential<T2>::degridder()
{
    deviceAllocations();
    copyH2D();

    // Kernel launch
    const size_t DSIZE = data.size();
    typedef hipComplex Complex;

    const int SSIZE = 2 * support + 1;

    // hipFuncSetCacheConfig(reinterpret_cast<const void*>(devGridKernelOlder), hipFuncCachePreferL1);

    int device;
    hipGetDevice(&device);
    hipDeviceProp devProp;
    hipGetDeviceProperties(&devProp, device);

    int gridSize = devProp.maxGridSize[0] / (support + 1);  // launch kernels for this number of samples at a time
    assert(SSIZE <= devProp.maxThreadsPerBlock);

    int count = 0;
    for (int dind = 0; dind < DSIZE; dind += gridSize)
    {
        // if there are less than dimGrid elements left, do the remaining
        if ((DSIZE - dind) < gridSize)
        {
            gridSize = DSIZE - dind;
        }

        ++count;

        devDegridKernelSequential << < gridSize, SSIZE >> > ((const Complex*)dGrid, GSIZE, (const Complex*)dC, support, dCOffset, dIU, dIV, (Complex*)dData, dind);

        gpuCheckErrors("hip kernel launch failure");
    }
    cout << "Used " << count << " kernel launches." << endl;

    hipMemcpy(data.data(), dData, SIZE_DATA, hipMemcpyDeviceToHost);
    gpuCheckErrors("hipMemcpy D2H failure");
}

template void DegridderGPUSequential<std::complex<float>>::degridder();
template void DegridderGPUSequential<std::complex<double>>::degridder();
template void DegridderGPUSequential<std::complex<float>>::deviceAllocations();
template void DegridderGPUSequential<std::complex<double>>::deviceAllocations();
template void DegridderGPUSequential<std::complex<float>>::copyH2D();
template void DegridderGPUSequential<std::complex<double>>::copyH2D();
template DegridderGPUSequential<std::complex<float>>::~DegridderGPUSequential();
template DegridderGPUSequential<std::complex<double>>::~DegridderGPUSequential();
