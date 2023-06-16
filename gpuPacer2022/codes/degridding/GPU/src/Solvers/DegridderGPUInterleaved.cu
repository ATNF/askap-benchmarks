#include "DegridderGPUInterleaved.h"

using std::cout;
using std::endl;
using std::vector;
using std::complex;

#define MAX_SSIZE 256

__global__
void devDegridKernelInterleaved(
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
            const Complex cpix = gpuCmulf(grid[gind + tind], C[cind + tind]);
            sdata_re[tind] = cpix.x;
            sdata_im[tind] = cpix.y;
            __syncthreads();

            for (unsigned int s = 1; s < SSIZE; s *= 2)
            {
                int index = 2 * s * tind;
                if (index + s < SSIZE) 
                {
                    //sdata[tind] = gpuCaddf(sdata[tind], sdata[tind + s]);
                    sdata_re[index] += sdata_re[index + s];
                    sdata_im[index] += sdata_im[index + s];
                }
                __syncthreads();
            }

        }

        if (tind == 0)
        {
            original = gpuCaddf(original, make_gpuComplex(sdata_re[tind], sdata_im[tind]));
        }
    }

    if (tind == 0)
    {
        data[dindLocal] = original;
    }

}

template<typename T2>
void DegridderGPUInterleaved<T2>::deviceAllocations()
{
    // Allocate device vectors
    gpuMalloc(&dData, SIZE_DATA);
    gpuMalloc(&dGrid, SIZE_GRID);
    gpuMalloc(&dC, SIZE_C);
    gpuMalloc(&dCOffset, SIZE_COFFSET);
    gpuMalloc(&dIU, SIZE_IU);
    gpuMalloc(&dIV, SIZE_IV);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T2>
void DegridderGPUInterleaved<T2>::copyH2D()
{
    gpuMemcpy(dData, this->data.data(), SIZE_DATA, gpuMemcpyHostToDevice);
    gpuMemcpy(dGrid, this->grid.data(), SIZE_GRID, gpuMemcpyHostToDevice);
    gpuMemcpy(dC, this->C.data(), SIZE_C, gpuMemcpyHostToDevice);
    gpuMemcpy(dCOffset, this->cOffset.data(), SIZE_COFFSET, gpuMemcpyHostToDevice);
    gpuMemcpy(dIU, this->iu.data(), SIZE_IU, gpuMemcpyHostToDevice);
    gpuMemcpy(dIV, this->iv.data(), SIZE_IV, gpuMemcpyHostToDevice);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T2>
DegridderGPUInterleaved<T2>::~DegridderGPUInterleaved()
{
    // Deallocate device vectors
    gpuFree(dData);
    gpuFree(dGrid);
    gpuFree(dC);
    gpuFree(dCOffset);
    gpuFree(dIU);
    gpuFree(dIV);
    gpuCheckErrors("gpuFree failure");
}

template <typename T2>
void DegridderGPUInterleaved<T2>::degridder()
{
    deviceAllocations();
    copyH2D();

    // Kernel launch
    const size_t DSIZE = this->data.size();
    //typedef gpuComplex Complex;

    const int SSIZE = 2 * this->support + 1;

    // hipFuncSetCacheConfig(reinterpret_cast<const void*>(devGridKernelOlder), hipFuncCachePreferL1);

    int device;
    gpuGetDevice(&device);
    gpuDeviceProp_t devProp;
    gpuGetDeviceProperties(&devProp, device);

    int gridSize = devProp.maxGridSize[0] / (this->support + 1);  // launch kernels for this number of samples at a time
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

        devDegridKernelInterleaved <<<gridSize, SSIZE>>> ((const Complex*)dGrid, GSIZE, (const Complex*)dC, this->support, dCOffset, dIU, dIV, (Complex*)dData, dind);

        gpuCheckErrors("gpu kernel launch failure");
    }
    cout << "Used " << count << " kernel launches." << endl;

    gpuMemcpy(this->data.data(), dData, SIZE_DATA, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template void DegridderGPUInterleaved<std::complex<float>>::degridder();
template void DegridderGPUInterleaved<std::complex<double>>::degridder();
template void DegridderGPUInterleaved<std::complex<float>>::deviceAllocations();
template void DegridderGPUInterleaved<std::complex<double>>::deviceAllocations();
template void DegridderGPUInterleaved<std::complex<float>>::copyH2D();
template void DegridderGPUInterleaved<std::complex<double>>::copyH2D();
template DegridderGPUInterleaved<std::complex<float>>::~DegridderGPUInterleaved();
template DegridderGPUInterleaved<std::complex<double>>::~DegridderGPUInterleaved();