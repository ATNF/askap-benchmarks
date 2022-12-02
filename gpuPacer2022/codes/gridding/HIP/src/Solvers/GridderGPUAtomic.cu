#include "GridderGPUAtomic.h"

using std::cout;
using std::endl;
using std::vector;
using std::complex;

__global__
void devGridKernelAtomic(
        const Complex* data,
        const int support,
        const Complex* C,
        const int* cOffset,
        const int* iu,
        const int* iv,
        Complex* grid,
        const int GSIZE,
        const int dind)
{

    const int SSIZE = 2 * support + 1;
    assert(SSIZE == blockDim.x);

    const int bind = blockIdx.x;
    const int tind = threadIdx.x;

    const int dindLocal = dind + bind;

    int gind = iu[dindLocal] + GSIZE * iv[dindLocal] - support;
    int cind = cOffset[dindLocal];
    const Complex dataLocal = data[dindLocal];

    for (int row = 0; row < SSIZE; ++row)
    {

        if (tind < SSIZE)
        {
            //grid[gind + tind] = hipCfmaf(dataLocal, C[cind + tind], grid[gind + tind]);
            const Complex tmp = hipCmulf(dataLocal, C[cind + tind]);
            //grid[gind + tind] = hipCaddf(grid[gind + tind], tmp);
            atomicAdd(&grid[gind].x + 2 * tind, tmp.x);
            atomicAdd(&grid[gind].y + 2 * tind + 1, tmp.y);
        }

        gind += GSIZE;
        cind += SSIZE;

    }
}

template<typename T2>
void GridderGPUAtomic<T2>::deviceAllocations()
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
void GridderGPUAtomic<T2>::copyH2D()
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
GridderGPUAtomic<T2>::~GridderGPUAtomic()
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
void GridderGPUAtomic<T2>::gridder()
{
    cout << "\nGridding on GPU" << endl;
    deviceAllocations();
    copyH2D();

    // Kernel launch
    cout << "Kernel launch" << endl;
    const size_t DSIZE = data.size();
    typedef hipComplex Complex;

    const int SSIZE = 2 * support + 1;

    hipFuncSetCacheConfig(reinterpret_cast<const void*>(devGridKernelAtomic), hipFuncCachePreferL1);

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

        devGridKernelAtomic <<<gridSize, SSIZE>>> ((const Complex*)dData, support, (const Complex*)dC,
            dCOffset, dIU, dIV, (Complex*)dGrid, GSIZE, dind);

        gpuCheckErrors("hip kernel launch failure");
    }
    cout << "Used " << count << " kernel launches." << endl;

    hipMemcpy(grid.data(), dGrid, SIZE_GRID, hipMemcpyDeviceToHost);
    gpuCheckErrors("hipMemcpy D2H failure");
}

template void GridderGPUAtomic<std::complex<float>>::gridder();
template void GridderGPUAtomic<std::complex<double>>::gridder();
template void GridderGPUAtomic<std::complex<float>>::deviceAllocations();
template void GridderGPUAtomic<std::complex<double>>::deviceAllocations();
template void GridderGPUAtomic<std::complex<float>>::copyH2D();
template void GridderGPUAtomic<std::complex<double>>::copyH2D();
template GridderGPUAtomic<std::complex<float>>::~GridderGPUAtomic();
template GridderGPUAtomic<std::complex<double>>::~GridderGPUAtomic();