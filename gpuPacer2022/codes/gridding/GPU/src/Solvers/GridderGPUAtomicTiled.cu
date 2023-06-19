#include "GridderGPUAtomicTiled.h"

using std::cout;
using std::endl;
using std::vector;
using std::complex;

__global__
void devGridKernelAtomicTiled(
    const Complex* data,
    const int support,
    const Complex* C,
    const int* cOffset,
    const int* iu,
    const int* iv,
    Complex* grid,
    const int GSIZE,
    const int i)
{
    const int SSIZE = 2 * support + 1;

    const int tID = threadIdx.x;
    const int dind = i + blockIdx.x * blockDim.x + threadIdx.x;

    // The actual starting grid point
    int gind = iu[dind] + GSIZE * iv[dind] - support;
    // The Convolution function point from which we offset
    int cind = cOffset[dind];

    Complex dataLocal = data[dind];

    __shared__ int suppU;
    __shared__ int suppV;

    if (tID == 0)
    {
        suppU = blockIdx.y;
        suppV = blockIdx.z;
    }
    __syncthreads();

    // blockIdx.z gives the support location in the v direction
    gind += GSIZE * suppV;
    cind += SSIZE * suppV;


    //Complex gLocal = gpuCfmaf(dataLocal, C[cind + suppU], grid[gind + suppU]);
    //grid[gind + suppU] = gpuCfmaf(dataLocal, C[cind + suppU], grid[gind + suppU]);
    //atomicAdd(&grid[gind + suppU].x, gLocal.x);
    //atomicAdd(&grid[gind + suppU].y, gLocal.y);
    //grid[gind + suppU].x += gLocal.x;
    atomicAdd(&grid[gind + suppU].x, dataLocal.x * C[cind + suppU].x - dataLocal.y * C[cind + suppU].y);
    atomicAdd(&grid[gind + suppU].y, dataLocal.x * C[cind + suppU].y + dataLocal.y * C[cind + suppU].x);

    //grid[gind + suppU] = gpuCfmaf(dataLocal, C[cind + suppU], grid[gind + suppU]);
}

template<typename T2>
void GridderGPUAtomicTiled<T2>::deviceAllocations()
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
void GridderGPUAtomicTiled<T2>::copyH2D()
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
GridderGPUAtomicTiled<T2>::~GridderGPUAtomicTiled()
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
void GridderGPUAtomicTiled<T2>::gridder()
{
    cout << "\nGridding on GPU" << endl;
    deviceAllocations();
    copyH2D();

    const int BLOCK_SIZE = 1024;
    const int GRID_SIZE_Y = 129;
    const int GRID_SIZE_Z = 129;
    const int GRID_SIZE_X = NSAMPLES / BLOCK_SIZE;

    // Kernel launch
    cout << "Kernel launch" << endl;
    const size_t DSIZE = this->data.size();

    gpuFuncSetCacheConfig(reinterpret_cast<const void*>(devGridKernelAtomicTiled), gpuFuncCachePreferL1);

    dim3 gridSize(GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);
    int stepSize = GRID_SIZE_X * BLOCK_SIZE;

    int count = 0;
    for (int dind = 0; dind < DSIZE; dind += stepSize)
    {

        ++count;

        devGridKernelAtomicTiled <<< gridSize, BLOCK_SIZE >>> ((const Complex*)dData, this->support, (const Complex*)dC,
            dCOffset, dIU, dIV, (Complex*)dGrid, GSIZE, dind);

        gpuCheckErrors("gpu kernel launch failure");
    }
    cout << "Used " << count << " kernel launches." << endl;

    gpuMemcpy(this->grid.data(), dGrid, SIZE_GRID, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template void GridderGPUAtomicTiled<std::complex<float>>::gridder();
template void GridderGPUAtomicTiled<std::complex<double>>::gridder();
template void GridderGPUAtomicTiled<std::complex<float>>::deviceAllocations();
template void GridderGPUAtomicTiled<std::complex<double>>::deviceAllocations();
template void GridderGPUAtomicTiled<std::complex<float>>::copyH2D();
template void GridderGPUAtomicTiled<std::complex<double>>::copyH2D();
template GridderGPUAtomicTiled<std::complex<float>>::~GridderGPUAtomicTiled();
template GridderGPUAtomicTiled<std::complex<double>>::~GridderGPUAtomicTiled();
