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


    //Complex gLocal = cuCfmaf(dataLocal, C[cind + suppU], grid[gind + suppU]);
    //grid[gind + suppU] = cuCfmaf(dataLocal, C[cind + suppU], grid[gind + suppU]);
    //atomicAdd(&grid[gind + suppU].x, gLocal.x);
    //atomicAdd(&grid[gind + suppU].y, gLocal.y);
    //grid[gind + suppU].x += gLocal.x;
    atomicAdd(&grid[gind + suppU].x, dataLocal.x * C[cind + suppU].x - dataLocal.y * C[cind + suppU].y);
    atomicAdd(&grid[gind + suppU].y, dataLocal.x * C[cind + suppU].y + dataLocal.y * C[cind + suppU].x);

    //grid[gind + suppU] = cuCfmaf(dataLocal, C[cind + suppU], grid[gind + suppU]);
}

template<typename T2>
void GridderGPUAtomicTiled<T2>::deviceAllocations()
{
    // Allocate device vectors
    cudaMalloc(&dData, SIZE_DATA);
    cudaMalloc(&dGrid, SIZE_GRID);
    cudaMalloc(&dC, SIZE_C);
    cudaMalloc(&dCOffset, SIZE_COFFSET);
    cudaMalloc(&dIU, SIZE_IU);
    cudaMalloc(&dIV, SIZE_IV);
    cudaCheckErrors("cudaMalloc failure");
}

template<typename T2>
void GridderGPUAtomicTiled<T2>::copyH2D()
{
    cudaMemcpy(dData, data.data(), SIZE_DATA, cudaMemcpyHostToDevice);
    cudaMemcpy(dGrid, grid.data(), SIZE_GRID, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), SIZE_C, cudaMemcpyHostToDevice);
    cudaMemcpy(dCOffset, cOffset.data(), SIZE_COFFSET, cudaMemcpyHostToDevice);
    cudaMemcpy(dIU, iu.data(), SIZE_IU, cudaMemcpyHostToDevice);
    cudaMemcpy(dIV, iv.data(), SIZE_IV, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");
}

template<typename T2>
GridderGPUAtomicTiled<T2>::~GridderGPUAtomic()
{
    // Deallocate device vectors
    cudaFree(dData);
    cudaFree(dGrid);
    cudaFree(dC);
    cudaFree(dCOffset);
    cudaFree(dIU);
    cudaFree(dIV);
    cudaCheckErrors("cudaFree failure");
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
    const size_t DSIZE = data.size();
    typedef cuComplex Complex;

    cudaFuncSetCacheConfig(reinterpret_cast<const void*>(devGridKernelAtomicTiled), cudaFuncCachePreferL1);

    dim3 gridSize(GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);
    int stepSize = GRID_SIZE_X * BLOCK_SIZE;

    int count = 0;
    for (int dind = 0; dind < DSIZE; dind += stepSize)
    {

        ++count;

        devGridKernelAtomicTiled << < gridSize, BLOCK_SIZE >> > ((const Complex*)dData, support, (const Complex*)dC,
            dCOffset, dIU, dIV, (Complex*)dGrid, GSIZE, dind);

        cudaCheckErrors("cuda kernel launch failure");
    }
    cout << "Used " << count << " kernel launches." << endl;

    cudaMemcpy(grid.data(), dGrid, SIZE_GRID, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");
}

template void GridderGPUAtomicTiled<std::complex<float>>::gridder();
template void GridderGPUAtomicTiled<std::complex<double>>::gridder();
template void GridderGPUAtomicTiled<std::complex<float>>::deviceAllocations();
template void GridderGPUAtomicTiled<std::complex<double>>::deviceAllocations();
template void GridderGPUAtomicTiled<std::complex<float>>::copyH2D();
template void GridderGPUAtomicTiled<std::complex<double>>::copyH2D();
template GridderGPUAtomicTiled<std::complex<float>>::~GridderGPUAtomicTiled();
template GridderGPUAtomicTiled<std::complex<double>>::~GridderGPUAtomicTiled();