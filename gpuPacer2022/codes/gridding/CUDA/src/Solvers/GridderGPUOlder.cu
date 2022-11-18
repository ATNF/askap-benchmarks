#include "GridderGPUOlder.h"

using std::cout;
using std::endl;
using std::vector;
using std::complex;

__global__
void devGridKernelOlder(
    const Complex * data,
    const int support,
    const Complex * C,
    const int* cOffset,
    const int* iu,
    const int* iv,
    Complex * grid,
    const int GSIZE,
    const int dind)
{
    // The actual starting grid point
    __shared__ int gindShared;

    // The Convolution function point from which we offset
    __shared__ int cindShared;

    // Calculate the data index offset for this block
    const int dindLocal = dind + blockIdx.y;

    // A copy of the visibility data
    // All threads can read it from shared memory
    // rather than all reading from device (global) memory
    __shared__ Complex dataLocal;

    if (threadIdx.x == 0)
    {
        gindShared = iu[dindLocal] + GSIZE * iv[dindLocal] - support;
        cindShared = cOffset[dindLocal];
        dataLocal = data[dindLocal];
    }

    __syncthreads();

    // Make a local copy from shared memory
    int gind = gindShared;
    int cind = cindShared;

    // blockIdx.x gives the support location in the v direction
    int SSIZE = 2 * support + 1;
    gind += GSIZE * blockIdx.x;
    cind += SSIZE * blockIdx.x;

    // threadIdx.x gives the support location in the u direction
    grid[gind + threadIdx.x] = cuCfmaf(dataLocal, C[cind + threadIdx.x], grid[gind + threadIdx.x]);
}

int GridderGPUOlder::gridStep(const int DSIZE, const int SSIZE, const int dind)
{
    const int MAXSAMPLES = 32;
    for (int step = 1; step <= MAXSAMPLES; ++step)
    {
        for (int check = (step - 1); check >= 0; --check)
        {
            if (!((dind + step) < DSIZE && (
                abs(iu[dind + step] - iu[dind + check]) > SSIZE ||
                abs(iv[dind + step] - iv[dind + check]) > SSIZE)))
            {
                return step;
            }
        }
    }
    return MAXSAMPLES;
}

void GridderGPUOlder::deviceAllocations()
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

void GridderGPUOlder::copyH2D()
{
    cudaMemcpy(dData, data.data(), SIZE_DATA, cudaMemcpyHostToDevice);
    cudaMemcpy(dGrid, grid.data(), SIZE_GRID, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), SIZE_C, cudaMemcpyHostToDevice);
    cudaMemcpy(dCOffset, cOffset.data(), SIZE_COFFSET, cudaMemcpyHostToDevice);
    cudaMemcpy(dIU, iu.data(), SIZE_IU, cudaMemcpyHostToDevice);
    cudaMemcpy(dIV, iv.data(), SIZE_IV, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");
}

GridderGPUOlder::~GridderGPUOlder()
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

void GridderGPUOlder::gridder()
{
    cout << "\nGridding on GPU" << endl;
    deviceAllocations();
    copyH2D();

    // Kernel launch
    cout << "Kernel launch" << endl;
    const size_t DSIZE = data.size();
    typedef cuComplex Complex;

    const int SSIZE = 2 * support + 1;

    cudaFuncSetCacheConfig(reinterpret_cast<const void*>(devGridKernelOlder), cudaFuncCachePreferL1);

    int step = 1;
    int count = 0;
    for (int dind = 0; dind < DSIZE; dind += step)
    {
        step = gridStep(DSIZE, SSIZE, dind);
        dim3 gridDim(SSIZE, step);
        /// PJE: make sure any chevron is tightly packed
        devGridKernelOlder <<<gridDim, SSIZE>>> ((const Complex*)dData, support, (const Complex*)dC,
            dCOffset, dIU, dIV, (Complex*)dGrid, GSIZE, dind);
        cudaCheckErrors("kernel launch (devGridKernel_v0) failure");
        count++;
    }
    cout << "Used " << count << " kernel launches." << endl;

    cudaMemcpy(grid.data(), dGrid, SIZE_GRID, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");
}
