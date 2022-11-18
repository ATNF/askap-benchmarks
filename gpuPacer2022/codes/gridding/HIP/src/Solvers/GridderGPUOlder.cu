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
    hipMalloc(&dData, SIZE_DATA);
    hipMalloc(&dGrid, SIZE_GRID);
    hipMalloc(&dC, SIZE_C);
    hipMalloc(&dCOffset, SIZE_COFFSET);
    hipMalloc(&dIU, SIZE_IU);
    hipMalloc(&dIV, SIZE_IV);
    gpuCheckErrors("hipMalloc failure");
}

void GridderGPUOlder::copyH2D()
{
    hipMemcpy(dData, data.data(), SIZE_DATA, hipMemcpyHostToDevice);
    hipMemcpy(dGrid, grid.data(), SIZE_GRID, hipMemcpyHostToDevice);
    hipMemcpy(dC, C.data(), SIZE_C, hipMemcpyHostToDevice);
    hipMemcpy(dCOffset, cOffset.data(), SIZE_COFFSET, hipMemcpyHostToDevice);
    hipMemcpy(dIU, iu.data(), SIZE_IU, hipMemcpyHostToDevice);
    hipMemcpy(dIV, iv.data(), SIZE_IV, hipMemcpyHostToDevice);
    gpuCheckErrors("hipMemcpy H2D failure");
}

GridderGPUOlder::~GridderGPUOlder()
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

void GridderGPUOlder::gridder()
{
    cout << "\nGridding on GPU" << endl;
    deviceAllocations();
    copyH2D();

    // Kernel launch
    cout << "Kernel launch" << endl;
    const size_t DSIZE = data.size();
    typedef hipComplex Complex;

    const int SSIZE = 2 * support + 1;

    hipFuncSetCacheConfig(reinterpret_cast<const void*>(devGridKernelOlder), hipFuncCachePreferL1);

    int step = 1;
    int count = 0;
    for (int dind = 0; dind < DSIZE; dind += step)
    {
        step = gridStep(DSIZE, SSIZE, dind);
        dim3 gridDim(SSIZE, step);
        /// PJE: make sure any chevron is tightly packed
        devGridKernelOlder <<<gridDim, SSIZE>>> ((const Complex*)dData, support, (const Complex*)dC,
            dCOffset, dIU, dIV, (Complex*)dGrid, GSIZE, dind);
        gpuCheckErrors("kernel launch (devGridKernel_v0) failure");
        count++;
    }
    cout << "Used " << count << " kernel launches." << endl;

    hipMemcpy(grid.data(), dGrid, SIZE_GRID, hipMemcpyDeviceToHost);
    gpuCheckErrors("hipMemcpy D2H failure");
}
