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
    grid[gind + threadIdx.x] = hipCfmaf(dataLocal, C[cind + threadIdx.x], grid[gind + threadIdx.x]);
}

template<typename T2>
int GridderGPUOlder<T2>::gridStep(const int DSIZE, const int SSIZE, const int dind)
{
    const int MAXSAMPLES = 32;
    for (int step = 1; step <= MAXSAMPLES; ++step)
    {
        for (int check = (step - 1); check >= 0; --check)
        {
            if (!((dind + step) < DSIZE && (
                abs(this->iu[dind + step] - this->iu[dind + check]) > SSIZE ||
                abs(this->iv[dind + step] - this->iv[dind + check]) > SSIZE)))
            {
                return step;
            }
        }
    }
    return MAXSAMPLES;
}

template<typename T2>
void GridderGPUOlder<T2>::deviceAllocations()
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
void GridderGPUOlder<T2>::copyH2D()
{
    hipMemcpy(dData, this->data.data(), SIZE_DATA, hipMemcpyHostToDevice);
    hipMemcpy(dGrid, this->grid.data(), SIZE_GRID, hipMemcpyHostToDevice);
    hipMemcpy(dC, this->C.data(), SIZE_C, hipMemcpyHostToDevice);
    hipMemcpy(dCOffset, this->cOffset.data(), SIZE_COFFSET, hipMemcpyHostToDevice);
    hipMemcpy(dIU, this->iu.data(), SIZE_IU, hipMemcpyHostToDevice);
    hipMemcpy(dIV, this->iv.data(), SIZE_IV, hipMemcpyHostToDevice);
    gpuCheckErrors("hipMemcpy H2D failure");
}

template<typename T2>
GridderGPUOlder<T2>::~GridderGPUOlder()
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
void GridderGPUOlder<T2>::gridder()
{
    cout << "\nGridding on GPU" << endl;
    deviceAllocations();
    copyH2D();

    // Kernel launch
    cout << "Kernel launch" << endl;
    const size_t DSIZE = this->data.size();
    typedef hipComplex Complex;

    const int SSIZE = 2 * this->support + 1;

    hipFuncSetCacheConfig(reinterpret_cast<const void*>(devGridKernelOlder), hipFuncCachePreferL1);

    int step = 1;
    int count = 0;
    for (int dind = 0; dind < DSIZE; dind += step)
    {
        step = gridStep(DSIZE, SSIZE, dind);
        dim3 gridDim(SSIZE, step);
        /// PJE: make sure any chevron is tightly packed
        devGridKernelOlder <<<gridDim, SSIZE>>> ((const Complex*)dData, this->support, (const Complex*)dC,
            dCOffset, dIU, dIV, (Complex*)dGrid, GSIZE, dind);
        gpuCheckErrors("kernel launch (devGridKernel_v0) failure");
        count++;
    }
    cout << "Used " << count << " kernel launches." << endl;

    hipMemcpy(this->grid.data(), dGrid, SIZE_GRID, hipMemcpyDeviceToHost);
    gpuCheckErrors("hipMemcpy D2H failure");
}

template void GridderGPUOlder<std::complex<float>>::gridder();
template void GridderGPUOlder<std::complex<double>>::gridder();
template void GridderGPUOlder<std::complex<float>>::deviceAllocations();
template void GridderGPUOlder<std::complex<double>>::deviceAllocations();
template void GridderGPUOlder<std::complex<float>>::copyH2D();
template void GridderGPUOlder<std::complex<double>>::copyH2D();
template GridderGPUOlder<std::complex<float>>::~GridderGPUOlder();
template GridderGPUOlder<std::complex<double>>::~GridderGPUOlder();
template int GridderGPUOlder<std::complex<float>>::gridStep(const int DSIZE, const int SSIZE, const int dind);
template int GridderGPUOlder<std::complex<double>>::gridStep(const int DSIZE, const int SSIZE, const int dind);
