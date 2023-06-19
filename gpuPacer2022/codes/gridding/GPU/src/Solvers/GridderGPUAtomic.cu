#include "GridderGPUAtomic.h"
#include "../../utilities/gpuCommon.h"
#include "../../utilities/LoggerUtil.h"

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
            //grid[gind + tind] = gpuCfmaf(dataLocal, C[cind + tind], grid[gind + tind]);
            const Complex tmp = gpuCmulf(dataLocal, C[cind + tind]);
            //grid[gind + tind] = gpuCaddf(grid[gind + tind], tmp);
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
    gpuMalloc(&dData, SIZE_DATA);
    gpuMalloc(&dGrid, SIZE_GRID);
    gpuMalloc(&dC, SIZE_C);
    gpuMalloc(&dCOffset, SIZE_COFFSET);
    gpuMalloc(&dIU, SIZE_IU);
    gpuMalloc(&dIV, SIZE_IV);
    gpuCheckErrors("gpuMalloc failure");
}

template<typename T2>
void GridderGPUAtomic<T2>::copyH2D()
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
GridderGPUAtomic<T2>::~GridderGPUAtomic()
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
void GridderGPUAtomic<T2>::gridder()
{
    cout << "\nGridding on GPU" << endl;
    deviceAllocations();
    copyH2D();

    // Kernel launch
    cout << "Kernel launch" << endl;
    const size_t DSIZE = this->data.size();

    const int SSIZE = 2 * this->support + 1;

    gpuFuncSetCacheConfig(reinterpret_cast<const void*>(devGridKernelAtomic), gpuFuncCachePreferL1);

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

        devGridKernelAtomic <<< gridSize, SSIZE >>> ((const Complex*)dData, this->support, (const Complex*)dC,
            dCOffset, dIU, dIV, (Complex*)dGrid, GSIZE, dind);

        gpuCheckErrors("gpu kernel launch failure");
    }
    cout << "Used " << count << " kernel launches." << endl;

    gpuMemcpy(this->grid.data(), dGrid, SIZE_GRID, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template void GridderGPUAtomic<std::complex<float>>::gridder();
template void GridderGPUAtomic<std::complex<double>>::gridder();
template void GridderGPUAtomic<std::complex<float>>::deviceAllocations();
template void GridderGPUAtomic<std::complex<double>>::deviceAllocations();
template void GridderGPUAtomic<std::complex<float>>::copyH2D();
template void GridderGPUAtomic<std::complex<double>>::copyH2D();
template GridderGPUAtomic<std::complex<float>>::~GridderGPUAtomic();
template GridderGPUAtomic<std::complex<double>>::~GridderGPUAtomic();
