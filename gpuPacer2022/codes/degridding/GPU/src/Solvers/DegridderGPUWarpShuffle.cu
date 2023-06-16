#include "DegridderGPUWarpShuffle.h"

using std::cout;
using std::endl;
using std::vector;
using std::complex;
#define FULL_MASK 0xffffffff

const int supportWS = 64;

// launch_bounds__(2*support+1, 8)
__global__
void devDegridKernelWarpShuffle(
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

    int dindLocal = dind + blockIdx.x;
    int gindStart = iu[dindLocal] + GSIZE * iv[dindLocal] - support;
    int cindStart = cOffset[dindLocal];
    int SSIZE = 2 * support + 1;
    int suppu = threadIdx.x;

    Complex dOrig = data[dindLocal];
    // suppv loop
    for (int suppv = 0; suppv < SSIZE; ++suppv)
    {
        int gind = gindStart + GSIZE * suppv;
        int cind = cindStart + SSIZE * suppv;
        Complex sum = gpuCmulf(grid[gind + suppu], C[cind + suppu]);

        __syncthreads();
        // Reduce within each warp
        if (suppu < SSIZE)
        {
            for (int offset = WARPSIZE / 2; offset > 0; offset /= 2)
            {
#ifdef __NVCC__		
                sum.x += __shfl_down_sync(FULL_MASK, sum.x, offset, WARPSIZE);
                sum.y += __shfl_down_sync(FULL_MASK, sum.y, offset, WARPSIZE);
#else	  
                sum.x += __shfl_down(sum.x, offset, WARPSIZE);
                sum.y += __shfl_down(sum.y, offset, WARPSIZE);
#endif	  
            }

        }

        // Gather warp sums into shared memory
        const int NUMWARPS = (2 * supportWS + 1) / WARPSIZE + 1;
        __shared__ Complex dataShared[NUMWARPS];

        int warp = suppu / WARPSIZE;
        int lane = threadIdx.x & (WARPSIZE - 1); // the lead thread in the warp

        if (lane == 0)
        {
            dataShared[warp] = sum;
        }

        __syncthreads();
        // combine warp sums using a single thread in this block
        if (suppu == 0)
        {
            for (int w = 1; w < NUMWARPS; w++)
            {
                sum = gpuCaddf(sum, dataShared[w]);
            }

            dOrig = gpuCaddf(dOrig, sum);
        }
    }
    if (suppu == 0)
    {
        data[dindLocal] = dOrig;
    }
}


template<typename T2>
void DegridderGPUWarpShuffle<T2>::deviceAllocations()
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
void DegridderGPUWarpShuffle<T2>::copyH2D()
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
DegridderGPUWarpShuffle<T2>::~DegridderGPUWarpShuffle()
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
void DegridderGPUWarpShuffle<T2>::degridder()
{
    deviceAllocations();
    copyH2D();

    // Kernel launch
    const size_t DSIZE = this->data.size();
    // typedef gpuComplex Complex;

    const int SSIZE = 2 * this->support + 1;

    // hipFuncSetCacheConfig(reinterpret_cast<const void*>(devGridKernelOlder), hipFuncCachePreferL1);

    int device;
    gpuGetDevice(&device);
    gpuDeviceProp_t devProp;
    gpuGetDeviceProperties(&devProp, device);

    int gridSize = devProp.maxGridSize[0] / (this->support + 1);  // launch kernels for this number of samples at a time
    assert(SSIZE <= devProp.maxThreadsPerBlock);

    int count = 0;
    //int gridSize = 1024 * devProp.multiProcessorCount; // is starting size, will be reduced as required
    for (int dind = 0; dind < DSIZE; dind += gridSize)
    {
        // if there are less than dimGrid elements left,
        // do the remaining
        if ((DSIZE - dind) < gridSize)
        {
            gridSize = DSIZE - dind;
        }
        devDegridKernelWarpShuffle <<<gridSize, SSIZE>>> ((const Complex*)dGrid, GSIZE, (const Complex*)dC, this->support, dCOffset, dIU, dIV, (Complex*)dData, dind);
        gpuCheckErrors("gpu kernel launch failure");
	++count;
    }
    cout << "Used " << count << " kernel launches." << endl;

    gpuMemcpy(this->data.data(), dData, SIZE_DATA, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}

template void DegridderGPUWarpShuffle<std::complex<float>>::degridder();
template void DegridderGPUWarpShuffle<std::complex<double>>::degridder();
template void DegridderGPUWarpShuffle<std::complex<float>>::deviceAllocations();
template void DegridderGPUWarpShuffle<std::complex<double>>::deviceAllocations();
template void DegridderGPUWarpShuffle<std::complex<float>>::copyH2D();
template void DegridderGPUWarpShuffle<std::complex<double>>::copyH2D();
template DegridderGPUWarpShuffle<std::complex<float>>::~DegridderGPUWarpShuffle();
template DegridderGPUWarpShuffle<std::complex<double>>::~DegridderGPUWarpShuffle();
