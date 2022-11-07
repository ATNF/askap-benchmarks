#include "DegridderGPUWarpShuffle.h"

using std::cout;
using std::endl;
using std::vector;
using std::complex;
#define FULL_MASK 0xffffffff

#ifdef __NVCC__
#define WARPSIZE 32
#else
#define WARPSIZE 64
#endif

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
        Complex sum = cuCmulf(grid[gind + suppu], C[cind + suppu]);

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
                sum = cuCaddf(sum, dataShared[w]);
            }

            dOrig = cuCaddf(dOrig, sum);
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
    cudaMalloc(&dData, SIZE_DATA);
    cudaMalloc(&dGrid, SIZE_GRID);
    cudaMalloc(&dC, SIZE_C);
    cudaMalloc(&dCOffset, SIZE_COFFSET);
    cudaMalloc(&dIU, SIZE_IU);
    cudaMalloc(&dIV, SIZE_IV);
    cudaCheckErrors("cudaMalloc failure");
}

template<typename T2>
void DegridderGPUWarpShuffle<T2>::copyH2D()
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
DegridderGPUWarpShuffle<T2>::~DegridderGPUWarpShuffle()
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
void DegridderGPUWarpShuffle<T2>::degridder()
{
    deviceAllocations();
    copyH2D();

    // Kernel launch
    const size_t DSIZE = data.size();
    typedef cuComplex Complex;

    const int SSIZE = 2 * support + 1;

    // cudaFuncSetCacheConfig(reinterpret_cast<const void*>(devGridKernelOlder), cudaFuncCachePreferL1);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, device);

    int gridSize = devProp.maxGridSize[0] / (support + 1);  // launch kernels for this number of samples at a time
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
        devDegridKernelWarpShuffle << < gridSize, SSIZE >> > ((const Complex*)dGrid, GSIZE, (const Complex*)dC, support, dCOffset, dIU, dIV, (Complex*)dData, dind);
        cudaCheckErrors("cuda kernel launch failure");
    }
    cout << "Used " << count << " kernel launches." << endl;

    cudaMemcpy(data.data(), dData, SIZE_DATA, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");
}

template void DegridderGPUWarpShuffle<std::complex<float>>::degridder();
template void DegridderGPUWarpShuffle<std::complex<double>>::degridder();
template void DegridderGPUWarpShuffle<std::complex<float>>::deviceAllocations();
template void DegridderGPUWarpShuffle<std::complex<double>>::deviceAllocations();
template void DegridderGPUWarpShuffle<std::complex<float>>::copyH2D();
template void DegridderGPUWarpShuffle<std::complex<double>>::copyH2D();
template DegridderGPUWarpShuffle<std::complex<float>>::~DegridderGPUWarpShuffle();
template DegridderGPUWarpShuffle<std::complex<double>>::~DegridderGPUWarpShuffle();