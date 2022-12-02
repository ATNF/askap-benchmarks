#include "DegridderGPUTiled.h"

using std::cout;
using std::endl;
using std::vector;
using std::complex;

const int BLOCK_SIZE = 32; // dim3(BLOCK_SIZE, BLOCK_SIZE)
const int GRID_SIZE = NSAMPLES;

__global__
void devDegridKernelTiled(
        const Complex* grid,
        const int GSIZE,
        const Complex* C,
        const int support,
        const int* cOffset,
        const int* iu,
        const int* iv,
        Complex* data,
        const int i)
{

    const int dind = i + blockIdx.x;

    // The actual starting grid point
    __shared__ int gindShared;
    // The Convolution function point from which we offset
    __shared__ int cindShared;

    int suppU = threadIdx.x;
    int suppV = threadIdx.y;
    int tID = blockDim.x * suppV + suppU;

    if (tID == 0)
    {
        gindShared = iu[dind] + GSIZE * iv[dind] - support;
        cindShared = cOffset[dind];
    }
    __syncthreads();

    const int SSIZE = 2 * support + 1;

    __shared__ float sdata_re[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float sdata_im[BLOCK_SIZE * BLOCK_SIZE];

    sdata_re[tID] = 0.0;
    sdata_im[tID] = 0.0;

    // Block-stride loading
    while (suppV < SSIZE)
    {
        while (suppU < SSIZE)
        {
            int gind = gindShared + GSIZE * (suppV);
            int cind = cindShared + SSIZE * (suppV);

            // copy the local convolution product to shared memory
            sdata_re[tID] += grid[gind + suppU].x * C[cind + suppU].x - grid[gind + suppU].y * C[cind + suppU].y;
            sdata_im[tID] += grid[gind + suppU].x * C[cind + suppU].y + grid[gind + suppU].y * C[cind + suppU].x;
            suppU += blockDim.x;
        }

        suppU = threadIdx.x;
        suppV += blockDim.y;
    }

    for (unsigned int s = (BLOCK_SIZE * BLOCK_SIZE) / 2; s > 0; s >>= 1)
    {
        __syncthreads();
        if (tID < s)
        {
            sdata_re[tID] += sdata_re[tID + s];
            sdata_im[tID] += sdata_im[tID + s];
        }
    }

    if (tID == 0)
    {
        data[dind].x = sdata_re[tID];
        data[dind].y = sdata_im[tID];
    }

}

template<typename T2>
void DegridderGPUTiled<T2>::deviceAllocations()
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
void DegridderGPUTiled<T2>::copyH2D()
{
    cudaMemcpy(dData, this->data.data(), SIZE_DATA, cudaMemcpyHostToDevice);
    cudaMemcpy(dGrid, this->grid.data(), SIZE_GRID, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, this->C.data(), SIZE_C, cudaMemcpyHostToDevice);
    cudaMemcpy(dCOffset, this->cOffset.data(), SIZE_COFFSET, cudaMemcpyHostToDevice);
    cudaMemcpy(dIU, this->iu.data(), SIZE_IU, cudaMemcpyHostToDevice);
    cudaMemcpy(dIV, this->iv.data(), SIZE_IV, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");
}

template<typename T2>
DegridderGPUTiled<T2>::~DegridderGPUTiled()
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
void DegridderGPUTiled<T2>::degridder()
{
    deviceAllocations();
    copyH2D();
    
    int gridSize = GRID_SIZE;
    // Kernel launch
    const size_t DSIZE = this->data.size();
    typedef cuComplex Complex;

    const int SSIZE = 2 * this->support + 1;

    // cudaFuncSetCacheConfig(reinterpret_cast<const void*>(devGridKernelOlder), cudaFuncCachePreferL1);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, device);

    int count = 0;

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    for (int dind = 0; dind < DSIZE; dind += GRID_SIZE)
    {
        if ((DSIZE - dind) < gridSize)
        {
            gridSize = DSIZE - dind;
        }

        devDegridKernelTiled << < GRID_SIZE, blockSize >> > ((const Complex*)dGrid, GSIZE, (const Complex*)dC, this->support, dCOffset, dIU, dIV, (Complex*)dData, dind);
        ++count;
        cudaCheckErrors("cuda kernel launch failure");
    }
    cout << "Used " << count << " kernel launches." << endl;

    cudaMemcpy(this->data.data(), dData, SIZE_DATA, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");
}

template void DegridderGPUTiled<std::complex<float>>::degridder();
template void DegridderGPUTiled<std::complex<double>>::degridder();
template void DegridderGPUTiled<std::complex<float>>::deviceAllocations();
template void DegridderGPUTiled<std::complex<double>>::deviceAllocations();
template void DegridderGPUTiled<std::complex<float>>::copyH2D();
template void DegridderGPUTiled<std::complex<double>>::copyH2D();
template DegridderGPUTiled<std::complex<float>>::~DegridderGPUTiled();
template DegridderGPUTiled<std::complex<double>>::~DegridderGPUTiled();
