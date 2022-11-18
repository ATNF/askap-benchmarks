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

void DegridderGPUTiled::deviceAllocations()
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

void DegridderGPUTiled::copyH2D()
{
    hipMemcpy(dData, data.data(), SIZE_DATA, hipMemcpyHostToDevice);
    hipMemcpy(dGrid, grid.data(), SIZE_GRID, hipMemcpyHostToDevice);
    hipMemcpy(dC, C.data(), SIZE_C, hipMemcpyHostToDevice);
    hipMemcpy(dCOffset, cOffset.data(), SIZE_COFFSET, hipMemcpyHostToDevice);
    hipMemcpy(dIU, iu.data(), SIZE_IU, hipMemcpyHostToDevice);
    hipMemcpy(dIV, iv.data(), SIZE_IV, hipMemcpyHostToDevice);
    gpuCheckErrors("hipMemcpy H2D failure");
}

DegridderGPUTiled::~DegridderGPUTiled()
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

void DegridderGPUTiled::degridder()
{
    deviceAllocations();
    copyH2D();
    
    int gridSize = GRID_SIZE;
    // Kernel launch
    const size_t DSIZE = data.size();
    typedef hipComplex Complex;

    const int SSIZE = 2 * support + 1;

    // hipFuncSetCacheConfig(reinterpret_cast<const void*>(devGridKernelOlder), hipFuncCachePreferL1);

    int device;
    hipGetDevice(&device);
    hipDeviceProp devProp;
    hipGetDeviceProperties(&devProp, device);

    int count = 0;

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    for (int dind = 0; dind < DSIZE; dind += GRID_SIZE)
    {
        if ((DSIZE - dind) < gridSize)
        {
            gridSize = DSIZE - dind;
        }

        devDegridKernelTiled <<<GRID_SIZE, blockSize>>> ((const Complex*)dGrid, GSIZE, (const Complex*)dC, support, dCOffset, dIU, dIV, (Complex*)dData, dind);
        ++count;
        gpuCheckErrors("hip kernel launch failure");
    }
    cout << "Used " << count << " kernel launches." << endl;

    hipMemcpy(data.data(), dData, SIZE_DATA, hipMemcpyDeviceToHost);
    gpuCheckErrors("hipMemcpy D2H failure");
}

