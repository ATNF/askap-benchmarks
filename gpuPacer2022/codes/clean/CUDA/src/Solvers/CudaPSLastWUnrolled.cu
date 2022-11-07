#include "CudaPSLastWUnrolled.h"

using std::vector;
using std::cout;
using std::endl;
using std::min;
using std::max;

__device__
void warpReduce_LW(volatile float* data, volatile size_t* index, int tileIdx)
{
    if (fabs(data[tileIdx + 32]) > fabs(data[tileIdx]))
    {
        data[tileIdx] = data[tileIdx + 32];
        index[tileIdx] = index[tileIdx + 32];
    }
    if (fabs(data[tileIdx + 16]) > fabs(data[tileIdx]))
    {
        data[tileIdx] = data[tileIdx + 16];
        index[tileIdx] = index[tileIdx + 16];
    }
    if (fabs(data[tileIdx + 8]) > fabs(data[tileIdx]))
    {
        data[tileIdx] = data[tileIdx + 8];
        index[tileIdx] = index[tileIdx + 8];
    }
    if (fabs(data[tileIdx + 4]) > fabs(data[tileIdx]))
    {
        data[tileIdx] = data[tileIdx + 4];
        index[tileIdx] = index[tileIdx + 4];
    }
    if (fabs(data[tileIdx + 2]) > fabs(data[tileIdx]))
    {
        data[tileIdx] = data[tileIdx + 2];
        index[tileIdx] = index[tileIdx + 2];
    }
    if (fabs(data[tileIdx + 1]) > fabs(data[tileIdx]))
    {
        data[tileIdx] = data[tileIdx + 1];
        index[tileIdx] = index[tileIdx + 1];
    }
}


// Find peak step 2 for last warp unrolled version
__global__
void dFindPeak_Step2_LW(float* data, size_t* inIndex, size_t* outIndex, size_t n)
{
    __shared__ float TILE_VAL[BLOCK_SIZE];
    __shared__ size_t TILE_IDX[BLOCK_SIZE];

    size_t tileIdx = threadIdx.x;
    TILE_VAL[tileIdx] = 0.0;
    TILE_IDX[tileIdx] = 0;
    size_t globalIdx = threadIdx.x + blockIdx.x * blockDim.x;

    // grid stride loop to load data
    while (globalIdx < n)
    {
        if (fabs(data[globalIdx]) > fabs(TILE_VAL[tileIdx]))
        {
            TILE_VAL[tileIdx] = data[globalIdx];
            TILE_IDX[tileIdx] = inIndex[globalIdx];
        }
        globalIdx += gridDim.x * blockDim.x;
    }

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        __syncthreads();

        // parallel sweep reduction
        if (tileIdx < s)
        {
            if (fabs(TILE_VAL[tileIdx + s]) > fabs(TILE_VAL[tileIdx]))
            {
                TILE_VAL[tileIdx] = TILE_VAL[tileIdx + s];
                TILE_IDX[tileIdx] = TILE_IDX[tileIdx + s];
            }
        }
    }

    if (tileIdx == 0)
    {
        outIndex[blockIdx.x] = TILE_IDX[tileIdx];
        data[blockIdx.x] = TILE_VAL[tileIdx];
    }
}

// Find peak step 1 for the version "last warp unrolled"
__global__
void dFindPeak_Step1_LW(const float* data, float* outMax, size_t* outIndex, size_t n)
{
    __shared__ float TILE_VAL[BLOCK_SIZE];
    __shared__ size_t TILE_IDX[BLOCK_SIZE];

    size_t tileIdx = threadIdx.x;

    TILE_VAL[tileIdx] = 0.0;
    TILE_IDX[tileIdx] = 0;

    size_t globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t gridSize = gridDim.x * blockDim.x;

    // grid stride loop to load data
    while (globalIdx < n)
    {
        if (fabs(data[globalIdx]) > fabs(TILE_VAL[tileIdx]))
        {
            TILE_VAL[tileIdx] = data[globalIdx];
            TILE_IDX[tileIdx] = globalIdx;
        }
        globalIdx += gridSize;
    }

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        // parallel sweep reduction
        if (tileIdx < s)
        {
            if (fabs(TILE_VAL[tileIdx + s]) > fabs(TILE_VAL[tileIdx]))
            {
                TILE_VAL[tileIdx] = TILE_VAL[tileIdx + s];
                TILE_IDX[tileIdx] = TILE_IDX[tileIdx + s];
            }
        }
        __syncthreads();
    }

    if (tileIdx < 32)
    {
        warpReduce_LW(TILE_VAL, TILE_IDX, tileIdx);
    }

    if (tileIdx == 0)
    {
        outMax[blockIdx.x] = TILE_VAL[tileIdx];
        outIndex[blockIdx.x] = TILE_IDX[tileIdx];
    }
}

__host__
void CudaPSLastWUnrolled::reportDevice()
{
    // Report the type of device being used
    int device;
    cudaDeviceProp devprop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devprop, device);
    std::cout << "    Using GPU Device " << device << ": "
        << devprop.name << std::endl;
}

__host__ __device__
CudaPSLastWUnrolled::Position CudaPSLastWUnrolled::idxToPos(const size_t idx, const int width)
{
    const int y = idx / width;
    const int x = idx % width;
    return CudaPSLastWUnrolled::Position(x, y);
}

__host__ __device__
size_t CudaPSLastWUnrolled::posToIdx(const int width, const CudaPSLastWUnrolled::Position& pos)
{
    return (pos.y * width) + pos.x;
}

// Subtract PSF for full unroll version
__global__
void dSubtractPSF_LW(const float* dPsf,
    float* dResidual,
    const int imageWidth,
    const int startx, const int starty,
    int const stopx, const int stopy,
    const int diffx, const int diffy,
    const float absPeakVal, const float gain)
{
    const int x = startx + threadIdx.x + (blockIdx.x * blockDim.x);
    const int y = starty + threadIdx.y + (blockIdx.y * blockDim.y);

    // Because thread blocks are of size 16, and the workload is not always
    // a multiple of 16, need to ensure only those threads whose responsibility
    // lies in the work area actually do work
    if (x <= stopx && y <= stopy)
    {
        dResidual[CudaPSLastWUnrolled::posToIdx(imageWidth, CudaPSLastWUnrolled::Position(x, y))] -= gain * absPeakVal
            * dPsf[CudaPSLastWUnrolled::posToIdx(imageWidth, CudaPSLastWUnrolled::Position(x - diffx, y - diffy))];
    }
}

__host__
void CudaPSLastWUnrolled::subtractPSF(const size_t peakPos,
    const size_t psfPeakPos,
    const float absPeakVal)
{
    const int blockDim = 16;

    const int rx = idxToPos(peakPos, imageWidth).x;
    const int ry = idxToPos(peakPos, imageWidth).y;

    const int px = idxToPos(psfPeakPos, imageWidth).x;
    const int py = idxToPos(psfPeakPos, imageWidth).y;

    const int diffx = rx - px;
    const int diffy = ry - px;

    const int startx = max(0, rx - px);
    const int starty = max(0, ry - py);

    const int stopx = min(imageWidth - 1, rx + (imageWidth - px - 1));
    const int stopy = min(imageWidth - 1, ry + (imageWidth - py - 1));

    // Note: Both start* and stop* locations are inclusive.
    const int blocksx = ceil((stopx - startx + 1.0) / static_cast<float>(blockDim));
    const int blocksy = ceil((stopy - starty + 1.0) / static_cast<float>(blockDim));

    dim3 numBlocks(blocksx, blocksy);
    dim3 threadsPerBlock(blockDim, blockDim);
    dSubtractPSF_LW <<<numBlocks, threadsPerBlock>>> (dPsf, dResidual, imageWidth,
        startx, starty, stopx, stopy, diffx, diffy, absPeakVal, gGain);
    gpuCheckErrors("kernel launch failure in subtractPSF");
}

void CudaPSLastWUnrolled::deconvolve()
{
    reportDevice();

    residual = dirty;

    // Allocate memory for device vectors
    memAlloc();

    // Copy data from host to device
    copyH2D();

    // Find peak of psf
    Peak psfPeak = findPeak(dPsf, psf.size());

    cout << "Found peak of PSF: " << "Maximum = " << psfPeak.val
        << " at location " << idxToPos(psfPeak.pos, imageWidth).x << ","
        << idxToPos(psfPeak.pos, imageWidth).y << endl;

    for (unsigned int i = 0; i < gNiters; ++i)
    {
        // Find peak in the residual image
        Peak peak = findPeak(dResidual, residual.size());
        if ((i + 1) % 100 == 0 || i == 0)
        {
            cout << "Iteration: " << i + 1 << " - Maximum = " << peak.val
                << " at location " << idxToPos(peak.pos, imageWidth).x << ","
                << idxToPos(peak.pos, imageWidth).y << ", index = " << peak.pos << endl;
        }

        // Check if threshold has been reached
        if (abs(peak.val) < gThreshold)
        {
            cout << "Reached stopping threshold" << endl;
            break;
        }

        // Subtract the PSF from the residual image
        // This function will launch a kernel
        // asynchronously, need to sync later
        subtractPSF(peak.pos, psfPeak.pos, peak.val);
        // Add to model
        model[peak.pos] += peak.val * gGain;
    }

    copyD2H();

}

__host__
CudaPSLastWUnrolled::Peak CudaPSLastWUnrolled::findPeak(const float* dData, size_t N)
{
    const size_t SIZE_DATA = N * sizeof(float);
    const size_t SIZE_MAX_VALUE = GRID_SIZE * sizeof(float);
    const size_t SIZE_MAX_INDEX = GRID_SIZE * sizeof(size_t);

    // Host vector for max values
    vector<float> hMax(GRID_SIZE, 0.0);
    // Host vector for index values
    vector<size_t> hIndex(GRID_SIZE, 0);

    // Device vectors
    float* dMax;
    size_t* dIndex;
    size_t* d2Index;

    cudaMalloc(&dMax, SIZE_MAX_VALUE);
    cudaMalloc(&dIndex, SIZE_MAX_INDEX);
    cudaMalloc(&d2Index, sizeof(size_t));
    gpuCheckErrors("cudaMalloc failure!");


    dFindPeak_Step1_LW <<<GRID_SIZE, BLOCK_SIZE>>> (dData, dMax, dIndex, N);
    gpuCheckErrors("cuda kernel launch 1 failure!");
    dFindPeak_Step2_LW <<<1, BLOCK_SIZE>>> (dMax, dIndex, d2Index, GRID_SIZE);
    gpuCheckErrors("cuda kernel launch 2 failure!");

    cudaMemcpy(hMax.data(), dMax, sizeof(float), cudaMemcpyDeviceToHost);
    gpuCheckErrors("cudaMemcpy D2H failure in findPeak (hmax)!");
    cudaMemcpy(hIndex.data(), d2Index, sizeof(size_t), cudaMemcpyDeviceToHost);
    gpuCheckErrors("cudaMemcpy D2H failure in findPeak (hindex)!");

    Peak p;
    p.val = hMax[0];
    p.pos = hIndex[0];


    cudaFree(dMax);
    cudaFree(dIndex);
    cudaFree(d2Index);
    gpuCheckErrors("cudaFree failure!");

    return p;
}

void CudaPSLastWUnrolled::memAlloc()
{
    cudaMalloc(&dDirty, SIZE_IMAGE);
    cudaMalloc(&dPsf, SIZE_IMAGE);
    cudaMalloc(&dResidual, SIZE_IMAGE);
    gpuCheckErrors("cudaMalloc failure");
}

CudaPSLastWUnrolled::~CudaPSLastWUnrolled()
{
    cudaFree(dDirty);
    cudaFree(dPsf);
    cudaFree(dResidual);
    gpuCheckErrors("cudaFree failure");
    cout << "Cuda PS Full Unroll destructor" << endl;
}

void CudaPSLastWUnrolled::copyH2D()
{
    cudaMemcpy(dDirty, dirty.data(), SIZE_IMAGE, cudaMemcpyHostToDevice);
    cudaMemcpy(dPsf, psf.data(), SIZE_IMAGE, cudaMemcpyHostToDevice);
    cudaMemcpy(dResidual, residual.data(), SIZE_IMAGE, cudaMemcpyHostToDevice);
    gpuCheckErrors("cudaMemcpy H2D failure");
}

void CudaPSLastWUnrolled::copyD2H()
{
    cudaMemcpy(residual.data(), dResidual, SIZE_IMAGE, cudaMemcpyDeviceToHost);
    gpuCheckErrors("cudaMemcpy D2H failure");
}
