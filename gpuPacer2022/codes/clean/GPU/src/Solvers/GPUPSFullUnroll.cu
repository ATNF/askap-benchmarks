#include "GPUPSFullUnroll.h"

using std::vector;
using std::cout;
using std::endl;
using std::min;
using std::max;

template <unsigned int blockSize>
__device__
void warpReduce_FU(volatile float* data, volatile size_t* index, int tileIdx)
{
    if (blockSize >= 64)
    {
        if (fabs(data[tileIdx + 32]) > fabs(data[tileIdx]))
        {
            data[tileIdx] = data[tileIdx + 32];
            index[tileIdx] = index[tileIdx + 32];
        }
    }
    if (blockSize >= 32)
    {
        if (fabs(data[tileIdx + 16]) > fabs(data[tileIdx]))
        {
            data[tileIdx] = data[tileIdx + 16];
            index[tileIdx] = index[tileIdx + 16];
        }
    }
    if (blockSize >= 16)
    {
        if (fabs(data[tileIdx + 8]) > fabs(data[tileIdx]))
        {
            data[tileIdx] = data[tileIdx + 8];
            index[tileIdx] = index[tileIdx + 8];
        }
    }
    if (blockSize >= 8)
    {
        if (fabs(data[tileIdx + 4]) > fabs(data[tileIdx]))
        {
            data[tileIdx] = data[tileIdx + 4];
            index[tileIdx] = index[tileIdx + 4];
        }
    }
    if (blockSize >= 4)
    {
        if (fabs(data[tileIdx + 2]) > fabs(data[tileIdx]))
        {
            data[tileIdx] = data[tileIdx + 2];
            index[tileIdx] = index[tileIdx + 2];
        }
    }
    if (blockSize >= 2)
    {
        if (fabs(data[tileIdx + 1]) > fabs(data[tileIdx]))
        {
            data[tileIdx] = data[tileIdx + 1];
            index[tileIdx] = index[tileIdx + 1];
        }
    }
}

// Find peak step 2 for full unroll version
__global__
void dFindPeak_Step2_FU(float* data, size_t* inIndex, size_t* outIndex, size_t n)
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

// Find peak step 1 for full unroll version
template <unsigned int blockSize>
__global__
void dFindPeak_Step1_FU(const float* data, float* outMax, size_t* outIndex, size_t n)
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

    __syncthreads();

    if (blockSize >= 512)
    {
        if (tileIdx < 256)
        {
            if (fabs(TILE_VAL[tileIdx + 256]) > fabs(TILE_VAL[tileIdx]))
            {
                TILE_VAL[tileIdx] = TILE_VAL[tileIdx + 256];
                TILE_IDX[tileIdx] = TILE_IDX[tileIdx + 256];
            }

        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tileIdx < 128)
        {
            if (fabs(TILE_VAL[tileIdx + 128]) > fabs(TILE_VAL[tileIdx]))
            {
                TILE_VAL[tileIdx] = TILE_VAL[tileIdx + 128];
                TILE_IDX[tileIdx] = TILE_IDX[tileIdx + 128];
            }
        }
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tileIdx < 64)
        {
            if (fabs(TILE_VAL[tileIdx + 64]) > fabs(TILE_VAL[tileIdx]))
            {
                TILE_VAL[tileIdx] = TILE_VAL[tileIdx + 64];
                TILE_IDX[tileIdx] = TILE_IDX[tileIdx + 64];
            }
        }
        __syncthreads();
    }

    if (tileIdx < 32)
    {
        warpReduce_FU<blockSize>(TILE_VAL, TILE_IDX, tileIdx);
    }

    if (tileIdx == 0)
    {
        outMax[blockIdx.x] = TILE_VAL[tileIdx];
        outIndex[blockIdx.x] = TILE_IDX[tileIdx];
    }
}

__host__
void gpuPSFullUnroll::reportDevice()
{
    GPUReportDevice();
}

__host__ __device__
gpuPSFullUnroll::Position gpuPSFullUnroll::idxToPos(const size_t idx, const int width)
{
    const int y = idx / width;
    const int x = idx % width;
    return gpuPSFullUnroll::Position(x, y);
}

__host__ __device__
size_t gpuPSFullUnroll::posToIdx(const int width, const gpuPSFullUnroll::Position& pos)
{
    return (pos.y * width) + pos.x;
}

// Subtract PSF for full unroll version
__global__
void dSubtractPSF_FU(const float* dPsf,
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
        dResidual[gpuPSFullUnroll::posToIdx(imageWidth, gpuPSFullUnroll::Position(x, y))] -= gain * absPeakVal
            * dPsf[gpuPSFullUnroll::posToIdx(imageWidth, gpuPSFullUnroll::Position(x - diffx, y - diffy))];
    }
}

__host__
void gpuPSFullUnroll::subtractPSF(const size_t peakPos,
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
    dSubtractPSF_FU <<<numBlocks, threadsPerBlock>>> (dPsf, dResidual, imageWidth,
        startx, starty, stopx, stopy, diffx, diffy, absPeakVal, gGain);
    gpuCheckErrors("kernel launch failure in subtractPSF");
}

void gpuPSFullUnroll::deconvolve()
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
gpuPSFullUnroll::Peak gpuPSFullUnroll::findPeak(const float* dData, size_t N)
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

    gpuMalloc(&dMax, SIZE_MAX_VALUE);
    gpuMalloc(&dIndex, SIZE_MAX_INDEX);
    gpuMalloc(&d2Index, sizeof(size_t));
    gpuCheckErrors("gpuMalloc failure!");


    switch (BLOCK_SIZE)
    {
    case 1024:
        dFindPeak_Step1_FU<1024> <<<GRID_SIZE, BLOCK_SIZE>>> (dData, dMax, dIndex, N);
        break;
    case 512:
        dFindPeak_Step1_FU<512> <<<GRID_SIZE, BLOCK_SIZE>>> (dData, dMax, dIndex, N);
        break;
    case 256:
        dFindPeak_Step1_FU<256> <<<GRID_SIZE, BLOCK_SIZE>>> (dData, dMax, dIndex, N);
        break;
    case 128:
        dFindPeak_Step1_FU<128> <<<GRID_SIZE, BLOCK_SIZE>>> (dData, dMax, dIndex, N);
        break;
    case 64:
        dFindPeak_Step1_FU<64> <<<GRID_SIZE, BLOCK_SIZE>>> (dData, dMax, dIndex, N);
        break;
    case 32:
        dFindPeak_Step1_FU<32> <<<GRID_SIZE, BLOCK_SIZE>>> (dData, dMax, dIndex, N);
        break;
    case 16:
        dFindPeak_Step1_FU<16> <<<GRID_SIZE, BLOCK_SIZE>>> (dData, dMax, dIndex, N);
        break;
    case 8:
        dFindPeak_Step1_FU<8> <<<GRID_SIZE, BLOCK_SIZE>>> (dData, dMax, dIndex, N);
        break;
    case 4:
        dFindPeak_Step1_FU<4> <<<GRID_SIZE, BLOCK_SIZE>>> (dData, dMax, dIndex, N);
        break;
    case 2:
        dFindPeak_Step1_FU<2> <<<GRID_SIZE, BLOCK_SIZE>>> (dData, dMax, dIndex, N);
        break;
    case 1:
        dFindPeak_Step1_FU<1> <<<GRID_SIZE, BLOCK_SIZE>>> (dData, dMax, dIndex, N);
        break;
    }

    gpuCheckErrors("gpu kernel launch 1 failure!");
    dFindPeak_Step2_FU <<<1, BLOCK_SIZE>>> (dMax, dIndex, d2Index, GRID_SIZE);
    gpuCheckErrors("gpu kernel launch 2 failure!");

    gpuMemcpy(hMax.data(), dMax, sizeof(float), gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure in findPeak (hmax)!");
    gpuMemcpy(hIndex.data(), d2Index, sizeof(size_t), gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure in findPeak (hindex)!");

    Peak p;
    p.val = hMax[0];
    p.pos = hIndex[0];


    gpuFree(dMax);
    gpuFree(dIndex);
    gpuFree(d2Index);
    gpuCheckErrors("gpuFree failure!");

    return p;
}

void gpuPSFullUnroll::memAlloc()
{
    gpuMalloc(&dDirty, SIZE_IMAGE);
    gpuMalloc(&dPsf, SIZE_IMAGE);
    gpuMalloc(&dResidual, SIZE_IMAGE);
    gpuCheckErrors("gpuMalloc failure");
}

gpuPSFullUnroll::~gpuPSFullUnroll()
{
    gpuFree(dDirty);
    gpuFree(dPsf);
    gpuFree(dResidual);
    gpuCheckErrors("gpuFree failure");
    cout << "gpu PS Full Unroll destructor" << endl;
}

void gpuPSFullUnroll::copyH2D()
{
    gpuMemcpy(dDirty, dirty.data(), SIZE_IMAGE, gpuMemcpyHostToDevice);
    gpuMemcpy(dPsf, psf.data(), SIZE_IMAGE, gpuMemcpyHostToDevice);
    gpuMemcpy(dResidual, residual.data(), SIZE_IMAGE, gpuMemcpyHostToDevice);
    gpuCheckErrors("gpuMemcpy H2D failure");
}

void gpuPSFullUnroll::copyD2H()
{
    gpuMemcpy(residual.data(), dResidual, SIZE_IMAGE, gpuMemcpyDeviceToHost);
    gpuCheckErrors("gpuMemcpy D2H failure");
}
