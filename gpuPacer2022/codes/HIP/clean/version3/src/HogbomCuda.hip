#include "hip/hip_runtime.h"
#include "HogbomCuda.h"

using std::vector;
using std::cout;
using std::endl;

// Error checking macro
#define gpuCheckErrors(msg) \
    do { \
        hipError_t __err = hipGetLastError(); \
        if (__err != hipSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, hipGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

struct Peak
{
    size_t pos;
    float val;
};

struct Position 
{
    __host__ __device__
        Position(int _x, int _y) : x(_x), y(_y) { };
    int x;
    int y;
};

__host__ __device__
static Position idxToPos(const size_t idx, const int width)
{
    const int y = idx / width;
    const int x = idx % width;
    return Position(x, y);
}

__host__ __device__
static size_t posToIdx(const int width, const Position& pos)
{
    return (pos.y * width) + pos.x;
}

__device__
void warpReduce(volatile float* data, volatile size_t* index, int tileIdx)
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

__global__
void dFindPeak_Step2(float* data, size_t* inIndex, size_t* outIndex, size_t n)
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

__global__
void dFindPeak_Step1(const float* data, float* outMax, size_t* outIndex, size_t n)
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
        warpReduce(TILE_VAL, TILE_IDX, tileIdx);
    }

    if (tileIdx == 0)
    {
        outMax[blockIdx.x] = TILE_VAL[tileIdx];
        outIndex[blockIdx.x] = TILE_IDX[tileIdx];
    }
}

__host__
static Peak findPeak(const float* dData, size_t N)
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

    hipMalloc(&dMax, SIZE_MAX_VALUE);
    hipMalloc(&dIndex, SIZE_MAX_INDEX);
    hipMalloc(&d2Index, sizeof(size_t));
    gpuCheckErrors("hipMalloc failure!");
    
    hipLaunchKernelGGL(dFindPeak_Step1, GRID_SIZE, BLOCK_SIZE, 0, 0, dData, dMax, dIndex, N);
    gpuCheckErrors("cuda kernel launch 1 failure!");
    hipLaunchKernelGGL(dFindPeak_Step2, 1, BLOCK_SIZE, 0, 0, dMax, dIndex, d2Index, GRID_SIZE);
    gpuCheckErrors("cuda kernel launch 2 failure!");

    hipMemcpy(hMax.data(), dMax, sizeof(float), hipMemcpyDeviceToHost);
    gpuCheckErrors("hipMemcpy D2H failure!");
    hipMemcpy(hIndex.data(), d2Index, sizeof(size_t), hipMemcpyDeviceToHost);
    gpuCheckErrors("hipMemcpy D2H failure!");

    Peak p;
    p.val = hMax[0];
    p.pos = hIndex[0];


    hipFree(dMax);
    hipFree(dIndex);
    hipFree(d2Index);
    gpuCheckErrors("hipFree failure!");
    
    return p;
}

__global__
void dSubtractPSF(const float* dPsf,
    const int psfWidth,
    float* dResidual,
    const int residualWidth,
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
        dResidual[posToIdx(residualWidth, Position(x, y))] -= gain * absPeakVal
            * dPsf[posToIdx(psfWidth, Position(x - diffx, y - diffy))];
    }
}

__host__
static void subtractPSF(const float* dPsf,
    const int psfWidth,
    float* dResidual,
    const int residualWidth,
    const size_t peakPos,
    const size_t psfPeakPos,
    const float absPeakVal,
    const float gain)
{
    const int blockDim = 16;

    const int rx = idxToPos(peakPos, residualWidth).x;
    const int ry = idxToPos(peakPos, residualWidth).y;

    const int px = idxToPos(psfPeakPos, psfWidth).x;
    const int py = idxToPos(psfPeakPos, psfWidth).y;

    const int diffx = rx - px;
    const int diffy = ry - px;

    const int startx = std::max(0, rx - px);
    const int starty = std::max(0, ry - py);

    const int stopx = std::min(residualWidth - 1, rx + (psfWidth - px - 1));
    const int stopy = std::min(residualWidth - 1, ry + (psfWidth - py - 1));

    // Note: Both start* and stop* locations are inclusive.
    const int blocksx = ceil((stopx - startx + 1.0) / static_cast<float>(blockDim));
    const int blocksy = ceil((stopy - starty + 1.0) / static_cast<float>(blockDim));

    dim3 numBlocks(blocksx, blocksy);
    dim3 threadsPerBlock(blockDim, blockDim);
    hipLaunchKernelGGL(dSubtractPSF, numBlocks, threadsPerBlock, 0, 0, dPsf, psfWidth, dResidual, residualWidth,
        startx, starty, stopx, stopy, diffx, diffy, absPeakVal, gain);
    gpuCheckErrors("kernel launch failure in subtractPSF");
}

__host__
void HogbomCuda::deconvolve(const vector<float>& dirty,
    const size_t dirtyWidth,
    const vector<float>& psf,
    const size_t psfWidth,
    vector<float>& model,
    vector<float>& residual)
{
    reportDevice();
    
    const size_t SIZE_DIRTY = dirty.size() * sizeof(float);
    const size_t SIZE_PSF = psf.size() * sizeof(float);
    const size_t SIZE_RESIDUAL = residual.size() * sizeof(float);
    
    residual = dirty;

    // Allocate device memory
    float* dDirty;
    float* dPsf;
    float* dResidual;

    hipMalloc(&dDirty, SIZE_DIRTY);
    hipMalloc(&dPsf, SIZE_PSF);
    hipMalloc(&dResidual, SIZE_RESIDUAL);
    gpuCheckErrors("hipMalloc failure");

    // Copy host to device
    hipMemcpy(dDirty, dirty.data(), SIZE_DIRTY, hipMemcpyHostToDevice);
    hipMemcpy(dPsf, psf.data(), SIZE_PSF, hipMemcpyHostToDevice);
    hipMemcpy(dResidual, residual.data(), SIZE_RESIDUAL, hipMemcpyHostToDevice);
    gpuCheckErrors("hipMemcpy H2D failure");

    // Find peak of psf
    Peak psfPeak = findPeak(dPsf, psf.size());

//    cout << "Found peak of PSF: " << "Maximum = " << psfPeak.val
//        << " at location " << idxToPos(psfPeak.pos, psfWidth).x << ","
//        << idxToPos(psfPeak.pos, psfWidth).y << endl;

    for (unsigned int i = 0; i < gNiters; ++i)
    {
        // Find peak in the residual image
        Peak peak = findPeak(dResidual, residual.size()); 

 //       if ((i + 1) % 100 == 0 || i == 0)
 //       {
   //         cout << "Iteration: " << i + 1 << " - Maximum = " << peak.val
     //           << " at location " << idxToPos(peak.pos, dirtyWidth).x << ","
       //         << idxToPos(peak.pos, dirtyWidth).y << ", index = " << peak.pos << endl;
     //   }
        
        // Check if threshold has been reached
        if (abs(peak.val) < gThreshold)
        {
            cout << "Reached stopping threshold" << endl;
            break;
        }

        // Subtract the PSF from the residual image
        // This function will launch a kernel
        // asynchronously, need to sync later
        subtractPSF(dPsf, psfWidth, dResidual, dirtyWidth, peak.pos, psfPeak.pos, peak.val, gGain);

        // Add to model
        model[peak.pos] += peak.val * gGain;

    }

    // Copy device arrays back into the host vector
    hipMemcpy(residual.data(), dResidual, SIZE_RESIDUAL, hipMemcpyDeviceToHost);
    gpuCheckErrors("hipMemcpy D2H failure");

    hipFree(dDirty);
    hipFree(dPsf);
    hipFree(dResidual);
    gpuCheckErrors("hipFree failure");
}

__host__
void HogbomCuda::reportDevice()
{
    // Report the type of device being used
    int device;
    hipDeviceProp_t devprop;
    hipGetDevice(&device);
    hipGetDeviceProperties(&devprop, device);
    std::cout << "    Using CUDA Device " << device << ": "
        << devprop.name << std::endl;
}
