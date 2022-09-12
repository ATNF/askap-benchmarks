#include "HogbomCuda.h"

using std::vector;
using std::cout;
using std::endl;


// Some constants for findPeak
const int findPeakNBlocks = 4;
const int findPeakWidth = 1024;

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

__global__
void dFindPeak(const float* image, size_t size, Peak* absPeak)
{
    __shared__ float maxVal[findPeakWidth];
    __shared__ size_t maxPos[findPeakWidth];

    const int column = blockDim.x * blockIdx.x + threadIdx.x;
    maxVal[threadIdx.x] = 0.0;
    maxPos[threadIdx.x] = 0;

    for (int idx = column; idx < size; idx += 4096)
    {
        if (abs(image[idx]) > abs(maxVal[threadIdx.x]))
        {
            maxVal[threadIdx.x] = image[idx];
            maxPos[threadIdx.x] = idx;
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        absPeak[blockIdx.x].val = 0.0;
        absPeak[blockIdx.x].pos = 0;
        for (int i = 0; i < findPeakWidth; ++i)
        {
            if (abs(maxVal[i]) > abs(absPeak[blockIdx.x].val))
            {
                absPeak[blockIdx.x].val = maxVal[i];
                absPeak[blockIdx.x].pos = maxPos[i];
            }
        }
    }
}

__host__
static Peak findPeak(const float* dImage, size_t size)
{
    const int nBlocks = findPeakNBlocks; // 4
    vector<Peak> peaks(nBlocks);

    // Initialise a peaks array on the device. Each thread block will return
    // a peak. 
    // Note: dPeaks array is not initialised (hence avoiding the memcpy)
    // It is up to do device function to do that.
    Peak* dPeak;
    gpuErrchk(hipMalloc(&dPeak, nBlocks * sizeof(Peak)));
    gpuCheckErrors("hipMalloc failure in findPeak");

    // Find peak
    dFindPeak <<<nBlocks, findPeakWidth>>>(dImage, size, dPeak);
    gpuCheckErrors("kernel launch failure in findPeak");

    // Get the peaks array back from the device
    gpuErrchk(hipMemcpy(peaks.data(), dPeak, nBlocks * sizeof(Peak), hipMemcpyDeviceToHost));
    gpuCheckErrors("hipMemcpy D2H failure in findPeak");

    //=================================================================================
    // OPTIMIZATION 1 - Unnecessary synchronization
    //=================================================================================
//    hipDeviceSynchronize();
//    gpuCheckErrors("hipDeviceSynchronize failure in findPeak");

    gpuErrchk(hipFree(dPeak));
    gpuCheckErrors("hipFree failure in findPeak");

    // Each thread block return a peak, find the absolute maximum
    Peak p;
    p.val = 0;
    p.pos = 0;
    for (int i = 0; i < nBlocks; ++i)
    {
        if (abs(peaks[i].val) > abs(p.val))
        {
            p.val = peaks[i].val;
            p.pos = peaks[i].pos;
        }
    }

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
    const int blocksx = std::ceil((stopx - startx + 1.0) / static_cast<float>(blockDim));
    const int blocksy = std::ceil((stopy - starty + 1.0) / static_cast<float>(blockDim));

    dim3 numBlocks(blocksx, blocksy);
    dim3 threadsPerBlock(blockDim, blockDim);
    dSubtractPSF <<<numBlocks, threadsPerBlock>>>(dPsf, psfWidth, dResidual, residualWidth,
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

    cout << "Found peak of PSF: " << "Maximum = " << psfPeak.val
        << " at location " << idxToPos(psfPeak.pos, psfWidth).x << ","
        << idxToPos(psfPeak.pos, psfWidth).y << endl;
    //=================================================================================
    // OPTIMIZATION 3 - Unnecessary assert
    //=================================================================================
    //assert(psfPeak.pos <= psf.size());
    // May be included in DEBUG

    for (unsigned int i = 0; i < gNiters; ++i)
    {
        // Find peak in the residual image
        Peak peak = findPeak(dResidual, residual.size()); 
        //=================================================================================
        // OPTIMIZATION 4 - Unnecessary assert
        //=================================================================================
        //assert(peak.pos <= residual.size());
        // May be included in DEBUG

	if ((i+1) % 100 == 0)
	{
	    cout << "Iteration: " << i + 1 << " - Maximum = " << peak.val 
		<< " at location " << idxToPos(peak.pos, dirtyWidth).x << "," 
		<< idxToPos(peak.pos, dirtyWidth).y << endl;
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
        subtractPSF(dPsf, psfWidth, dResidual, dirtyWidth, peak.pos, psfPeak.pos, peak.val, gGain);

        // Add to model
        model[peak.pos] += peak.val * gGain;

        // Wait for the PSF subtraction to finish
        //=================================================================================
        // OPTIMIZATION 2 - Unnecessary synchronization?? !!!!!!!but CAREFUL!!!!!!!!!!!!!!!!!!!!!
        //=================================================================================
        //hipDeviceSynchronize();
        //gpuCheckErrors("hipDeviceSynchronize failure in deconvolve");

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
