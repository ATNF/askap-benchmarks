#include "CudaOlder.h"

using std::vector;
using std::cout;
using std::endl;
using std::min;
using std::max;


// Some constants for findPeak
const int findPeakNBlocks = 4;
const int findPeakWidth = 1024;

struct Peak
{
    size_t pos;
    float val;
};

__host__
static Peak findPeak(const float* dData, size_t N);


__host__
void CudaOlder::reportDevice()
{
    // Report the type of device being used
    int device;
    hipDeviceProp_t devprop;
    hipGetDevice(&device);
    hipGetDeviceProperties(&devprop, device);
    std::cout << "    Using CUDA Device " << device << ": "
        << devprop.name << std::endl;
}

__host__ __device__
CudaOlder::Position CudaOlder::idxToPos(const size_t idx, const int width)
{
    const int y = idx / width;
    const int x = idx % width;
    return CudaOlder::Position(x, y);
}

__host__ __device__
size_t CudaOlder::posToIdx(const int width, const CudaOlder::Position& pos)
{
    return (pos.y * width) + pos.x;
}

__global__
void dSubtractPSF_Older(const float* dPsf,
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
        dResidual[CudaOlder::posToIdx(imageWidth, CudaOlder::Position(x, y))] -= gain * absPeakVal
            * dPsf[CudaOlder::posToIdx(imageWidth, CudaOlder::Position(x - diffx, y - diffy))];
    }
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
void CudaOlder::subtractPSF(const size_t peakPos,
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
    dSubtractPSF_Older <<<numBlocks, threadsPerBlock>>> (dPsf, dResidual, imageWidth,
        startx, starty, stopx, stopy, diffx, diffy, absPeakVal, gGain);
    gpuCheckErrors("kernel launch failure in subtractPSF");
}

void CudaOlder::deconvolve()
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
Peak findPeak(const float* dData, size_t N)
{
    const int nBlocks = findPeakNBlocks; // 4
    vector<Peak> peaks(nBlocks);

    // Initialise a peaks array on the device. Each thread block will return
    // a peak. 
    // Note: dPeaks array is not initialised (hence avoiding the memcpy)
    // It is up to do device function to do that.
    Peak* dPeak;
    hipMalloc(&dPeak, nBlocks * sizeof(Peak));
    gpuCheckErrors("cudaMalloc failure in findPeak");

    // Find peak
    dFindPeak <<<nBlocks, findPeakWidth>>> (dData, N, dPeak);
    gpuCheckErrors("kernel launch failure in findPeak");

    // Get the peaks array back from the device
    hipMemcpy(peaks.data(), dPeak, nBlocks * sizeof(Peak), hipMemcpyDeviceToHost);
    gpuCheckErrors("cudaMemcpy D2H failure in findPeak");

    hipDeviceSynchronize();
    gpuCheckErrors("cudaDeviceSynchronize failure in findPeak");

    hipFree(dPeak);
    gpuCheckErrors("cudaFree failure in findPeak");

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

void CudaOlder::memAlloc()
{
    hipMalloc(&dDirty, SIZE_IMAGE);
    hipMalloc(&dPsf, SIZE_IMAGE);
    hipMalloc(&dResidual, SIZE_IMAGE);
    gpuCheckErrors("cudaMalloc failure");
}

CudaOlder::~CudaOlder()
{
    hipFree(dDirty);
    hipFree(dPsf);
    hipFree(dResidual);
    gpuCheckErrors("cudaFree failure");
    cout << "Cuda Older destructor" << endl;
}

void CudaOlder::copyH2D()
{
    hipMemcpy(dDirty, dirty.data(), SIZE_IMAGE, cudaMemcpyHostToDevice);
    hipMemcpy(dPsf, psf.data(), SIZE_IMAGE, cudaMemcpyHostToDevice);
    hipMemcpy(dResidual, residual.data(), SIZE_IMAGE, cudaMemcpyHostToDevice);
    gpuCheckErrors("cudaMemcpy H2D failure");
}

void CudaOlder::copyD2H()
{
    hipMemcpy(residual.data(), dResidual, SIZE_IMAGE, cudaMemcpyDeviceToHost);
    gpuCheckErrors("cudaMemcpy D2H failure");
}
