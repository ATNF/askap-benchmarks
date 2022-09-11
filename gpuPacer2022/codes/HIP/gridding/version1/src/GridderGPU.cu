#include "GridderGPU.h"

using std::cout;
using std::endl;
using std::vector;
using std::complex;

// Error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

int gridStep(const int DSIZE, const int SSIZE, const int dind, const std::vector<int>&iu, const std::vector<int>&iv);

template <typename T2>
void GridderGPU<T2>::gridder()
{
    cout << "\nGridding on GPU" << endl;

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaCheckErrors("cudaEvent create failure");

    // Device parameters
    const size_t SIZE_DATA = data.size() * sizeof(T2);
    const size_t SIZE_GRID = gpuGrid.size() * sizeof(T2);
    const size_t SIZE_C = C.size() * sizeof(T2);
    const size_t SIZE_COFFSET = cOffset.size() * sizeof(int);
    const size_t SIZE_IU = iu.size() * sizeof(int);
    const size_t SIZE_IV = iv.size() * sizeof(int);

    T2* dData;
    T2* dGrid;
    T2* dC;
    int* dCOffset;
    int* dIU;
    int* dIV;

    // Allocate device vectors
    cudaMalloc(&dData, SIZE_DATA);
    cudaMalloc(&dGrid, SIZE_GRID);
    cudaMalloc(&dC, SIZE_C);
    cudaMalloc(&dCOffset, SIZE_COFFSET);
    cudaMalloc(&dIU, SIZE_IU);
    cudaMalloc(&dIV, SIZE_IV);
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(dData, data.data(), SIZE_DATA, cudaMemcpyHostToDevice);
    cudaMemcpy(dGrid, gpuGrid.data(), SIZE_GRID, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), SIZE_C, cudaMemcpyHostToDevice);
    cudaMemcpy(dCOffset, cOffset.data(), SIZE_COFFSET, cudaMemcpyHostToDevice);
    cudaMemcpy(dIU, iu.data(), SIZE_IU, cudaMemcpyHostToDevice);
    cudaMemcpy(dIV, iv.data(), SIZE_IV, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    /*******************************************************************************************************/
    /*******************************************************************************************************/
    // Kernel launch
    cout << "Kernel launch" << endl;
    const size_t DSIZE = data.size();
    typedef cuComplex Complex;
    cudaFuncSetCacheConfig(devGridKernel, cudaFuncCachePreferL1);

    const int SSIZE = 2 * support + 1;
    int step = 1;

    /*
    This loop steps through each spectral sample
    either 1 or 2 at a time. It will do multiple samples
    if the regions involved do not overlap. If they do,
    only the non-overlapping samples are gridded.

    Gridding multiple points is better, because giving the
    GPU more work to do allows it to hide memory latency
    better. The call to d_gridKernel() is asynchronous
    so subsequent calls to gridStep() overlap with the actual gridding.
    */

    int count = 0;
    for (int dind = 0; dind < DSIZE; dind += step)
    {
        step = gridStep(DSIZE, SSIZE, dind, iu, iv);
        cout << "Step = " << step << endl;
        dim3 gridDim(SSIZE, step);
        devGridKernel << <gridDim, SSIZE >> > ((const Complex*)dData, support, (const Complex*)dC, dCOffset, dIU, dIV, (Complex*)dGrid, GSIZE, dind);
        cudaCheckErrors("kernel launch (devGridKernel_v0) failure");
        count++;
    }
    cout << "Used " << count << " kernel launches." << endl;

    cudaMemcpy(gpuGrid.data(), dGrid, SIZE_GRID, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");

    // Deallocate device vectors
    cudaFree(dData);
    cudaFree(dGrid);
    cudaFree(dC);
    cudaFree(dCOffset);
    cudaFree(dIU);
    cudaFree(dIV);
    cudaCheckErrors("cudaFree failure");
}

int gridStep(const int DSIZE, const int SSIZE, const int dind, const std::vector<int>& iu, const std::vector<int>& iv)
{
    const int MAXSAMPLES = 32;
    for (int step = 1; step <= MAXSAMPLES; ++step)
    {
        for (int check = (step - 1); check >= 0; --check)
        {
            if (!((dind + step) < DSIZE && (
                abs(iu[dind + step] - iu[dind + check]) > SSIZE ||
                abs(iv[dind + step] - iv[dind + check]) > SSIZE)
                ))
            {
                return step;
            }
        }
    }
    return MAXSAMPLES;
}

template void GridderGPU<std::complex<float>>::gridder();
template void GridderGPU<std::complex<double>>::gridder();

