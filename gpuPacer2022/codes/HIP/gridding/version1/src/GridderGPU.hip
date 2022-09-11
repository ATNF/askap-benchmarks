#include "GridderGPU.h"

using std::cout;
using std::endl;
using std::vector;
using std::complex;

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

int gridStep(const int DSIZE, const int SSIZE, const int dind, const std::vector<int>&iu, const std::vector<int>&iv);

template <typename T2>
void GridderGPU<T2>::gridder()
{
    cout << "\nGridding on GPU" << endl;

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
    hipMalloc(&dData, SIZE_DATA);
    hipMalloc(&dGrid, SIZE_GRID);
    hipMalloc(&dC, SIZE_C);
    hipMalloc(&dCOffset, SIZE_COFFSET);
    hipMalloc(&dIU, SIZE_IU);
    hipMalloc(&dIV, SIZE_IV);
    gpuCheckErrors("hipMalloc failure");

    hipMemcpy(dData, data.data(), SIZE_DATA, hipMemcpyHostToDevice);
    hipMemcpy(dGrid, gpuGrid.data(), SIZE_GRID, hipMemcpyHostToDevice);
    hipMemcpy(dC, C.data(), SIZE_C, hipMemcpyHostToDevice);
    hipMemcpy(dCOffset, cOffset.data(), SIZE_COFFSET, hipMemcpyHostToDevice);
    hipMemcpy(dIU, iu.data(), SIZE_IU, hipMemcpyHostToDevice);
    hipMemcpy(dIV, iv.data(), SIZE_IV, hipMemcpyHostToDevice);
    gpuCheckErrors("hipMemcpy H2D failure");

    /*******************************************************************************************************/
    /*******************************************************************************************************/
    // Kernel launch
    cout << "Kernel launch" << endl;
    const size_t DSIZE = data.size();
    typedef hipComplex Complex;
    hipFuncSetCacheConfig(reinterpret_cast<const void*>(devGridKernel), hipFuncCachePreferL1);

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
        dim3 gridDim(SSIZE, step);
        devGridKernel << <gridDim, SSIZE >> > ((const Complex*)dData, support, (const Complex*)dC, dCOffset, dIU, dIV, (Complex*)dGrid, GSIZE, dind);
        gpuCheckErrors("kernel launch (devGridKernel_v0) failure");
        count++;
    }
    cout << "Used " << count << " kernel launches." << endl;

    hipMemcpy(gpuGrid.data(), dGrid, SIZE_GRID, hipMemcpyDeviceToHost);
    gpuCheckErrors("hipMemcpy D2H failure");

    // Deallocate device vectors
    hipFree(dData);
    hipFree(dGrid);
    hipFree(dC);
    hipFree(dCOffset);
    hipFree(dIU);
    hipFree(dIV);
    gpuCheckErrors("hipFree failure");
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

