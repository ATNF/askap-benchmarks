#include "hip/hip_runtime.h"
#include "DegridderGPU.h"
#include <cmath>
#include "../utilities/MaxError.h"

using std::abs;
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

void degridHelper(const Complex* dGrid,
    const int SSIZE,
    const int DSIZE,
    const int GSIZE,
    const int support,
    const Complex* dC,
    const int* dCOffset,
    const int* dIU,
    const int* dIV,
    Complex* dData)
{
    int device;
    hipGetDevice(&device);
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, device);

    int count = 0;
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    
    for (int dind = 0; dind < DSIZE; dind += GRID_SIZE)
    {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(devDegridKernel<64>), GRID_SIZE, blockSize, 0, 0, dGrid, GSIZE, dC, dCOffset, dIU, dIV, dData, dind);
        ++count;
        gpuCheckErrors("cuda kernel launch failure");
    }
    cout << "Used " << count << " kernel launches." << endl;

}

template <typename T2>
void DegridderGPU<T2>::degridder()
{
    //cout << "\nDegridding on GPU" << endl;

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

 
    // Kernel launch
    typedef hipComplex Complex;
    degridHelper((const Complex*)dGrid, SSIZE, DSIZE, GSIZE, support, (const Complex*)dC, dCOffset, dIU, dIV, (Complex*)dData);

    hipMemcpy(data.data(), dData, SIZE_DATA, hipMemcpyDeviceToHost);
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

template void DegridderGPU<std::complex<float>>::degridder();
template void DegridderGPU<std::complex<double>>::degridder();
