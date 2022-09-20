#include "DegridderGPU.h"

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
    cudaGetDevice(&device);
    cudaDeviceProp_t devProp;
    cudaGetDeviceProperties(&devProp, device);

    //cout << "maxGridSize "<<devProp.maxGridSize[0]<<" maxThreadsPerBlock = "<<devProp.maxThreadsPerBlock << endl;
    int gridSize = devProp.maxGridSize[0]/(support+1);  // launch kernels for this number of samples at a time
    assert(SSIZE <= devProp.maxThreadsPerBlock);

    int count = 0;
    for (int dind = 0; dind < DSIZE; dind += gridSize)
    {
        // if there are less than dimGrid elements left, do the remaining
        if ((DSIZE - dind) < gridSize)
        {
            gridSize = DSIZE - dind;
        }

        ++count;

        devDegridKernel <<< gridSize, SSIZE >>>(dGrid, GSIZE, dC, support, dCOffset, dIU, dIV, dData, dind);

        cudaCheckErrors("cuda kernel launch failure");
    }
    cout << "Used " << count << " kernel launches." << endl;

}

template <typename T2>
void DegridderGPU<T2>::degridder()
{
    cout << "\nDegridding on GPU" << endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float tAlloc{ 0.0 }; // in milliseconds
    float tH2D{ 0.0 }; // in milliseconds
    float tKernel{ 0.0 }; // in milliseconds
    float tD2H{ 0.0 }; // in milliseconds
    float tFrees{ 0.0 }; // in milliseconds

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
    cudaEventRecord(start);
    cudaEventSynchronize(start);
    cudaMalloc(&dData, SIZE_DATA);
    cudaMalloc(&dGrid, SIZE_GRID);
    cudaMalloc(&dC, SIZE_C);
    cudaMalloc(&dCOffset, SIZE_COFFSET);
    cudaMalloc(&dIU, SIZE_IU);
    cudaMalloc(&dIV, SIZE_IV);
    cudaCheckErrors("cudaMalloc failure");
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tAlloc, start, stop);

    cudaEventRecord(start);
    cudaEventSynchronize(start);
    cudaMemcpy(dData, data.data(), SIZE_DATA, cudaMemcpyHostToDevice);
    cudaMemcpy(dGrid, gpuGrid.data(), SIZE_GRID, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), SIZE_C, cudaMemcpyHostToDevice);
    cudaMemcpy(dCOffset, cOffset.data(), SIZE_COFFSET, cudaMemcpyHostToDevice);
    cudaMemcpy(dIU, iu.data(), SIZE_IU, cudaMemcpyHostToDevice);
    cudaMemcpy(dIV, iv.data(), SIZE_IV, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tH2D, start, stop);

    // Kernel launch
    cudaEventRecord(start);
    cudaEventSynchronize(start);
    typedef cuComplex Complex;
    degridHelper((const Complex*)dGrid, SSIZE, DSIZE, GSIZE, support,
                 (const Complex*)dC, dCOffset, dIU, dIV, (Complex*)dData);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tKernel, start, stop);

    cudaEventRecord(start);
    cudaEventSynchronize(start);
    cudaMemcpy(data.data(), dData, SIZE_DATA, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tD2H, start, stop);

    // Deallocate device vectors
    cudaEventRecord(start);
    cudaEventSynchronize(start);
    cudaFree(dData);
    cudaFree(dGrid);
    cudaFree(dC);
    cudaFree(dCOffset);
    cudaFree(dIU);
    cudaFree(dIV);
    cudaCheckErrors("cudaFree failure");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tFrees, start, stop);

    cout << "\nDegridderGPU IN MILLISECONDS:" << endl;
    cout << left << setw(21) << "cudaMallocs"
         << left << setw(21) << "cudaMemcpys (H2D)"
         << left << setw(21) << "kernel"
         << left << setw(21) << "cudaMemcpys (D2H)"
         << left << setw(21) << "frees" << endl;;

    cout << left << setw(21) << tAlloc
         << left << setw(21) << tH2D
         << left << setw(21) << tKernel
         << left << setw(21) << tD2H
         << left << setw(21) << tFrees << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

template void DegridderGPU<std::complex<float>>::degridder();
template void DegridderGPU<std::complex<double>>::degridder();
