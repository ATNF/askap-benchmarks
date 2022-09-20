#include "DegridderGPU.h"

#include <iomanip>

using std::cout;
using std::endl;
using std::vector;
using std::complex;
using std::left;
using std::setw;
using std::fixed;

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

        gpuCheckErrors("cuda kernel launch failure");
    }
    cout << "Used " << count << " kernel launches." << endl;

}

template <typename T2>
void DegridderGPU<T2>::degridder()
{
    cout << "\nDegridding on GPU" << endl;

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
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
    hipEventRecord(start);
    hipEventSynchronize(start);
    gpuErrchk(hipMalloc(&dData, SIZE_DATA));
    gpuErrchk(hipMalloc(&dGrid, SIZE_GRID));
    gpuErrchk(hipMalloc(&dC, SIZE_C));
    gpuErrchk(hipMalloc(&dCOffset, SIZE_COFFSET));
    gpuErrchk(hipMalloc(&dIU, SIZE_IU));
    gpuErrchk(hipMalloc(&dIV, SIZE_IV));
    gpuCheckErrors("hipMalloc failure");
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&tAlloc, start, stop);

    hipEventRecord(start);
    hipEventSynchronize(start);
    gpuErrchk(hipMemcpy(dData, data.data(), SIZE_DATA, hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy(dGrid, gpuGrid.data(), SIZE_GRID, hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy(dC, C.data(), SIZE_C, hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy(dCOffset, cOffset.data(), SIZE_COFFSET, hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy(dIU, iu.data(), SIZE_IU, hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy(dIV, iv.data(), SIZE_IV, hipMemcpyHostToDevice));
    gpuCheckErrors("hipMemcpy H2D failure");
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&tH2D, start, stop);

    // Kernel launch
    hipEventRecord(start);
    hipEventSynchronize(start);
    typedef hipComplex Complex;
    degridHelper((const Complex*)dGrid, SSIZE, DSIZE, GSIZE, support,
                 (const Complex*)dC, dCOffset, dIU, dIV, (Complex*)dData);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&tKernel, start, stop);

    hipEventRecord(start);
    hipEventSynchronize(start);
    gpuErrchk(hipMemcpy(data.data(), dData, SIZE_DATA, hipMemcpyDeviceToHost));
    gpuCheckErrors("hipMemcpy D2H failure");
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&tD2H, start, stop);

    // Deallocate device vectors
    hipEventRecord(start);
    hipEventSynchronize(start);
    gpuErrchk(hipFree(dData));
    gpuErrchk(hipFree(dGrid));
    gpuErrchk(hipFree(dC));
    gpuErrchk(hipFree(dCOffset));
    gpuErrchk(hipFree(dIU));
    gpuErrchk(hipFree(dIV));
    gpuCheckErrors("hipFree failure");
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&tFrees, start, stop);

    cout << "\nDegridderGPU IN MILLISECONDS:" << endl;
    cout << left << setw(21) << "hipMallocs"
         << left << setw(21) << "hipMemcpys (H2D)"
         << left << setw(21) << "kernel"
         << left << setw(21) << "hipMemcpys (D2H)"
         << left << setw(21) << "frees" << endl;;

    cout << left << setw(21) << tAlloc
         << left << setw(21) << tH2D
         << left << setw(21) << tKernel
         << left << setw(21) << tD2H
         << left << setw(21) << tFrees << endl;

    hipEventDestroy(start);
    hipEventDestroy(stop);

}

template void DegridderGPU<std::complex<float>>::degridder();
template void DegridderGPU<std::complex<double>>::degridder();
