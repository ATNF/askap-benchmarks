#include "DegridderGPU.h"
#include <cmath>
#include "../utilities/MaxError.h"

using std::abs;
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
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, device);

    int count = 0;
    //int gridSize = 1024 * devProp.multiProcessorCount; // is starting size, will be reduced as required
    //int gridSize = 256;
    //cout << "Multi processor count: " << devProp.multiProcessorCount << endl;
    //cout << "support: " << support << endl;
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    
    for (int dind = 0; dind < DSIZE; dind += GRID_SIZE)
    {
        // if there are less than dimGrid elements left,
        // do the remaining
        //if ((DSIZE - dind) < gridSize)
        //{
        //    gridSize = DSIZE - dind;
        //}

        devDegridKernel<64> << < GRID_SIZE, blockSize >> > (dGrid, GSIZE, dC, dCOffset, dIU, dIV, dData, dind);

        ++count;

        /*
        ++count;
        switch (support)
        {
        case 64:
            devDegridKernel<64> << < gridSize, SSIZE >> > (dGrid, GSIZE, dC, dCOffset, dIU, dIV, dData, dind);
            break;
        default:
            assert(0);
        }
        */
        cudaCheckErrors("cuda kernel launch failure");
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
    typedef cuComplex Complex;

    
    degridHelper((const Complex*)dGrid, SSIZE, DSIZE, GSIZE, support, (const Complex*)dC, dCOffset, dIU, dIV, (Complex*)dData);

    
    /*******************************************************************************************************/
    /*******************************************************************************************************/

    cudaMemcpy(data.data(), dData, SIZE_DATA, cudaMemcpyDeviceToHost);
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

template void DegridderGPU<std::complex<float>>::degridder();
template void DegridderGPU<std::complex<double>>::degridder();
