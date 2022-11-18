#pragma once

#include "../IGridder.h"

#include <vector>
#include <iostream>
#include <complex>

#include <hip/hip_runtime_api.h>
#include "hip/hip_runtime.h"
//#include "device_launch_parameters.h"
#include <hip/hip_complex.h>

#include <cassert>

typedef hipComplex Complex;

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

class GridderGPUAtomic : public IGridder
{
private:
    // Device vectors
    std::complex<float>* dData;
    std::complex<float>* dGrid;
    std::complex<float>* dC;
    int* dCOffset;
    int* dIU;
    int* dIV;

    // Device parameters
    const size_t SIZE_DATA = data.size() * sizeof(std::complex<float>);
    const size_t SIZE_GRID = grid.size() * sizeof(std::complex<float>);
    const size_t SIZE_C = C.size() * sizeof(std::complex<float>);
    const size_t SIZE_COFFSET = cOffset.size() * sizeof(int);
    const size_t SIZE_IU = iu.size() * sizeof(int);
    const size_t SIZE_IV = iv.size() * sizeof(int);

    // Private methods
    void deviceAllocations();

    // Memory copy from host to device
    void copyH2D();

    friend
        __global__
        void devGridKernelAtomic(
            const Complex* data,
            const int support,
            const Complex* C,
            const int* cOffset,
            const int* iu,
            const int* iv,
            Complex* grid,
            const int GSIZE,
            const int dind);

public:
    GridderGPUAtomic(const size_t support,
        const size_t GSIZE,
        const std::vector<std::complex<float>>& data,
        const std::vector<std::complex<float>>& C,
        const std::vector<int>& cOffset,
        const std::vector<int>& iu,
        const std::vector<int>& iv,
        std::vector<std::complex<float>>& grid) : IGridder(support, GSIZE, data, C, cOffset, iu, iv, grid) {}

    virtual ~GridderGPUAtomic();

    void gridder() override;
};

