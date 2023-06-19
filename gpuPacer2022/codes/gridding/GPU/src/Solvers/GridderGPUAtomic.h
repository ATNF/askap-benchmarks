#pragma once

#include "../IGridder.h"
#include "../../utilities/gpuCommon.h"
#include "../../utilities/LoggerUtil.h"

#include <vector>
#include <iostream>
#include <complex>
#include <cassert>

template <typename T2>
class GridderGPUAtomic : public IGridder<T2>
{
private:
    // Device vectors
    T2* dData;
    T2* dGrid;
    T2* dC;
    int* dCOffset;
    int* dIU;
    int* dIV;

    // Device parameters
    const size_t SIZE_DATA = this->data.size() * sizeof(T2);
    const size_t SIZE_GRID = this->grid.size() * sizeof(T2);
    const size_t SIZE_C = this->C.size() * sizeof(T2);
    const size_t SIZE_COFFSET = this->cOffset.size() * sizeof(int);
    const size_t SIZE_IU = this->iu.size() * sizeof(int);
    const size_t SIZE_IV = this->iv.size() * sizeof(int);

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
        const std::vector<T2>& data,
        const std::vector<T2>& C,
        const std::vector<int>& cOffset,
        const std::vector<int>& iu,
        const std::vector<int>& iv,
        std::vector<T2>& grid) : IGridder<T2>(support, GSIZE, data, C, cOffset, iu, iv, grid) {}

    virtual ~GridderGPUAtomic();

    void gridder() override;
};

