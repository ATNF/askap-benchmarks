#pragma once

#include "../IGridder.h"

#include <vector>
#include <iostream>
#include <complex>

template <typename T2>
class GridderCPU : public IGridder<T2>
{
private:
    
public:
    GridderCPU(const size_t support,
        const size_t GSIZE,
        const std::vector<T2>& data,
        const std::vector<T2>& C,
        const std::vector<int>& cOffset,
        const std::vector<int>& iu,
        const std::vector<int>& iv,
        std::vector<T2>& grid) : IGridder<T2>(support, GSIZE, data, C, cOffset, iu, iv, grid) {}
    
    virtual ~GridderCPU() {}

    void gridder() override;
};