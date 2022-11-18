#pragma once

#include "../IGridder.h"

#include <vector>
#include <iostream>
#include <complex>

class GridderCPU : public IGridder
{
private:
    
public:
    GridderCPU(const size_t support,
        const size_t GSIZE,
        const std::vector<std::complex<float>>& data,
        const std::vector<std::complex<float>>& C,
        const std::vector<int>& cOffset,
        const std::vector<int>& iu,
        const std::vector<int>& iv,
        std::vector<std::complex<float>>& grid) : IGridder(support, GSIZE, data, C, cOffset, iu, iv, grid) {}
    
    virtual ~GridderCPU() {}

    void gridder() override;
};