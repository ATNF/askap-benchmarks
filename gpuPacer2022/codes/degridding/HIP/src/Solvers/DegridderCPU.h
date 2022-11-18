#pragma once

#include "../IDegridder.h"

#include <vector>
#include <iostream>
#include <complex>

class DegridderCPU : public IDegridder
{
private:
    
public:
    DegridderCPU(const std::vector<std::complex<float>>& grid,
        const size_t DSIZE,
        const size_t SSIZE,
        const size_t GSIZE,
        const size_t support,
        const std::vector<std::complex<float>>& C,
        const std::vector<int>& cOffset,
        const std::vector<int>& iu,
        const std::vector<int>& iv,
        std::vector<std::complex<float>>& data) : IDegridder(grid, DSIZE, SSIZE, GSIZE, support, C, cOffset, iu, iv, data) {}
    
    virtual ~DegridderCPU() {}

    void degridder() override;
};