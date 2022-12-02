#pragma once

#include "../IDegridder.h"

#include <vector>
#include <iostream>
#include <complex>

template <typename T2>
class DegridderCPU : public IDegridder<T2>
{
private:
    
public:
    DegridderCPU(const std::vector<T2>& grid,
        const size_t DSIZE,
        const size_t SSIZE,
        const size_t GSIZE,
        const size_t support,
        const std::vector<T2>& C,
        const std::vector<int>& cOffset,
        const std::vector<int>& iu,
        const std::vector<int>& iv,
        std::vector<T2>& data) : IDegridder<T2>(grid, DSIZE, SSIZE, GSIZE, support, C, cOffset, iu, iv, data) {}
    
    virtual ~DegridderCPU() {}

    void degridder() override;
};