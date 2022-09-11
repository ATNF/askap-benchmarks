#pragma once

#include <vector>
#include <iostream>
#include <complex>

template <typename T2>
class GridderCPU
{
private:
    const size_t support;
    const size_t GSIZE;
    const std::vector<T2>& data;
    const std::vector<T2>& C;
    const std::vector<int>& cOffset;
    const std::vector<int>& iu;
    const std::vector<int>& iv;
    std::vector<T2>& cpuGrid;
    
public:
    GridderCPU(const size_t support,
        const size_t GSIZE,
        const std::vector<T2>& data,
        const std::vector<T2>& C,
        const std::vector<int>& cOffset,
        const std::vector<int>& iu,
        const std::vector<int>& iv,
        std::vector<T2>& cpuGrid) : support{ support }, GSIZE{ GSIZE }, data{ data }, C{ C },
        cOffset{ cOffset }, iu{ iu }, iv{ iv }, cpuGrid{ cpuGrid } {}
    void gridder();
};