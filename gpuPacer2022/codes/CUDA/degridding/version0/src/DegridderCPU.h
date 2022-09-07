#pragma once

#include <vector>
#include <iostream>
#include <complex>

template <typename T2>
class DegridderCPU
{
private:
    const std::vector<T2>& grid;
    const size_t DSIZE;
    const size_t GSIZE;
    const size_t support;
    const std::vector<T2>& C;
    const std::vector<int>& cOffset;
    const std::vector<int>& iu;
    const std::vector<int>& iv;
    std::vector<T2>& data;

public:
    DegridderCPU(const std::vector<T2>& grid,
        const size_t DSIZE,
        const size_t GSIZE,
        const size_t support,
        const std::vector<T2>& C,
        const std::vector<int>& cOffset,
        const std::vector<int>& iu,
        const std::vector<int>& iv,
        std::vector<T2>& data) : grid{ grid }, DSIZE{ DSIZE }, GSIZE{ GSIZE }, support{ support }, C{ C },
        cOffset{ cOffset }, iu{ iu }, iv{ iv }, data{ data } {}
    void cpuKernel();
};