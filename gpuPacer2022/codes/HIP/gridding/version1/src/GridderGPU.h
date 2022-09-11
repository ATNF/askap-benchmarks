#pragma once

#include "hip/hip_runtime.h"
#include "device_launch_parameters.h"

#include <complex>
#include <iostream>
#include <hip/hip_complex.h>
#include <vector>

#include "gridKernelGPU.h"

template <typename T2>
class GridderGPU
{
private:
    const size_t support;
    const size_t GSIZE;
    const std::vector<T2>& data;
    const std::vector<T2>& C;
    const std::vector<int>& cOffset;
    const std::vector<int>& iu;
    const std::vector<int>& iv;
    std::vector<T2>& gpuGrid;

public:
    GridderGPU(const size_t support,
        const size_t GSIZE,
        const std::vector<T2>& data,
        const std::vector<T2>& C,
        const std::vector<int>& cOffset,
        const std::vector<int>& iu,
        const std::vector<int>& iv,
        std::vector<T2>& gpuGrid) : support{ support }, GSIZE{ GSIZE }, data{ data }, C{ C },
        cOffset{ cOffset }, iu{ iu }, iv{ iv }, gpuGrid{ gpuGrid } {}
    void gridder();
};
