#pragma once

#include "hip/hip_runtime.h"
//#include "device_launch_parameters.h"

#include <complex>
#include <iostream>
#include <hip/hip_complex.h>
#include <vector>
#include <cassert>

#include "degridKernelGPU.cuh"

template <typename T2>
class DegridderGPU
{
private:
    const std::vector<T2>& gpuGrid;
    const int SSIZE;
    const int DSIZE;
    const int GSIZE;
    const int support;
    const std::vector<T2>& C;
    const std::vector<int>& cOffset;
    const std::vector<int>& iu;
    const std::vector<int>& iv;
    std::vector<T2>& data;

public:
    DegridderGPU(const std::vector<T2>& gpuGrid,
        const int SSIZE,
        const int DSIZE,
        const int GSIZE,
        const int support,
        const std::vector<T2>& C,
        const std::vector<int>& cOffset,
        const std::vector<int>& iu,
        const std::vector<int>& iv,
        std::vector<T2>& data) : gpuGrid{ gpuGrid }, SSIZE{ SSIZE }, DSIZE{ DSIZE }, GSIZE {GSIZE}, support{support}, C{ C },
        cOffset{ cOffset }, iu{ iu }, iv{ iv }, data{ data } {}
    void degridder();
};
