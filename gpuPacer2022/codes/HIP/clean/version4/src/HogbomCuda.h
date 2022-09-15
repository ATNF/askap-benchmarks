#pragma once

// Cuda includes
#include <hip/hip_runtime_api.h>
#include "hip/hip_runtime.h"
//#include "device_launch_parameters.h"

#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>
#include <stdio.h>

#include "../utilities/Parameters.h"

class HogbomCuda
{
private:
    void reportDevice();

public:
    void deconvolve(const std::vector<float>& dirty,
        const size_t dirtyWidth,
        const std::vector<float>& psf,
        const size_t psfWidth,
        std::vector<float>& model,
        std::vector<float>& residual);
};
