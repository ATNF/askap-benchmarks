#pragma once

// Cuda includes
#include "CommonGPU.h"

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
