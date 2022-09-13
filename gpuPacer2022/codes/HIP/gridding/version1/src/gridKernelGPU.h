#pragma once

#include "CommonGPU.h"
#ifdef __NVCC__
#include "device_launch_parameters.h"
#endif

typedef hipComplex Complex;

__global__
void devGridKernel(
    const Complex* data,
    const int support,
    const Complex* C,
    const int* cOffset,
    const int* iu,
    const int* iv,
    Complex* grid,
    const int GSIZE,
    const int dind);