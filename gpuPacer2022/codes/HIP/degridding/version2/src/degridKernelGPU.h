#include "hip/hip_runtime.h"
#pragma once

#include "hip/hip_runtime.h"

#ifdef __NVCC__
#include "device_launch_parameters.h"
#endif

#include "hip/hip_complex.h"

typedef hipComplex Complex;

template <int support>
__global__
void devDegridKernel(
    const Complex* grid,
    const int GSIZE,
    const Complex* C,
    const int* cOffset,
    const int* iu,
    const int* iv,
    Complex* data,
    const int dind);
