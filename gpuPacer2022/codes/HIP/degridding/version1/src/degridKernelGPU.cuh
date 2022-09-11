#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuComplex.h"

typedef cuComplex Complex;

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