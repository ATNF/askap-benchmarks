#pragma once

#include "CommonGPU.h"

typedef hipComplex Complex;

__global__
void devDegridKernel(
    const Complex* grid,
    const int GSIZE,
    const Complex* C,
    const int support,
    const int* cOffset,
    const int* iu,
    const int* iv,
    Complex* data,
    const int dind);