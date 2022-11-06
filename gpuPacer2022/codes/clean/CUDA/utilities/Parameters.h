#pragma once

#include <string>
#include <iostream>

static const int IMAGE_WIDTH = 4096;

static const std::string gDirtyFile = "data/dirty_" + std::to_string(IMAGE_WIDTH) + ".img";
static const std::string gPsfFile = "data/psf_" + std::to_string(IMAGE_WIDTH) + ".img";

static const size_t gNiters = 1000;
static const float gGain = 0.1;
static const float gThreshold = 0.00001;

static const int BLOCK_SIZE = 128; // CUDA maximum is 1024
static const int GRID_SIZE = 512;

// Solver selection
//static const std::string refSolverName = "Golden";
static const std::string refSolverName = "CudaOlder";
//static const std::string refSolverName = "CudaPS";
//static const std::string refSolverName = "CudaPSFullUnroll";
// static const std::string testSolverName = "CudaPS";
static const std::string testSolverName = "CudaPSFullUnroll";

/*
	Solvers explanation:
	- Golden: CPU solver
	- CudaOlder: Cuda - Solver from the previous hackathon, uses shared memory, standard find max
	- CudaPS: Cuda - parallel sweep
	- CudaPSLastWUnrolled: Cuda - parallel sweep, last warp unrolled
	- CudaPSFullUnroll: Cuda - parallel sweep, full unroll 
*/


