#pragma once

#include "../IHogbom.h"

// CUDA libs
#include <hip/hip_runtime_api.h>
#include "hip/hip_runtime.h"
//#include "device_launch_parameters.h"

#include <cmath>
#include <iostream>

// Error checking macro
#define gpuCheckErrors(msg) \
    do { \
        hipError_t __err = hipGetLastError(); \
        if (__err != hipSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, hipGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


class CudaOlder : public IHogbom
{
private:
	// device vectors
	float* dDirty;
	float* dPsf;
	float* dResidual;

	const size_t SIZE_IMAGE = dirty.size() * sizeof(float);

	void reportDevice();

	struct Position
	{
		__host__ __device__
			Position(int x, int y) : x{ x }, y{ y } {}
		int x;
		int y;
	};

	// Private methods
	__host__ __device__
		static Position idxToPos(const size_t idx, const int width);

	__host__ __device__
		static size_t posToIdx(const int width, const Position& pos);

	
	void subtractPSF(const size_t peakPos,
		const size_t psfPeakPos,
		const float absPeakVal) override;

	void memAlloc();
	void copyH2D();
	void copyD2H();
	
	/*friend
		__global__
		void dFindPeak(const float* image, size_t size, Peak* absPeak);
		*/

	friend
		__global__
		void dSubtractPSF_Older(const float* dPsf,
			float* dResidual,
			const int imageWidth,
			const int startx, const int starty,
			int const stopx, const int stopy,
			const int diffx, const int diffy,
			const float absPeakVal, const float gain);

public:
	CudaOlder(const std::vector<float>& dirty,
		const std::vector<float>& psf,
		const size_t imageWidth,
		std::vector<float>& model,
		std::vector<float>& residual) : IHogbom(dirty, psf, imageWidth,
			model, residual) {}

	virtual ~CudaOlder();

	
	// Public methods
	void deconvolve() override;


};

class naber
{
	
};