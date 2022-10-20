#pragma once

#include "../IHogbom.h"

#include <cmath>
#include <iostream>

class Golden : public IHogbom
{
private:
	struct Position
	{
		Position(int x, int y) : x{ x }, y{ y } {}
		int x;
		int y;
	};

	// Private methods
	Position idxToPos(const int idx, const size_t width);
	size_t posToIdx(const size_t width, const Position& pos);

	void findPeak(const std::vector<float>& image, float& maxVal, size_t& maxPos);
	void subtractPSF(const size_t peakPos,
		const size_t psfPeakPos,
		const float absPeakVal);

public:
	Golden(const std::vector<float>& dirty,
		const std::vector<float>& psf,
		const size_t imageWidth,
		std::vector<float>& model,
		std::vector<float>& residual) : IHogbom(dirty, psf, imageWidth,
		model, residual) {}

	virtual ~Golden() { std::cout << "Golden destructor" << std::endl; }

	// Public methods
	void deconvolve() override;
};