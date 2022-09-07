#pragma once

#include <vector>
#include <iostream>
#include <cmath>
#include "../utilities/Parameters.h"

class HogbomGolden
{
private:
	struct Position
	{
		Position(int x, int y) : x{x}, y{y} {}
		int x;
		int y;
	};

	void findPeak(const std::vector<float>& image, float& maxVal, size_t& maxPos);
	void subtractPSF(const std::vector<float>& psf,
		const size_t psfWidth,
		std::vector<float>& residual,
		const size_t residualWidth,
		const size_t peakPos, const size_t psfPeakPos,
		const float absPeakVal, const float gain);

	Position idxToPos(const int idx, const size_t width);
	size_t posToIdx(const size_t width, const Position& pos);

public:
	void deconvolve(const std::vector<float>& dirty,
		const size_t dirtyWidth,
		const std::vector<float>& psf,
		const size_t psfWidth,
		std::vector<float>& model,
		std::vector<float>& residual);

};