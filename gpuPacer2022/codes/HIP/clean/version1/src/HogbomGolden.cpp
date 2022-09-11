#include "HogbomGolden.h"
#include <cmath>

using std::cout;
using std::endl;
using std::max;
using std::min;
using std::fabs;

void HogbomGolden::findPeak(const std::vector<float>& image, float& maxVal, size_t& maxPos)
{
	maxVal = 0.0;
	maxPos = 0;
	const size_t SIZE = image.size();

	for (auto i = 0; i < SIZE; ++i)
	{
		if (fabs(image[i]) > fabs(maxVal))
		{
			maxVal = image[i];
			maxPos = i;
		}
	}
}

void HogbomGolden::subtractPSF(const std::vector<float>& psf, const size_t psfWidth, std::vector<float>& residual, const size_t residualWidth, const size_t peakPos, const size_t psfPeakPos, const float absPeakVal, const float gain)
{
	// *****************************************************************************
	// Calling idxToPos twice (may not be important)
	const int rx = idxToPos(peakPos, residualWidth).x;
	const int ry = idxToPos(peakPos, residualWidth).y;

	const int px = idxToPos(psfPeakPos, psfWidth).x;
	const int py = idxToPos(psfPeakPos, psfWidth).y;

	const int diffx = rx - px;
	const int diffy = ry - py;

	// *****************************************************************************
	// Calling max and min twice (may not be important)
	const int startx = max(0, rx - px);
	const int starty = max(0, ry - py);

	const int stopx = min(residualWidth - 1, rx + (psfWidth - px - 1));
	const int stopy = min(residualWidth - 1, ry + (psfWidth - py - 1));

	for (int y = starty; y <= stopy; ++y)
	{
		for (int x = startx; x <= stopx; ++x)
		{
			// *****************************************************************************
			// Check here
			residual[posToIdx(residualWidth, Position(x, y))] -= gain * absPeakVal
				* psf[posToIdx(psfWidth, Position(x - diffx, y - diffy))];
		}
	}
}

HogbomGolden::Position HogbomGolden::idxToPos(const int idx, const size_t width)
{
	const int y = idx / width;
	const int x = idx % width;
	return Position(x, y);
}

size_t HogbomGolden::posToIdx(const size_t width, const HogbomGolden::Position& pos)
{
	return (pos.y * width) + pos.x;
}

void HogbomGolden::deconvolve(const std::vector<float>& dirty, 
	const size_t dirtyWidth, 
	const std::vector<float>& psf, 
	const size_t psfWidth, 
	std::vector<float>& model, 
	std::vector<float>& residual)
{
	residual = dirty;

	// Find the peak of the PSF
	float psfPeakVal = 0.0;
	size_t psfPeakPos = 0;
	findPeak(psf, psfPeakVal, psfPeakPos);
	
	cout << "PSF peak: " << psfPeakVal << ", at location: " << idxToPos(psfPeakPos, psfWidth).x
		<< ", " << idxToPos(psfPeakPos, psfWidth).y << endl;

	for (unsigned int i = 0; i < gNiters; ++i)
	{
		// Find the peak in the residual image
		float absPeakVal = 0.0;
		size_t absPeakPos = 0;
		findPeak(residual, absPeakVal, absPeakPos);

		if ((i + 1) % 100 == 0)
		{
			cout << "Iteration: " << i + 1 << " - Maximum = " << absPeakVal
				<< " at location " << idxToPos(absPeakPos, dirtyWidth).x << ","
				<< idxToPos(absPeakPos, dirtyWidth).y << ", index: " << absPeakPos << endl;
		}

		// Check if the threshold is reached
		if (abs(absPeakVal) < gThreshold) 
		{
			cout << "Reached stopping threshold" << endl;
			break;
		}

		// Add to model
		model[absPeakPos] += absPeakVal * gGain;

		// Subtract the PSF from the residual image 
		subtractPSF(psf, psfWidth, residual, dirtyWidth, absPeakPos, psfPeakPos, absPeakVal, gGain);
	}
}
