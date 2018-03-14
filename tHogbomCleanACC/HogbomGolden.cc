/// @copyright (c) 2011 CSIRO
/// Australia Telescope National Facility (ATNF)
/// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
/// PO Box 76, Epping NSW 1710, Australia
/// atnf-enquiries@csiro.au
///
/// This file is part of the ASKAP software distribution.
///
/// The ASKAP software distribution is free software: you can redistribute it
/// and/or modify it under the terms of the GNU General Public License as
/// published by the Free Software Foundation; either version 2 of the License,
/// or (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program; if not, write to the Free Software
/// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
///
/// @author Ben Humphreys <ben.humphreys@csiro.au>

// Include own header file first
#include "HogbomGolden.h"

// System includes
#include <vector>
#include <iostream>
#include <cstddef>
#include <cmath>

// Local includes
#include "Parameters.h"

using namespace std;

HogbomGolden::HogbomGolden()
{
}

HogbomGolden::~HogbomGolden()
{
}

void HogbomGolden::deconvolve(const vector<float>& dirty,
        const size_t dirtyWidth,
        const vector<float>& psf,
        const size_t psfWidth,
        vector<float>& model,
        vector<float>& residual)
{
    residual = dirty;

    // Find the peak of the PSF
    float psfPeakVal = 0.0;
    size_t psfPeakPos = 0;
    findPeak(psf, psfPeakVal, psfPeakPos);
    cout << "Found peak of PSF: " << "Maximum = " << psfPeakVal
        << " at location " << idxToPos(psfPeakPos, psfWidth).x << ","
       << idxToPos(psfPeakPos, psfWidth).y << endl;

    for (unsigned int i = 0; i < g_niters; ++i) {
        // Find the peak in the residual image
        float absPeakVal = 0.0;
        size_t absPeakPos = 0;
        findPeak(residual, absPeakVal, absPeakPos);
        //cout << "Iteration: " << i + 1 << " - Maximum = " << absPeakVal
        //    << " at location " << idxToPos(absPeakPos, dirtyWidth).x << ","
        //    << idxToPos(absPeakPos, dirtyWidth).y << endl;

        // Check if threshold has been reached
        if (abs(absPeakVal) < g_threshold) {
            cout << "Reached stopping threshold" << endl;
            break;
        }

        // Add to model
        model[absPeakPos] += absPeakVal * g_gain;

        // Subtract the PSF from the residual image
        subtractPSF(psf, psfWidth, residual, dirtyWidth, absPeakPos, psfPeakPos, absPeakVal, g_gain);
    }
}

void HogbomGolden::subtractPSF(const vector<float>& psf,
        const size_t psfWidth,
        vector<float>& residual,
        const size_t residualWidth,
        const size_t peakPos, const size_t psfPeakPos,
        const float absPeakVal,
        const float gain)
{
    const int rx = idxToPos(peakPos, residualWidth).x;
    const int ry = idxToPos(peakPos, residualWidth).y;

    const int px = idxToPos(psfPeakPos, psfWidth).x;
    const int py = idxToPos(psfPeakPos, psfWidth).y;

    const int diffx = rx - px;
    const int diffy = ry - px;

    const int startx = max(0, rx - px);
    const int starty = max(0, ry - py);

    const int stopx = min(residualWidth - 1, rx + (psfWidth - px - 1));
    const int stopy = min(residualWidth - 1, ry + (psfWidth - py - 1));

    for (int y = starty; y <= stopy; ++y) {
        for (int x = startx; x <= stopx; ++x) {
            residual[posToIdx(residualWidth, Position(x, y))] -= gain * absPeakVal
                * psf[posToIdx(psfWidth, Position(x - diffx, y - diffy))];
        }
    }
}

void HogbomGolden::findPeak(const vector<float>& image,
                        float& maxVal, size_t& maxPos)
{
    maxVal = 0.0;
    maxPos = 0;
    const size_t size = image.size();
    for (size_t i = 0; i < size; ++i) {
        if (abs(image[i]) > abs(maxVal)) {
            maxVal = image[i];
            maxPos = i;
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
