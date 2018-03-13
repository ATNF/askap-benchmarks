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
#include "HogbomMIC.h"

// System includes
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <offload.h>
#include <immintrin.h>

// Local includes
#include "Parameters.h"

using namespace std;

struct Position {
    inline Position(int _x, int _y) : x(_x), y(_y) { };
    int x;
    int y;
};

__declspec(target(mic))
static inline Position idxToPos(const int idx, const size_t width)
{
    const int y = idx / width;
    const int x = idx % width;
    return Position(x, y);
}

__declspec(target(mic))
static inline size_t posToIdx(const size_t width, const Position& pos)
{
    return (pos.y * width) + pos.x;
}

__declspec(target(mic))
static void subtractPSF(const float* psf,
        const size_t psfWidth,
        float* residual,
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
    const float MUL = gain * absPeakVal;

    int lhsIdx;
    int rhsIdx;

    #pragma omp parallel for default(shared) private(lhsIdx,rhsIdx)
    for (int y = starty; y <= stopy; ++y) {
        lhsIdx = y * residualWidth + startx;
        rhsIdx = (y - diffy) * psfWidth + (startx - diffx);
        for (int x = startx; x <= stopx; ++x, lhsIdx++, rhsIdx++) {
            residual[lhsIdx] -= MUL * psf[rhsIdx];
        }
    }
}

__declspec(target(mic))
static void findPeak(const float* image, const size_t imageSize,
        float& maxVal, size_t& maxPos)
{
    maxVal = 0.0;
    maxPos = 0;
    #pragma omp parallel
    {
        float threadAbsMaxVal = 0.0;
        size_t threadAbsMaxPos = 0;
        #pragma omp for
        for (size_t i = 0; i < imageSize; ++i) {
#ifdef __MIC__
            _mm_vprefetch2 (&image[i + 1024], _MM_PFHINT_NT);
            if (fabsf(image[i]) > fabsf(threadAbsMaxVal)) {
                threadAbsMaxVal = image[i];
                threadAbsMaxPos = i;
            }
            _mm_vprefetch1 (&image[i + 512], _MM_PFHINT_NT);
#endif
        }

        // Avoid using the double-checked locking optimization here unless
        // we can be certain that the load of a float is atomic
        #pragma omp critical
        if (fabsf(threadAbsMaxVal) > fabsf(maxVal)) {
            maxVal = threadAbsMaxVal;
            maxPos = threadAbsMaxPos;
        }
    }
}


void mic_deconvolve(const float* dirty,
        const size_t dirtyWidth,
        const float* psf,
        const size_t psfWidth,
        float* model,
        float* residual)
{
    const int dirtySize = dirtyWidth*dirtyWidth;
    const int psfSize = psfWidth*psfWidth;
    memcpy(residual, dirty, dirtySize * sizeof(float));

    #pragma offload target(mic) in(dirty:length(dirtySize)) \
            in(psf:length(psfSize))                         \
            inout(model:length(dirtySize))                  \
            inout(residual:length(dirtySize))
    {
        // Find the peak of the PSF
        float psfPeakVal = 0.0;
        size_t psfPeakPos = 0;
        findPeak(psf, psfSize, psfPeakVal, psfPeakPos);
        printf("Found peak of PSF: Maximum = %.2f at location %d,%d\n",
                psfPeakVal, idxToPos(psfPeakPos, psfWidth).x,
                idxToPos(psfPeakPos, psfWidth).y);
        for (unsigned int i = 0; i < g_niters; ++i) {
            // Find the peak in the residual image
            float absPeakVal = 0.0;
            size_t absPeakPos = 0;
            findPeak(residual, dirtySize, absPeakVal, absPeakPos);

            // Check if threshold has been reached
            if (fabsf(absPeakVal) < g_threshold) {
                printf("Reached stopping threshold\n");
                break;
            }

            // Add to model
            model[absPeakPos] += absPeakVal * g_gain;

            // Subtract the PSF from the residual image
            subtractPSF(psf, psfWidth, residual, dirtyWidth, absPeakPos, psfPeakPos, absPeakVal, g_gain);
        }
    } // End pragma offload
}
