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

// MIC includes
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
        float& maxVal, int& maxPos)
{
#ifdef __MIC__
    maxVal = 0.0;
    int *aligned_maxPos = (int*)_mm_malloc(16*sizeof(int),64);
    maxPos = 0;

    __m512 reduceMaxAbsVal_v = _mm512_setzero_ps();
    __mmask reduceThreadMaxMask;
    __m512i reduceThreadMaxPos_v = _mm512_setzero_pi();
    #pragma omp parallel
    {
        int m=0;
        __m512 threadMaxVal_v = _mm512_setzero_ps();
        __m512i threadMaxPos_v = _mm512_setzero_pi();

        int *seq_1_16 = (int*)_mm_malloc(16*sizeof(int),64);
        for(m=0; m < 16; m++) { seq_1_16[m] = m; }
        __m512 seq_1_16_v = _mm512_load_epi32(seq_1_16, _MM_UPCONV_EPI32_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);

        float *threadAbs_image = (float*)_mm_malloc(16*sizeof(float),64);
        __m512 threadAbs_image_v;
        __mmask threadMaxMask;

        #pragma omp for
        for (int i = 0; i < imageSize; i+=16) {
          const float *inPtr; inPtr = image + i;
          _mm_vprefetch2 (inPtr + 384, _MM_PFHINT_NONE);
          /* Compute absolute of 16 elements in the input vector */
          for(m=0; m < 16; m++) threadAbs_image[m] = fabs(*(inPtr + m));
          _mm_vprefetch1 (inPtr + 256, _MM_PFHINT_NONE);

          threadAbs_image_v = _mm512_loadd(threadAbs_image, _MM_FULLUPC_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
          /* Compute the vector maxima of 16 elements in parallel - across 16 lanes */
          threadMaxVal_v = _mm512_max_ps(threadMaxVal_v,threadAbs_image_v);

          __m512i thread_i_v = _mm512_set_1to16_pi(i);

          /* Determine which locations the maxima occur */
          threadMaxMask = _mm512_cmpeq_ps(threadMaxVal_v,threadAbs_image_v);
          threadMaxPos_v = _mm512_mask_add_pi(threadMaxPos_v,threadMaxMask,thread_i_v,seq_1_16_v);
        }
        #pragma omp critical
        {
          reduceMaxAbsVal_v = _mm512_max_ps(reduceMaxAbsVal_v, threadMaxVal_v);
          reduceThreadMaxMask = _mm512_cmpeq_ps(reduceMaxAbsVal_v,threadMaxVal_v);
          reduceThreadMaxPos_v = _mm512_mask_mov_epi32(reduceThreadMaxPos_v,reduceThreadMaxMask,threadMaxPos_v);
        }

        _mm_free(seq_1_16);
        _mm_free(threadAbs_image);
    }
    float maxAbsVal = _mm512_reduce_max_ps(reduceMaxAbsVal_v);
    __m512 maxAbsVal_v = _mm512_set_1to16_ps(maxAbsVal);
    __mmask maxPosMask = _mm512_cmpeq_ps(reduceMaxAbsVal_v,maxAbsVal_v);
    _mm512_mask_store_epi32(aligned_maxPos,maxPosMask,reduceThreadMaxPos_v,_MM_DOWNCONV_EPI32_NONE,_MM_HINT_NONE);
    maxPos = aligned_maxPos[(int)log2(_mm512_mask2int(maxPosMask))];
    maxVal = image[(int)maxPos];

    _mm_free(aligned_maxPos);
#endif
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

    // Use one less than the number of cores in the card
    omp_set_num_threads_target(TARGET_MIC, 0, 31);

    #pragma offload target(mic) in(dirty:length(dirtySize)) \
            in(psf:length(psfSize))                         \
            inout(model:length(dirtySize))                  \
            inout(residual:length(dirtySize))
    {
        // Find the peak of the PSF
        float psfPeakVal = 0.0;
        int psfPeakPos = 0;
        findPeak(psf, psfSize, psfPeakVal, psfPeakPos);
        printf("Found peak of PSF: Maximum = %.2f at location %d,%d\n",
                psfPeakVal, idxToPos(psfPeakPos, psfWidth).x,
                idxToPos(psfPeakPos, psfWidth).y);
        for (unsigned int i = 0; i < g_niters; ++i) {
            // Find the peak in the residual image
            float absPeakVal = 0.0;
            int absPeakPos = 0;
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
