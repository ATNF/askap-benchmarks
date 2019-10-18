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

#ifndef HOGBOM_ACC_H
#define HOGBOM_ACC_H

// System includes
#include <vector>
#include <cstddef>

class HogbomACC {
    public:
        HogbomACC();
        ~HogbomACC();

        void deconvolve(const std::vector<float>& dirty,
                const size_t dirtyWidth,
                const std::vector<float>& psf,
                const size_t psfWidth,
                std::vector<float>& model,
                std::vector<float>& residual);

    private:

        struct Position {
            Position(int _x, int _y) : x(_x), y(_y) { };
            int x;
            int y;
        };

        //void findPeak(const std::vector<float>& image,
        //        float& maxVal, size_t& maxPos);
        void findPeak(const float *image,
                float& maxVal, size_t& maxPos, const size_t size);

        //void subtractPSF(const std::vector<float>& psf,
        //        const size_t psfWidth,
        //        std::vector<float>& residual,
        void subtractPSF(const float *psf,
                const size_t psfWidth,
                float *residual,
                const size_t residualWidth,
                const size_t peakPos, const size_t psfPeakPos,
                const float absPeakVal, const float gain);

        Position idxToPos(const int idx, const size_t width);

        size_t posToIdx(const size_t width, const Position& pos);
};

#endif
