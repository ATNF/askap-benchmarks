/// @copyright (c) 2011 CSIRO
/// Australia Telescope National Facility (ATNF)
/// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
/// PO Box 76, Epping NSW 1710, Australia
/// atnf-enquiries@csiro.au
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

#ifndef HOGBOM_GOLDEN_H
#define HOGBOM_GOLDEN_H

// System includes
#include <vector>
#include <cstddef>

class HogbomGolden {
    public:

        /// Executes the Hogbom Clean
        ///
        /// @param[in] dirty        a vector containing the dirty image to be cleaned
        /// @param[in] dirtyWidth   the width of the dirty image in pixels (it has to be
        ///                          square so this is also the height)
        /// @param[in] psf          a vector containing the PSF
        /// @param[in] psfWidth     the width of the PSF image in pixels (it has to be
        ///                          square so this is also the height)
        /// @param[inout]   model   the image to which model componenets will be added.
        /// @param[out] residual    the residual image.
        static void deconvolve(const std::vector<float>& dirty,
                               const size_t dirtyWidth,
                               const std::vector<float>& psf,
                               const size_t psfWidth,
                               std::vector<float>& model,
                               std::vector<float>& residual);

    private:

        // Represents a pixel position as x and y coordinates
        struct Position {
            Position(int _x, int _y) : x(_x), y(_y) { };
            int x;
            int y;
        };

        // Finds the peak position and the value at the peak position
        // in an STL vector.
        static void findPeak(const std::vector<float>& image,
                             float& maxVal, size_t& maxPos);

        // Subtracts the PSF from the residual image with PSF pixel "psfPeakPos"
        // aligned with residual pixel "peakPos"
        static void subtractPSF(const std::vector<float>& psf,
                                const size_t psfWidth,
                                std::vector<float>& residual,
                                const size_t residualWidth,
                                const size_t peakPos, const size_t psfPeakPos,
                                const float absPeakVal, const float gain);

        // Converts a 1D array index to a 2D position.
        static Position idxToPos(const int idx, const size_t width);

        // Converts a 2D position to a 1D array index
        static size_t posToIdx(const size_t width, const Position& pos);
};

#endif
