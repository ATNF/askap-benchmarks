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

#ifndef HOGBOM_CUDA_H
#define HOGBOM_CUDA_H

// System includes
#include <vector>
#include <cstddef>

// Cuda includes
#include <cuda_runtime_api.h>

struct Peak {
    float val;
    int pos;
};

class HogbomCuda {
    public:
        HogbomCuda(size_t psfSize, size_t residualSize);
        ~HogbomCuda();

        void deconvolve(const std::vector<float>& dirty,
                const size_t dirtyWidth,
                const std::vector<float>& psf,
                const size_t psfWidth,
                std::vector<float>& model,
                std::vector<float>& residual);

    private:

        void reportDevice(void);

        float* d_psf;
        float* d_residual;
        Peak*  d_peaks;     // temporary array for per-block peaks
};

#endif
