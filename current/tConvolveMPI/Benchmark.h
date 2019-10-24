/// @copyright (c) 2007 CSIRO
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
/// @author Tim Cornwell  <tim.cornwell@csiro.au>

#ifndef BENCHMARK_H
#define BENCHMARK_H

// System includes
#include <vector>
#include <complex>

// Typedefs
typedef double Coord;
typedef float Real;
typedef std::complex<Real> Value;

class Benchmark {
    public:
        Benchmark();

        int randomInt();
        void init();
        void runGrid();
        void runDegrid();
        //void runGridCheck();
        //void runDegridCheck();

        void gridKernel(const std::vector<Value>& C,
                        std::vector<Value>& grid, const int gSize);

        void degridKernel(const std::vector<Value>& grid, const int gSize,
                          const std::vector<Value>&C, std::vector<Value>& data);

        void initC(const Coord uvCellSize, const int wSize,
                   int& support, int& overSample,
                   Coord& wCellSize, std::vector<Value>& C);

        void initCOffset(const std::vector<Coord>& u, const std::vector<Coord>& v,
                         const std::vector<Coord>& w, const std::vector<Coord>& freq,
                         const Coord uvCellSize, const Coord wCellSize, const int wSize,
                         const int gSize, const int overSample);

        int getSupport() {return m_support;}
        long nVisibilitiesGridded() {return nSamples * nChan;}
        long nPixelsGridded();
        std::vector<float> requiredRate();

        void setMPIrank(const int rank) {mpirank = rank;}
        void setSort(const int type) {doSort = type;}
        void setRunType(const int type) {runType = type;}
        int getRunType() {return runType;}

    private:

        int mpirank;
        int doSort;
        int runType;

        int nSamples; // Number of data samples
        int wSize; // Number of lookup planes in w projection
        int nChan; // Number of spectral channels
        int gSize; // Size of output grid in pixels
        Coord uvCellSize; // Cellsize of output grid in wavelengths
        Real baseline; // Maximum baseline in meters

        std::vector<Value> grid1;
        //std::vector<Value> grid2;
        std::vector<Coord> u;           // [nSamples]
        std::vector<Coord> v;           // [nSamples]
        std::vector<Coord> w;           // [nSamples*nChan]
        std::vector<Value> outdata1;    // [nSamples*nChan]
        std::vector<Value> outdata2;    // [nSamples*nChan]

        std::vector<Value> data;        // [nSamples*nChan]
        std::vector<int> iu;            // [nSamples*nChan]
        std::vector<int> iv;            // [nSamples*nChan]
        std::vector<int> wPlane;        // [nSamples*nChan]
        std::vector<int> cOffset;       // [nSamples*nChan]

        std::vector<Value> C;           // [sum_w(sSize**2)*overSample**2]
        std::vector<int> cOffset0;      // [wSize]
        std::vector<int> sSize;         // [wSize]
        std::vector<int> numPerPlane;   // [wSize]

        int m_support;
        int overSample;

        Coord wCellSize; // Cellsize in the w direction (separation of w-planes) in wavelengths

        // For random number generator
        unsigned long next;
};
#endif
