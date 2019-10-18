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

// Change these if necessary to adjust run time
const int nSamples = 160000; // Number of data samples
const int wSize = 33; // Number of lookup planes in w projection
const int nChan = 1; // Number of spectral channels

// Don't change any of these numbers unless you know what you are doing!
const int gSize = 4096; // Size of output grid in pixels
const Coord cellSize = 5.0; // Cellsize of output grid in wavelengths
const int baseline = 2000; // Maximum baseline in meters

struct Sample {
    Value data;
    int iu;
    int iv;
    int cOffset;
};

class Benchmark {
    public:
        Benchmark(int rank, int numtasks);

        int randomInt();
        void init();
        bool runGrid(double& time);
        bool runDegrid(double& time);

        void gridKernel(const int support,
                        const std::vector<Value>& C,
                        std::vector<Value>& grid, const int gSize);

        void degridKernel(const std::vector<Value>& grid, const int gSize, const int support,
                          const std::vector<Value>&C, std::vector<Value>& data);

        void initC(const std::vector<Coord>& freq,
                   const Coord cellSize, const int wSize,
                   int& support, int& overSample,
                   Coord& wCellSize, std::vector<Value>& C);

        void initCOffset(const std::vector<Coord>& u, const std::vector<Coord>& v,
                         const std::vector<Coord>& w, const std::vector<Coord>& freq,
                         const Coord cellSize, const Coord wCellSize, const int wSize,
                         const int gSize, const int support, const int overSample);

    private:
        bool shareAndCompare(std::vector<Value>& data);
        void checkError(const int error, const std::string& location);

        int m_rank;
        int m_numtasks;

        std::vector<Value> grid;
        std::vector<Coord> u;
        std::vector<Coord> v;
        std::vector<Coord> w;
        std::vector<Sample> m_samples;
        std::vector<Value> m_outdata;

        std::vector< Value > C;
        int m_support;
        int m_overSample;

        Coord m_wCellSize;

        // For random number generator
        unsigned long m_next;
};
#endif
