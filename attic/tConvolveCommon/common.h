/// @copyright (c) 2009 CSIRO
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
/// @author Pascal Elahi <pascal.elahi@csiro.au>

#ifndef COMMON_H
#define COMMON_H

// System includes
#include <iostream>
#include <cmath>
#include <ctime>
#include <complex>
#include <vector>
#include <algorithm>
#include <limits>

#ifdef _USEMPI 
#include <mpi.h>
#endif

using std::cout;
using std::endl;
using std::abs;

// Typedefs for easy testing
// Cost of using double for Coord is low, cost for
// double for Real is also low
typedef double Coord;
typedef float Real;
typedef std::complex<Real> Value;

// store options for code
struct Options {
    // Change these if necessary to adjust run time
    unsigned long long nSamples = 160000; // Number of data samples
    int wSize = 33; // Number of lookup planes in w projection
    int nChan = 1; // Number of spectral channels
    int nIterations = 10; // number of interations for bench marking

    // Don't change any of these numbers unless you know what you are doing!
    unsigned int gSize = 4096; // Size of output grid in pixels
    Coord cellSize = 5.0; // Cellsize of output grid in wavelengths
    unsigned int baseline = 2000; // Maximum baseline in meters

};

void configerror(std::string &);
void getinput(int argc, char **argv, struct Options &opt);
void report_timings(const double time, Options &opt, const int sSize, const double griddings);

int verify_result(std::string compname, 
    std::vector<Value> ref, std::vector<Value> comp, 
    double abserr = 1e-6, double relerr = 1e-6);

/////////////////////////////////////////////////////////////////////////////////
// Initialize W project convolution function
// - This is application specific and should not need any changes.
//
// freq - temporal frequency (inverse wavelengths)
// cellSize - size of one grid cell in wavelengths
// support - Total width of convolution function=2*support+1
// wCellSize - size of one w grid cell in wavelengths
// wSize - Size of lookup table in w
void initC(const std::vector<Coord>& freq, const Coord cellSize,
           const Coord baseline,
           const int wSize, int& support, int& overSample,
           Coord& wCellSize, std::vector<Value>& C);

// Initialize Lookup function
// - This is application specific and should not need any changes.
//
// freq - temporal frequency (inverse wavelengths)
// cellSize - size of one grid cell in wavelengths
// gSize - size of grid in pixels (per axis)
// support - Total width of convolution function=2*support+1
// wCellSize - size of one w grid cell in wavelengths
// wSize - Size of lookup table in w
void initCOffset(const std::vector<Coord>& u, const std::vector<Coord>& v,
                 const std::vector<Coord>& w, const std::vector<Coord>& freq,
                 const Coord cellSize, const Coord wCellSize,
                 const int wSize, const int gSize, const int support, const int overSample,
                 std::vector<int>& cOffset, std::vector<int>& iu,
                 std::vector<int>& iv);

// Return a pseudo-random integer in the range 0..2147483647
// Based on an algorithm in Kernighan & Ritchie, "The C Programming Language"
int randomInt();

#endif 