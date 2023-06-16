// ****************************************************************************************
// ****************************************************************************************
// ****************************************************************************************
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
/// @author Ben Humphreys <ben.humphreys@csiro.au>
/// @author Tim Cornwell  <tim.cornwell@csiro.au>
// ****************************************************************************************
// ****************************************************************************************
// ****************************************************************************************

/* Parameters:
    data      : values to be gridded in a 1D vector
    support   : total width of the convolution function = 2*support + 1
    C         : convolution function; shape: (2*support + 1, 2*support + 1, *)
    cOffset   : offset into convolution function per data point
    iu, iv    : integer locations of grid points
    grid      : output grid; shape: (gSize, *)
    gSize     : size of 1 axis of grid
              : size of grid in pixels

    freq      : temporal frequency (inverse wavelengths)
    cellSize  : size of 1 grid cell in wavelengths
    wCellSize : size of 1 w grid cell in wavelengths
    wSize     : size of lookup table in w

    nSamples  : number of visibility samples
*/

#include "utilities/MaxError.h"
#include "utilities/Parameters.h"
#include "utilities/PrintVector.h"
#include "utilities/Setup.h"
#include "src/IDegridder.h"
#include "src/SolverFactory.h"
#include "utilities/WarmupGPU.h"
#include "utilities/LoggerUtil.h"

#include <iostream>
#include <complex>
#include <vector>
#include <string>
#include <omp.h>
#include <iomanip>

using std::cout;
using std::endl;
using std::complex;
using std::vector;
using std::polar;
using std::left;
using std::setprecision;
using std::setw;
using std::fixed;


int main()
{
	// report the parellelism and affinity
	LogParallelAPI();
	LogBinding();
    // Print vector object
    PrintVector<Value> printVectorComplex;
    PrintVector<Coord> printVector;

    // Maximum error evaluator
    MaxError<Value> maximumError;

    // Initialize the data to be gridded
    auto timer = NewTimerHostOnly();

    vector<Coord> u(NSAMPLES, 0.0);
    vector<Coord> v(NSAMPLES, 0.0);
    vector<Coord> w(NSAMPLES, 0.0);

    //vector<Value> data(NSAMPLES * NCHAN);
    vector<Value> refData(NSAMPLES * NCHAN, 0.0);
    vector<Value> testData(NSAMPLES * NCHAN, 0.0);
    
    auto timeInitData = timer.get();

    // Measure frequency in inverse wavelengths
    vector<Coord> freq(NCHAN, 0.0);

    // Initialize convolution function & offsets
    vector<Value> C;
    int support;
    int overSample;
    vector<int> cOffset;

    // Vectors of grid centers
    vector<int> iu;
    vector<int> iv;
    Coord wCellSize;

    Setup<Real, Coord, Value> setup(support, overSample, wCellSize, u, v, w, freq, cOffset, iu, iv, C);
    
    timer = NewTimerHostOnly();
    setup.setup();
    auto timeSetup = timer.get();

    const size_t SSIZE = 2 * support + 1;
    const size_t DSIZE = NSAMPLES * NCHAN;
    vector<Value> grid(GSIZE * GSIZE);
    grid.assign(grid.size(), static_cast<Value>(1.0));

    WarmupGPU warmupGPU;

    // WARMUP
    if (refSolverName != "cpu")
    {
        // Warmup
        warmupGPU.warmup();
    }
   
    // Reference degridder
    LocalLog() << "Solver: " << refSolverName << endl;
    SolverFactory<Value> refSolverFactory(grid, DSIZE, SSIZE, GSIZE, support, C, cOffset, iu, iv, refData);
    std::shared_ptr<IDegridder<Value>> refGridder = refSolverFactory.getSolver(refSolverName);
    timer = NewTimerHostOnly();
    refGridder->degridder();
    auto timeDegridRef = timer.get();

    // WARMUP
    if (refSolverName == "cpu")
    {
        // Warmup
        warmupGPU.warmup();
    }

    // Test gridder
    LocalLog() << "Solver: " << testSolverName << endl;
    SolverFactory<Value> testSolverFactory(grid, DSIZE, SSIZE, GSIZE, support, C, cOffset, iu, iv, testData);
    std::shared_ptr<IDegridder<Value>> testGridder = testSolverFactory.getSolver(testSolverName);
    timer = NewTimerHostOnly();
    testGridder->degridder();
    auto timeDegridTest = timer.get();
    
    LocalLog() << "Verifying the code" << endl;
    maximumError.maxError(refData, testData);

    LocalLog() << "RUNTIME IN us" << endl;
    LocalLog() << left << setw(21) << "Setup"
        << left << setw(21) << "Degridding - Ref"
        << left << setw(21) << "Degridding - Test"
        << left << setw(21) << "Speedup" << endl;;

    cout << setprecision(2) << fixed;
    LocalLog() << left << setw(21) << timeSetup
        << left << setw(21) << timeDegridRef
        << left << setw(21) << timeDegridTest 
        << left << setw(21) << timeDegridRef/timeDegridTest << endl;
	return 0;
}