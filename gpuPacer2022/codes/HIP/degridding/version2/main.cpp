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

#include "src/Setup.h"
#include "src/DegridderCPU.h"
#include "src/DegridderGPU.h"

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
    // Print vector object
    PrintVector<Value> printVectorComplex;
    PrintVector<Coord> printVector;

    // Maximum error evaluator
    MaxError<Value> maximumError;

    // Initialize the data to be gridded
    //auto tInit = omp_get_wtime();

    vector<Coord> u(NSAMPLES, 0.0);
    vector<Coord> v(NSAMPLES, 0.0);
    vector<Coord> w(NSAMPLES, 0.0);

    //vector<Value> data(NSAMPLES * NCHAN);
    vector<Value> data(NSAMPLES * NCHAN, 1.0);
    vector<Value> cpuOutData(NSAMPLES * NCHAN, 0.0);
    vector<Value> gpuOutData(NSAMPLES * NCHAN, 0.0);
    
    //auto tFin = omp_get_wtime();
    //auto timeInitData = (tFin - tInit) * 1000.0; // in ms

    // Measure frequency in inverse wavelengths
    vector<Coord> freq(NCHAN, 0.0);

    // Initialize convolution function & offsets
    vector<Value> C;
    int support;
    int overSample;
    vector<int> cOffset;

    cout << "Data size: " << NSAMPLES << endl;

    // Vectors of grid centers
    vector<int> iu;
    vector<int> iv;
    Coord wCellSize;

    Setup<Real, Coord, Value> setup(support, overSample, wCellSize, u, v, w, freq, cOffset, iu, iv, C);

    //tInit = omp_get_wtime();
    setup.setup();
    //tFin = omp_get_wtime();
    //auto timeSetup = (tFin - tInit) * 1000.0; // in ms
    
    const int SSIZE = 2 * support + 1;
    vector<Value> cpuGrid(GSIZE * GSIZE);
    cpuGrid.assign(cpuGrid.size(), static_cast<Value>(1.0));
    vector<Value> gpuGrid(GSIZE * GSIZE);
    gpuGrid.assign(gpuGrid.size(), static_cast<Value>(1.0));

    // Degridding on CPU
    const size_t DSIZE = data.size();
    /*DegridderCPU<Value> degridderCPU(cpuGrid, DSIZE, GSIZE, support, C, cOffset, iu, iv, cpuOutData);
    tInit = omp_get_wtime();
    degridderCPU.cpuKernel();
    tFin = omp_get_wtime();
    auto timeDegridCPU = (tFin - tInit) * 1000.0; // in ms
    */

    // Degridding on GPU
    DegridderGPU<Value> degridderGPU(gpuGrid, SSIZE, DSIZE, GSIZE, support, C, cOffset, iu, iv, gpuOutData);
    auto tInit = omp_get_wtime();
    degridderGPU.degridder();
    auto tFin = omp_get_wtime();
    auto timeDegridGPU = (tFin - tInit) * 1000.0; // in ms

    //cout << "Verify the code" << endl;
    //maximumError.maxError(cpuOutData, gpuOutData);
    
    
    cout << "\nRUNTIME IN MILLISECONDS:" << endl;
    //cout << left << setw(21) << "Setup"
      //  << left << setw(21) << "Gridding CPU"
       cout  << left << setw(21) << "Gridding GPU" << endl;
        //<< left << setw(21) << "Speedup" << endl;

    cout << setprecision(2) << fixed;
    
	// cout << left << setw(21) << timeSetup
       //  << left << setw(21) << timeDegridCPU
       cout << left << setw(21) << timeDegridGPU << endl;
       //  << left << setw(21) << timeDegridCPU/timeDegridGPU << endl;
      
}
