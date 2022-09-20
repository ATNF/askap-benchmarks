#pragma once

/* Parameters:
    GSIZE     : size of 1 axis of grid
              : size of grid in pixels

    CELLSIZE  : size of 1 grid cell in wavelengths
    WSIZE     : size of lookup table in w
    NSAMPLES  : number of visibility samples
    
    
    BASELINE  :
    NCHAN     :
*/

#include <complex>

typedef float Real;             // T0
typedef double Coord;           // T1
typedef std::complex<Real> Value;    // T2

// Can be changed for testing purposes
//const int NSAMPLES = 1<<27; //(134 M)
//const int NSAMPLES = 1<<25; //(34 M)
//const int NSAMPLES = 1<<24; //(17 M)
//const int NSAMPLES = 1<<20; //(1 M)
//const int NSAMPLES = 1<<18; //(262k)
const int NSAMPLES = 1<<21;
//const int NSAMPLES = 1;
const int WSIZE = 33;
const int NCHAN = 1;

// Shouldn't be changed
const int GSIZE = 4096;
const Coord CELLSIZE = 5.0;
const int BASELINE = 2000;

const int BLOCK_SIZE = 32; // dim3(BLOCK_SIZE, BLOCK_SIZE)
const int GRID_SIZE = NSAMPLES;
//const int GRID_SIZE = 1;
