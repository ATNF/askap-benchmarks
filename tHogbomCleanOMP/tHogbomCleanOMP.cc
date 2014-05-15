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
/// @detail
///
/// @author Ben Humphreys <ben.humphreys@csiro.au>

// System includes
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstddef>
#include <cmath>
#include <sys/stat.h>

// Local includes
#include "Parameters.h"
#include "Stopwatch.h"
#include "HogbomGolden.h"
#include "HogbomOMP.h"

using namespace std;

vector<float> readImage(const string& filename)
{
    struct stat results;
    if (stat(filename.c_str(), &results) != 0) {
        cerr << "Error: Could not stat " << filename << endl;
        exit(1);
    }

    vector<float> image(results.st_size / sizeof(float));
    ifstream file(filename.c_str(), ios::in | ios::binary);
    file.read(reinterpret_cast<char *>(&image[0]), results.st_size);
    file.close();
    return image;
}

void writeImage(const string& filename, vector<float>& image)
{
    ofstream file(filename.c_str(), ios::out | ios::binary | ios::trunc);
    file.write(reinterpret_cast<char *>(&image[0]), image.size() * sizeof(float));
    file.close();
}

size_t checkSquare(vector<float>& vec)
{
    const size_t size = vec.size();
    const size_t singleDim = sqrt(size);
    if (singleDim * singleDim != size) {
        cerr << "Error: Image is not square" << endl;
        exit(1);
    }

    return singleDim;
}

void zeroInit(vector<float>& vec)
{
    for (vector<float>::size_type i = 0; i < vec.size(); ++i) {
        vec[i] = 0.0;
    }
}

bool compare(const vector<float>& expected, const vector<float>& actual)
{
    if (expected.size() != actual.size()) {
        cout << "Fail (Vector sizes differ)" << endl;
        return false;
    }

    const size_t len = expected.size();
    for (size_t i = 0; i < len; ++i) {
        if (fabs(expected[i] - actual[i]) > 0.00001) {
            cout << "Fail (Expected " << expected[i] << " got "
                << actual[i] << " at index " << i << ")" << endl;
            return false;
        }
    }

    return true;
}

int main(int /*argc*/, char** /* argv*/)
{
    cout << "Reading dirty image and psf image" << endl;
    // Load dirty image and psf
    vector<float> dirty = readImage(g_dirtyFile);
    const size_t dim = checkSquare(dirty);
    vector<float> psf = readImage(g_psfFile);
    const size_t psfDim = checkSquare(psf);

    // Reports some numbers
    cout << "Iterations = " << g_niters << endl;
    cout << "Image dimensions = " << dim << "x" << dim << endl;
    //
    // Run the golden version of the code
    //
    vector<float> goldenResidual;
    vector<float> goldenModel(dirty.size());
    zeroInit(goldenModel);
    {
        // Now we can do the timing for the serial (Golden) CPU implementation
        cout << "+++++ Forward processing (CPU Golden) +++++" << endl;
        HogbomGolden golden;

        Stopwatch sw;
        sw.start();
        golden.deconvolve(dirty, dim, psf, psfDim, goldenModel, goldenResidual);
        const double time = sw.stop();

        // Report on timings
        cout << "    Time " << time << " (s) " << endl;
        cout << "    Time per cycle " << time / g_niters * 1000 << " (ms)" << endl;
        cout << "    Cleaning rate  " << g_niters / time << " (iterations per second)" << endl;
        cout << "Done" << endl;
    }

    // Write images out
    writeImage("residual.img", goldenResidual);
    writeImage("model.img", goldenModel);

    //
    // Run the OpenMP version of the code
    //
    vector<float> ompResidual;
    vector<float> ompModel(dirty.size());
    zeroInit(ompModel);
    {
        // Now we can do the timing for the OpenMP CPU implementation
        cout << "+++++ Forward processing (OpenMP) +++++" << endl;
        HogbomOMP omp;

        Stopwatch sw;
        sw.start();
        omp.deconvolve(dirty, dim, psf, psfDim, ompModel, ompResidual);
        const double time = sw.stop();

        // Report on timings
        cout << "    Time " << time << " (s) " << endl;
        cout << "    Time per cycle " << time / g_niters * 1000 << " (ms)" << endl;
        cout << "    Cleaning rate  " << g_niters / time << " (iterations per second)" << endl;
        cout << "Done" << endl;
    }

    cout << "Verifying model...";
    const bool modelDiff = compare(goldenModel, ompModel);
    if (!modelDiff) {
        return 1;
    } else {
        cout << "Pass" << endl;
    }

    cout << "Verifying residual...";
    const bool residualDiff = compare(goldenResidual, ompResidual);
    if (!residualDiff) {
        return 1;
    } else {
        cout << "Pass" << endl;
    }

    return 0;
}
