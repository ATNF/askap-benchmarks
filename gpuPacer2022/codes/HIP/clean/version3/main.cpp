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

#include <iostream>
#include <iomanip>
#include <vector>
#include <omp.h>

#include "utilities/ImageProcess.h"
#include "utilities/RunInfo.h"
#include "utilities/Parameters.h"
#include "utilities/MaxError.h"

#include "src/HogbomGolden.h"
#include "src/HogbomCuda.h"

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::left;
using std::setprecision;
using std::setw;
using std::fixed;

int main()
{
	RunInfo r;
	r.GetInfo();

	// Maximum error evaluator
	MaxError<float> maximumError;

	// Initiate an image process class
	ImageProcess imagProc;

	// Load dirty image and psf
	cout << "Reading dirty image & psf image" << endl;
	vector<float> dirty = imagProc.readImage(gDirtyFile);
	vector<float> psf = imagProc.readImage(gPsfFile);
	const size_t DIRTY_DIM = imagProc.checkSquare(dirty);
	const size_t PSF_DIM = imagProc.checkSquare(psf);
	if (PSF_DIM != DIRTY_DIM)
	{
		cerr << "Wrong set of PSF and DIRTY images" << endl;
		return -1;
	}

	// Reports some parameters
	cout << "Iterations: " << gNiters << endl;
	cout << "Image dimension: " << DIRTY_DIM << " x " << DIRTY_DIM << endl;
	
	//==================================================================================
	//==================================================================================
	// GOLDEN (SERIAL) VERSION
	vector<float> goldenResidual;
	vector<float> goldenModel(dirty.size(), 0.0);
	float runtimeGolden = 0.0;
	{
		auto t0 = omp_get_wtime();
		cout << "Forward Processing - CPU Golden" << endl;
		HogbomGolden golden;

		golden.deconvolve(dirty, DIRTY_DIM, psf, PSF_DIM, goldenModel, goldenResidual);
		auto t1 = omp_get_wtime();
		runtimeGolden = t1 - t0;

		// Report on timings
//		cout << "Time " << runtimeGolden << " (s) " << endl;
//		cout << "Time per cycle " << runtimeGolden / gNiters * 1000 << " (ms)" << endl;
//		cout << "Cleaning rate  " << gNiters / runtimeGolden << " (iterations per second)" << endl;
//		cout << "Done" << endl;
	}

	// Write images out
//	imagProc.writeImage("residual.img", goldenResidual);
//	imagProc.writeImage("model.img", goldenModel);

	//==================================================================================
	//==================================================================================
	// CUDA (GPU) VERSION
	vector<float> gpuResidual(dirty.size());
	vector<float> gpuModel(dirty.size(), 0.0);
	float runtimeGPU = 0.0;
	{
		auto t0 = omp_get_wtime();
		cout << "Forward Processing - GPU" << endl;
		HogbomCuda gpu;

		gpu.deconvolve(dirty, DIRTY_DIM, psf, PSF_DIM, gpuModel, gpuResidual);
		
		auto t1 = omp_get_wtime();
		runtimeGPU = t1 - t0;
		// Report on timings
//		cout << "Time " << runtimeGPU << " (s) " << endl;
//		cout << "Time per cycle " << runtimeGPU / gNiters * 1000 << " (ms)" << endl;
//		cout << "Cleaning rate  " << gNiters / runtimeGPU << " (iterations per second)" << endl;
//		cout << "Done" << endl;
	}

	cout << "Verifying model..." << endl;
	maximumError.maxError(goldenModel, gpuModel);
	
	cout << "Verifying residual..." << endl;
	maximumError.maxError(goldenResidual, gpuResidual);

	cout << "\nRUNTIME IN SECONDS:" << endl;
	cout << left << setw(21) << "CLEAN CPU"
		<< left << setw(21) << "CLEAN GPU"
		<< left << setw(21) << "Speedup" << endl;;

	cout << setprecision(2) << fixed;
	cout << left << setw(21) << runtimeGolden
		<< left << setw(21) << runtimeGPU
		<< left << setw(21) << runtimeGolden / runtimeGPU << endl;

	return 0;
}
