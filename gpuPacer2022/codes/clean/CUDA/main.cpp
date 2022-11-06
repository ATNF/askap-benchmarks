#include <iostream>
#include <iomanip>
#include <vector>
#include <omp.h>
#include <memory>

#include "utilities/ImageProcess.h"
#include "utilities/Parameters.h"
#include "utilities/MaxError.h"
#include "utilities/WarmupGPU.h"

#include "src/IHogbom.h"
#include "src/SolverFactory.h"

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
	// Maximum error evaluator
	MaxError<float> maximumError;

	// Initiate an image process class
	ImageProcess imagProc;

	// Load dirty image and psf
	float runtimeImagProc = 0.0;
	auto t0 = omp_get_wtime();
	cout << "Reading dirty image & psf image" << endl;
	vector<float> dirty = imagProc.readImage(gDirtyFile);
	vector<float> psf = imagProc.readImage(gPsfFile);
	const size_t DIRTY_DIM = imagProc.checkSquare(dirty);
	const size_t PSF_DIM = imagProc.checkSquare(psf);
	auto t1 = omp_get_wtime();
	runtimeImagProc = t1 - t0;

	if (PSF_DIM != DIRTY_DIM)
	{
		cerr << "Wrong set of PSF and DIRTY images" << endl;
		return -1;
	}
	
	const size_t IMAGE_DIM = DIRTY_DIM;

	// Reports some parameters
	cout << "Iterations: " << gNiters << endl;
	cout << "Image dimension: " << DIRTY_DIM << " x " << DIRTY_DIM << endl;
	WarmupGPU warmupGPU;

	// WARMUP
	if (refSolverName != "Golden")
	{
		// Warmup
		warmupGPU.warmup();
	}
	
	// REFERENCE SOLVER
	vector<float> refResidual(dirty.size(), 0.0);
	vector<float> refModel(dirty.size(), 0.0);
	float runtimeRef = 0.0;
	cout << "\nSolver: " << refSolverName << endl;
	SolverFactory refSolverFactory(dirty, psf, IMAGE_DIM, refModel, refResidual);
	std::shared_ptr<IHogbom> hogbom = refSolverFactory.getSolver(refSolverName);
	t0 = omp_get_wtime();
	hogbom->deconvolve();
	t1 = omp_get_wtime();
	runtimeRef = t1 - t0;
	
	// WARMUP
	if (refSolverName == "Golden")
	{
		// Warmup
		warmupGPU.warmup();
	}

	// NEW SOLVER TEST
	vector<float> testResidual(dirty.size(), 0.0);
	vector<float> testModel(dirty.size(), 0.0);
	float runtimeTest = 0.0;
	cout << "\nSolver: " << testSolverName << endl;
	SolverFactory testSolverFactory(dirty, psf, IMAGE_DIM, testModel, testResidual);
	hogbom = testSolverFactory.getSolver(testSolverName);
	t0 = omp_get_wtime();
	hogbom->deconvolve();
	t1 = omp_get_wtime();
	runtimeTest = t1 - t0;

	cout << "\nVerifying model..." << endl;
	maximumError.maxError(refModel, testModel);
	
	cout << "Verifying residual..." << endl;
	maximumError.maxError(refResidual, testResidual);

	cout << "\nRUNTIME IN SECONDS:" << endl;
	cout << left << setw(21) << refSolverName
		<< left << setw(21) << testSolverName
		<< left << setw(21) << "Speedup" << endl;;

	cout << setprecision(2) << fixed;
	cout << left << setw(21) << runtimeRef
		<< left << setw(21) << runtimeTest
		<< left << setw(21) << runtimeRef / runtimeTest << endl;
		
	/*
	// Write images out
	imagProc.writeImage("residual.img", refResidual);
	imagProc.writeImage("model.img", refModel);
	*/
	
	return 0;
}