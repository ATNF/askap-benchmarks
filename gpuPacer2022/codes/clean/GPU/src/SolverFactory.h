#pragma once

#include <memory>
#include <string>

#include "IHogbom.h"
#include "Solvers/GPUPS.h"
#include "Solvers/GPUPSLastWUnrolled.h"
#include "Solvers/GPUPSFullUnroll.h"
#include "Solvers/GPUOlder.h"
#include "Solvers/Golden.h"

class SolverFactory
{
private:
	const std::vector<float>& dirty;
	const std::vector<float>& psf;
	const size_t imageWidth;
	std::vector<float>& model;
	std::vector<float>& residual;

	std::shared_ptr<IHogbom> solverSelect;

public:
	SolverFactory(const std::vector<float>& dirty,
		const std::vector<float>& psf,
		const size_t imageWidth,
		std::vector<float>& model,
		std::vector<float>& residual) : dirty{ dirty }, psf{ psf }, imageWidth{ imageWidth },
		model{ model }, residual{ residual } {}
	std::shared_ptr<IHogbom> getSolver(std::string solverType)
	{
		if (solverType == "Golden")
		{
			solverSelect = std::make_shared<Golden>(dirty, psf, imageWidth,
					model, residual);
		}
		else if (solverType == "gpuOlder")
		{
			solverSelect = std::make_shared<gpuOlder>(dirty, psf, imageWidth,
				model, residual);
		}
		else if (solverType == "gpuPS")
		{
			solverSelect = std::make_shared<gpuPS>(dirty, psf, imageWidth,
				model, residual);
		}
		else if (solverType == "gpuPSLastWUnrolled")
		{
			solverSelect = std::make_shared<gpuPSLastWUnrolled>(dirty, psf, imageWidth,
				model, residual);
		}
		else if (solverType == "gpuPSFullUnroll")
		{
			solverSelect = std::make_shared<gpuPSFullUnroll>(dirty, psf, imageWidth,
				model, residual);
		}
		return solverSelect;
	}

};