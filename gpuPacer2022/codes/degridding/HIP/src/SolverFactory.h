#pragma once

#include <memory>
#include <string>

#include "IHogbom.h"
#include "Solvers/CudaPS.h"
#include "Solvers/CudaPSLastWUnrolled.h"
#include "Solvers/CudaPSFullUnroll.h"
#include "Solvers/CudaOlder.h"
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
		else if (solverType == "CudaOlder")
		{
			solverSelect = std::make_shared<CudaOlder>(dirty, psf, imageWidth,
				model, residual);
		}
		else if (solverType == "CudaPS")
		{
			solverSelect = std::make_shared<CudaPS>(dirty, psf, imageWidth,
				model, residual);
		}
		else if (solverType == "CudaPSLastWUnrolled")
		{
			solverSelect = std::make_shared<CudaPSLastWUnrolled>(dirty, psf, imageWidth,
				model, residual);
		}
		else if (solverType == "CudaPSFullUnroll")
		{
			solverSelect = std::make_shared<CudaPSFullUnroll>(dirty, psf, imageWidth,
				model, residual);
		}
		return solverSelect;
	}

};