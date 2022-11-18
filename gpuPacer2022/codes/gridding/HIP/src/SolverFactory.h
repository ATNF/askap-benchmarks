#pragma once

#include <memory>
#include <string>

#include "IGridder.h"
#include "Solvers/GridderCPU.h"
#include "Solvers/GridderGPUOlder.h"
#include "Solvers/GridderGPUAtomic.h"
#include "Solvers/GridderGPUAtomicTiled.h"

class SolverFactory
{
private:
	const size_t support;
	const size_t GSIZE;
	const std::vector<std::complex<float>>& data;
	const std::vector<std::complex<float>>& C;
	const std::vector<int>& cOffset;
	const std::vector<int>& iu;
	const std::vector<int>& iv;
	std::vector<std::complex<float>>& grid;

	std::shared_ptr<IGridder> solverSelect;

public:
	SolverFactory(const size_t support,
		const size_t GSIZE,
		const std::vector<std::complex<float>>& data,
		const std::vector<std::complex<float>>& C,
		const std::vector<int>& cOffset,
		const std::vector<int>& iu,
		const std::vector<int>& iv,
		std::vector<std::complex<float>>& grid) : support{ support }, GSIZE{ GSIZE }, data{ data }, C{ C },
		cOffset{ cOffset }, iu{ iu }, iv{ iv }, grid{ grid } {}
	
	std::shared_ptr<IGridder> getSolver(std::string solverType)
	{
		if (solverType == "cpu")
		{
			solverSelect = std::make_shared<GridderCPU>(support, GSIZE, data, C, cOffset, iu, iv, grid);
		}
		else if (solverType == "gpuOlder")
		{
			solverSelect = std::make_shared<GridderGPUOlder>(support, GSIZE, data, C, cOffset, iu, iv, grid);
		}
		else if (solverType == "gpuAtomic")
		{
			solverSelect = std::make_shared<GridderGPUAtomic>(support, GSIZE, data, C, cOffset, iu, iv, grid);
		}
		else if (solverType == "gpuAtomicTiled")
		{
			solverSelect = std::make_shared<GridderGPUAtomicTiled>(support, GSIZE, data, C, cOffset, iu, iv, grid);
		}

		return solverSelect;
	}

};