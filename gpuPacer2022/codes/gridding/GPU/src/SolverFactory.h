#pragma once

#include <memory>
#include <string>

#include "IGridder.h"
#include "Solvers/GridderCPU.h"
#include "Solvers/GridderGPUOlder.h"
#include "Solvers/GridderGPUAtomic.h"
#include "Solvers/GridderGPUAtomicTiled.h"

template <typename T2>
class SolverFactory
{
private:
	const size_t support;
	const size_t GSIZE;
	const std::vector<T2>& data;
	const std::vector<T2>& C;
	const std::vector<int>& cOffset;
	const std::vector<int>& iu;
	const std::vector<int>& iv;
	std::vector<T2>& grid;

	std::shared_ptr<IGridder<T2>> solverSelect;

public:
	SolverFactory(const size_t support,
		const size_t GSIZE,
		const std::vector<T2>& data,
		const std::vector<T2>& C,
		const std::vector<int>& cOffset,
		const std::vector<int>& iu,
		const std::vector<int>& iv,
		std::vector<T2>& grid) : support{ support }, GSIZE{ GSIZE }, data{ data }, C{ C },
		cOffset{ cOffset }, iu{ iu }, iv{ iv }, grid{ grid } {}
	
	std::shared_ptr<IGridder<T2>> getSolver(std::string solverType)
	{
		if (solverType == "cpu")
		{
			solverSelect = std::make_shared<GridderCPU<T2>>(support, GSIZE, data, C, cOffset, iu, iv, grid);
		}
		else if (solverType == "gpuOlder")
		{
			solverSelect = std::make_shared<GridderGPUOlder<T2>>(support, GSIZE, data, C, cOffset, iu, iv, grid);
		}
		else if (solverType == "gpuAtomic")
		{
			solverSelect = std::make_shared<GridderGPUAtomic<T2>>(support, GSIZE, data, C, cOffset, iu, iv, grid);
		}
		else if (solverType == "gpuAtomicTiled")
		{
			solverSelect = std::make_shared<GridderGPUAtomicTiled<T2>>(support, GSIZE, data, C, cOffset, iu, iv, grid);
		}

		return solverSelect;
	}

};