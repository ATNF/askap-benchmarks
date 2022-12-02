#pragma once

#include <memory>
#include <string>

#include "IDegridder.h"
#include "Solvers/DegridderCPU.h"
#include "Solvers/DegridderGPUInterleaved.h"
#include "Solvers/DegridderGPUSequential.h"
#include "Solvers/DegridderGPULessIdle.h"
#include "Solvers/DegridderGPUTiled.h"
#include "Solvers/DegridderGPUWarpShuffle.h"

template <typename T2>
class SolverFactory
{
private:
	const std::vector<T2>& grid;
	const size_t DSIZE;
	const size_t SSIZE;
	const size_t GSIZE;
	const size_t support;
	const std::vector<T2>& C;
	const std::vector<int>& cOffset;
	const std::vector<int>& iu;
	const std::vector<int>& iv;
	std::vector<T2>& data;

	std::shared_ptr<IDegridder<T2>> solverSelect;

public:
	SolverFactory(const std::vector<T2>& grid,
		const size_t DSIZE,
		const size_t SSIZE,
		const size_t GSIZE,
		const size_t support,
		const std::vector<T2>& C,
		const std::vector<int>& cOffset,
		const std::vector<int>& iu,
		const std::vector<int>& iv,
		std::vector<T2>& data) : grid{ grid }, DSIZE{ DSIZE }, SSIZE{ SSIZE }, GSIZE{ GSIZE }, support{ support }, C{ C },
		cOffset{ cOffset }, iu{ iu }, iv{ iv }, data{ data } {}
	
	std::shared_ptr<IDegridder<T2>> getSolver(std::string solverType)
	{
		if (solverType == "cpu")
		{
			solverSelect = std::make_shared<DegridderCPU<T2>>(grid, DSIZE, SSIZE, GSIZE, support, C, cOffset, iu, iv, data);
		}
		else if (solverType == "gpuInterleaved")
		{
			solverSelect = std::make_shared<DegridderGPUInterleaved<T2>>(grid, DSIZE, SSIZE, GSIZE, support, C, cOffset, iu, iv, data);
		}
		else if (solverType == "gpuSequential")
		{
			solverSelect = std::make_shared<DegridderGPUSequential<T2>>(grid, DSIZE, SSIZE, GSIZE, support, C, cOffset, iu, iv, data);
		}
		else if (solverType == "gpuLessIdle")
		{
			solverSelect = std::make_shared<DegridderGPULessIdle<T2>>(grid, DSIZE, SSIZE, GSIZE, support, C, cOffset, iu, iv, data);
		}
		else if (solverType == "gpuTiled")
		{
			solverSelect = std::make_shared<DegridderGPUTiled<T2>>(grid, DSIZE, SSIZE, GSIZE, support, C, cOffset, iu, iv, data);
		}
		else if (solverType == "gpuWarpShuffle")
		{
			solverSelect = std::make_shared<DegridderGPUWarpShuffle<T2>>(grid, DSIZE, SSIZE, GSIZE, support, C, cOffset, iu, iv, data);
		}
		return solverSelect;
	}

};