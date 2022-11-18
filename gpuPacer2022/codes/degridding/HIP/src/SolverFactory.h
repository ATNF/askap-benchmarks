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

class SolverFactory
{
private:
	const std::vector<std::complex<float>>& grid;
	const size_t DSIZE;
	const size_t SSIZE;
	const size_t GSIZE;
	const size_t support;
	const std::vector<std::complex<float>>& C;
	const std::vector<int>& cOffset;
	const std::vector<int>& iu;
	const std::vector<int>& iv;
	std::vector<std::complex<float>>& data;

	std::shared_ptr<IDegridder> solverSelect;

public:
	SolverFactory(const std::vector<std::complex<float>>& grid,
		const size_t DSIZE,
		const size_t SSIZE,
		const size_t GSIZE,
		const size_t support,
		const std::vector<std::complex<float>>& C,
		const std::vector<int>& cOffset,
		const std::vector<int>& iu,
		const std::vector<int>& iv,
		std::vector<std::complex<float>>& data) : grid{ grid }, DSIZE{ DSIZE }, SSIZE{ SSIZE }, GSIZE{ GSIZE }, support{ support }, C{ C },
		cOffset{ cOffset }, iu{ iu }, iv{ iv }, data{ data } {}
	
	std::shared_ptr<IDegridder> getSolver(std::string solverType)
	{
		if (solverType == "cpu")
		{
			solverSelect = std::make_shared<DegridderCPU>(grid, DSIZE, SSIZE, GSIZE, support, C, cOffset, iu, iv, data);
		}
		else if (solverType == "gpuInterleaved")
		{
			solverSelect = std::make_shared<DegridderGPUInterleaved>(grid, DSIZE, SSIZE, GSIZE, support, C, cOffset, iu, iv, data);
		}
		else if (solverType == "gpuSequential")
		{
			solverSelect = std::make_shared<DegridderGPUSequential>(grid, DSIZE, SSIZE, GSIZE, support, C, cOffset, iu, iv, data);
		}
		else if (solverType == "gpuLessIdle")
		{
			solverSelect = std::make_shared<DegridderGPULessIdle>(grid, DSIZE, SSIZE, GSIZE, support, C, cOffset, iu, iv, data);
		}
		else if (solverType == "gpuTiled")
		{
			solverSelect = std::make_shared<DegridderGPUTiled>(grid, DSIZE, SSIZE, GSIZE, support, C, cOffset, iu, iv, data);
		}
		else if (solverType == "gpuWarpShuffle")
		{
			solverSelect = std::make_shared<DegridderGPUTiled>(grid, DSIZE, SSIZE, GSIZE, support, C, cOffset, iu, iv, data);
		}
		return solverSelect;
	}

};