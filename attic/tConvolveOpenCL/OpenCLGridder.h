#ifndef OPENCL_GRIDDER_H
#define OPENCL_GRIDDER_H

#include <vector>
#include <complex>

void gridKernelOpenCL(const std::vector< std::complex<float> >& data,
		const int support,
		const std::vector< std::complex<float> >& C,
		const std::vector<int>& cOffset,
		const std::vector<int>& iu,
		const std::vector<int>& iv,
		std::vector< std::complex<float> >& grid,
		const int gSize,
		double &time);

void degridKernelOpenCL(const std::vector< std::complex<float> >& grid,
                const int gSize,
                const int support,
                const std::vector< std::complex<float> >& C,
                const std::vector<int>& cOffset,
                const std::vector<int>& iu,
                const std::vector<int>& iv,
                std::vector< std::complex<float> >& data,
		double &time);

#endif
