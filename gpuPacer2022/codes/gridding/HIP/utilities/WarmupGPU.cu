#include "WarmupGPU.h"

using std::vector;
using std::cout;
using std::endl;

__global__
void vectorAdd(const float* a, const float* b, float* c, const size_t N)
{
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
	{
		for (int j = 0; j < 250; ++j)
		{
			c[i] = a[i] + b[i];
		}
	}
}

void WarmupGPU::warmup() const
{
	vector<float> a(N, 1.0);
	vector<float> b(N, 2.0);
	vector<float> c(N, 0.0);
	vector<float> cAnswer(N, 3.0);

	const size_t SIZE = N * sizeof(float);

	float* dA;
	float* dB;
	float* dC;

	hipMalloc(&dA, SIZE);
	hipMalloc(&dB, SIZE);
	hipMalloc(&dC, SIZE);

	hipMemcpy(dA, a.data(), SIZE, hipMemcpyHostToDevice);
	hipMemcpy(dB, b.data(), SIZE, hipMemcpyHostToDevice);

	const int blockSize = 1024;
	const int gridSize = N / 1024;

	vectorAdd<<<gridSize, blockSize>>> (dA, dB, dC, N);

	hipMemcpy(c.data(), dC, SIZE, hipMemcpyDeviceToHost);

	MaxError<float> maximumError;
	cout << "Verifying warmup launch" << endl;
	maximumError.maxError(c, cAnswer);

	hipFree(dA);
	hipFree(dB);
	hipFree(dC);
}
