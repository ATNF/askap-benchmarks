#include "GridderCPU.h"

using std::cout;
using std::endl;
using std::vector;

void GridderCPU::gridder()
{
    const int SSIZE = 2 * support + 1;
    const int N = static_cast<int>(data.size());

    for (int dind = 0; dind < N; ++dind)
    {
        // The actual grid point
        int gind = iu[dind] + GSIZE * iv[dind] - support;

        // The convolution function point from which we offset
        int cind = cOffset[dind];

        for (int suppV = 0; suppV < SSIZE; ++suppV)
        {
            std::complex<float>* gptr = &grid[gind];
            const std::complex<float>* cptr = &C[cind];
            const std::complex<float> d = data[dind];

            for (int suppU = 0; suppU < SSIZE; ++suppU)
            {
                *(gptr++) += d * (*(cptr++));
            }

            gind += GSIZE;
            cind += SSIZE;
        }
    }
}
