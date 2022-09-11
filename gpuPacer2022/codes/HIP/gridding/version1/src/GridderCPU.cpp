#include "GridderCPU.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T2>
void GridderCPU<T2>::gridder()
{
    cout << "\nGridding on CPU" << endl;
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
            T2* gptr = &cpuGrid[gind];
            const T2* cptr = &C[cind];
            const T2 d = data[dind];

            for (int suppU = 0; suppU < SSIZE; ++suppU)
            {
                *(gptr++) += d * (*(cptr++));
            }

            gind += GSIZE;
            cind += SSIZE;
        }
    }
    cout << "Completed..." << endl;
}

template void GridderCPU<std::complex<float>>::gridder();
template void GridderCPU<std::complex<double>>::gridder();