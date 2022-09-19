#include "DegridderCPU.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T2>
void DegridderCPU<T2>::cpuKernel()
{
   // cout << "Degridding on CPU" << endl;
    const int SSIZE = 2 * support + 1;
    cout << "SSIZE = " << SSIZE << endl;
    cout << "DSIZE = " << DSIZE << endl;
    for (int dind = 0; dind < DSIZE; ++dind)
    {
        data[dind] = 0.0;

        // The actual grid point from which we offset
        int gind = iu[dind] + GSIZE * iv[dind] - support;

        // The convolution function point from which we offset
        int cind = cOffset[dind];

        for (int suppv = 0; suppv < SSIZE; ++suppv)
        {
            T2* d = &data[dind];
            const T2* gptr = &grid[gind];
            const T2* cptr = &C[cind];

            for (int suppu = 0; suppu < SSIZE; ++suppu)
            {
                (*d) += (*(gptr++)) * (*(cptr++));
            }
            gind += GSIZE;
            cind += SSIZE;
        }
    }
}

template void DegridderCPU<std::complex<float>>::cpuKernel();
template void DegridderCPU<std::complex<double>>::cpuKernel();