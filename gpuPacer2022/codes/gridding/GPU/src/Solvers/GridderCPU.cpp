#include "GridderCPU.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T2>
void GridderCPU<T2>::gridder()
{
    const int SSIZE = 2 * this->support + 1;
    const int N = static_cast<int>(this->data.size());

    for (int dind = 0; dind < N; ++dind)
    {
        // The actual grid point
        int gind = this->iu[dind] + GSIZE * this->iv[dind] - this->support;

        // The convolution function point from which we offset
        int cind = this->cOffset[dind];

        for (int suppV = 0; suppV < SSIZE; ++suppV)
        {
            T2* gptr = &(this->grid[gind]);
            const T2* cptr = &(this->C[cind]);
            const T2 d = this->data[dind];

            for (int suppU = 0; suppU < SSIZE; ++suppU)
            {
                *(gptr++) += d * (*(cptr++));
            }

            gind += GSIZE;
            cind += SSIZE;
        }
    }
}

template void GridderCPU<std::complex<float>>::gridder();
template void GridderCPU<std::complex<double>>::gridder();
