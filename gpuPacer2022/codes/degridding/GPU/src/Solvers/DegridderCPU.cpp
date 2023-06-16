#include "DegridderCPU.h"
#include "../../utilities/LoggerUtil.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T2>
void DegridderCPU<T2>::degridder()
{
    LocalLog() << "Degridding on CPU" << endl;
    const int SSIZE = 2 * this->support + 1;
    LocalLog() << "SSIZE = " << SSIZE << endl;
    for (int dind = 0; dind < this->DSIZE; ++dind)
    {
        this->data[dind] = 0.0;

        // The actual grid point from which we offset
        int gind = this->iu[dind] + GSIZE * this->iv[dind] - this->support;

        // The convolution function point from which we offset
        int cind = this->cOffset[dind];

        for (int suppv = 0; suppv < SSIZE; ++suppv)
        {
            T2* d = &(this->data[dind]);
            const T2* gptr = &(this->grid[gind]);
            const T2* cptr = &(this->C[cind]);

            for (int suppu = 0; suppu < SSIZE; ++suppu)
            {
                (*d) += (*(gptr++)) * (*(cptr++));
            }

            gind += GSIZE;
            cind += SSIZE;
        }
    }
}

template void DegridderCPU<std::complex<float>>::degridder();
template void DegridderCPU<std::complex<double>>::degridder();