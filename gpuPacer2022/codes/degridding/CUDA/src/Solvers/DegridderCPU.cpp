#include "DegridderCPU.h"

using std::cout;
using std::endl;
using std::vector;


void DegridderCPU::degridder()
{
    cout << "Degridding on CPU" << endl;
    const int SSIZE = 2 * support + 1;
    cout << "SSIZE = " << SSIZE << endl;
    for (int dind = 0; dind < DSIZE; ++dind)
    {
        data[dind] = 0.0;

        // The actual grid point from which we offset
        int gind = iu[dind] + GSIZE * iv[dind] - support;

        // The convolution function point from which we offset
        int cind = cOffset[dind];

        for (int suppv = 0; suppv < SSIZE; ++suppv)
        {
            std::complex<float>* d = &data[dind];
            const std::complex<float>* gptr = &grid[gind];
            const std::complex<float>* cptr = &C[cind];

            for (int suppu = 0; suppu < SSIZE; ++suppu)
            {
                (*d) += (*(gptr++)) * (*(cptr++));
            }

            gind += GSIZE;
            cind += SSIZE;
        }
    }
}
