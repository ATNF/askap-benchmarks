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

     

        int uIdx = -1;
        int vIdx = -1;
        int tID = 0;
        for (int suppv = 0; suppv < SSIZE; ++suppv)
        {
            ++vIdx;
            //cout << vIdx << endl;
            T2* d = &data[dind];
            const T2* gptr = &grid[gind];
            const T2* cptr = &C[cind];

            for (int suppu = 0; suppu < SSIZE; ++suppu)
            {
                ++uIdx;
                
                tID = uIdx + vIdx * 32;
                (*d) += (*(gptr++)) * (*(cptr++));
                /*if (vIdx == 31 && uIdx == 31)
                {
                    cout << "CPU - tId: " << tID << 
                       ", suppU: " << uIdx <<
                       ", suppV: " << vIdx << 
                        ", sData[tID]: " << grid[gind + suppu] * C[cind + suppu]
                        << endl; 
                }*/

                
            }
            uIdx = -1;

            //
            
            //vIdx += (GSIZE - SSIZE);
            gind += GSIZE;
            cind += SSIZE;
        }
    }
}

template void DegridderCPU<std::complex<float>>::cpuKernel();
template void DegridderCPU<std::complex<double>>::cpuKernel();