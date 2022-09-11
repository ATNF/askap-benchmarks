#include "Setup.h"

using std::cout;
using std::endl;
using std::vector;
using std::complex;
using std::setw;
using std::left;
using std::setprecision;

template<typename T0, typename T1, typename T2>
void Setup<T0, T1, T2>::initCoord()
{
    // Random vector generator
    RandomVectorGenerator<T1> vectorGenerator;

    vectorGenerator.randomVector(u);
    vectorGenerator.randomVector(v);
    vectorGenerator.randomVector(w);

    for (auto i{ 0 }; i < NSAMPLES; ++i)
    {
        u[i] = BASELINE * u[i] - BASELINE / 2.0;
        v[i] = BASELINE * v[i] - BASELINE / 2.0;
        w[i] = BASELINE * w[i] - BASELINE / 2.0;
    }

    for (auto& i : freq)
    {
        i = (1.4e9 - 2.0e5 * i / NCHAN) / 2.998e8;
    }
    cout << "Coordinate and frequency vectors initiated" << endl;
}

template <typename T0, typename T1, typename T2>
void Setup<T0, T1, T2>::initC()
{
    support = static_cast<int>(1.5 * sqrt(abs(BASELINE)
        * static_cast<T1>(CELLSIZE)
        * freq[0]) / CELLSIZE);

    overSample = 8;
    wCellSize = 2.0 * BASELINE * freq[0] / WSIZE;
    cout << setw(20) << left << "Support    : " << support << " pixels." << endl;
    cout << setw(20) << left << "W cell size: " << wCellSize << " wavelengths.\n" << endl;

    const int SSIZE = 2 * support + 1;
    const int CCENTER = (SSIZE - 1) / 2;

    C.resize(SSIZE * SSIZE * overSample * overSample * WSIZE);
    cout << "Size of the convolution function : " << SSIZE * SSIZE * overSample * overSample * WSIZE * sizeof(T2) / (1024 * 1024) << " MB." << endl;
    cout << "Shape of the convolution function: [" << SSIZE << ", " << SSIZE << ", " << overSample << ", " << overSample << ", " << WSIZE << "]\n" << endl;

    for (auto k = 0; k < WSIZE; ++k)
    {
        double w = static_cast<double>(k - WSIZE / 2);
        double fScale = sqrt(abs(w) * wCellSize * freq[0]) / CELLSIZE;

        for (auto osj = 0; osj < overSample; ++osj)
        {
            for (auto osi = 0; osi < overSample; ++osi)
            {
                for (auto j = 0; j < SSIZE; ++j)
                {
                    double j2 = pow((static_cast<double>(j - CCENTER)
                        + static_cast<double>(osj) / static_cast<double>(overSample)), 2);

                    for (auto i = 0; i < SSIZE; ++i)
                    {
                        double r2 = j2 + pow((static_cast<double>(i - CCENTER)
                            + static_cast<double>(osi) / static_cast<double>(overSample)), 2);
                        long int cind = i + SSIZE * (j + SSIZE
                            * (osi + overSample * (osj + overSample * k)));
                        if (w != 0.0)
                        {
                            C[cind] = static_cast<T2>(cos(r2 / (w * fScale)));
                        }
                        else
                        {
                            C[cind] = static_cast<T2>(exp(-r2));
                        }
                    } // for i 

                } // for j
            } // for osi

        } // for osj

    } // for k

    // Normalize the convolution function
    T0 sumC = 0.0;

    for (const auto& i : C)
    {
        sumC += abs(i);
    }

    for (auto& i : C)
    {
        i *= static_cast<T2>(WSIZE * overSample * overSample / sumC);
    }

    cout << "W-projection convolution vector initiated" << endl;
}

template <typename T0, typename T1, typename T2>
void Setup<T0, T1, T2>::initCOffset()
{
    const int NSAMPLES = u.size();
    const int NCHAN = freq.size();
    const int SSIZE = 2 * support + 1;

    // Calculate the offset for each visibility point
    cOffset.resize(NSAMPLES * NCHAN);
    iu.resize(NSAMPLES * NCHAN);
    iv.resize(NSAMPLES * NCHAN);

    for (auto i = 0; i < NSAMPLES; ++i)
    {
        for (auto chan = 0; chan < NCHAN; ++chan)
        {
            auto dind = i * NCHAN + chan;

            T1 uScaled = freq[chan] * u[i] / CELLSIZE;
            iu[dind] = static_cast<int>(uScaled);

            if (uScaled < static_cast<T1>(iu[dind]))
            {
                iu[dind] -= -1;
            }

            auto fracu = static_cast<int>(overSample * (uScaled - static_cast<T1>(iu[dind])));
            iu[dind] += GSIZE / 2;

            T1 vScaled = freq[chan] * v[i] / CELLSIZE;
            iv[dind] = static_cast<int>(vScaled);


            if (vScaled < static_cast<T1>(iv[dind]))
            {
                iv[dind] -= -1;
            }

            auto fracv = static_cast<int>(overSample * (vScaled - static_cast<T1>(iv[dind])));
            iv[dind] += GSIZE / 2;

            // The beginning of the convolution function for this point
            T1 wScaled = freq[chan] + w[i] / wCellSize;
            auto wOff = WSIZE / 2 + static_cast<int>(wScaled);
            cOffset[dind] = SSIZE * SSIZE * (fracu + overSample * (fracv + overSample * wOff));
        }
    }

    cout << "Convolution offset vector (lookup function) initiated" << endl;
}

template<typename T0, typename T1, typename T2>
void Setup<T0, T1, T2>::setup()
{
    cout << "DEGRIDDING SETUP" << endl;
    initCoord();
    initC();
    initCOffset();
    cout << "DEGRIDDING SETUP COMPLETED" << endl;
}

template void Setup<float, double, complex<float>>::initC();
template void Setup<float, double, complex<float>>::initCOffset();
template void Setup<float, double, complex<float>>::initCoord();
template void Setup<float, double, complex<float>>::setup();
template void Setup<double, double, complex<double>>::initC();
template void Setup<double, double, complex<double>>::initCOffset();
template void Setup<double, double, complex<double>>::initCoord();
template void Setup<double, double, complex<double>>::setup();
template void Setup<float, float, complex<float>>::initC();
template void Setup<float, float, complex<float>>::initCOffset();
template void Setup<float, float, complex<float>>::initCoord();
template void Setup<float, float, complex<float>>::setup();
template void Setup<double, float, complex<double>>::initC();
template void Setup<double, float, complex<double>>::initCOffset();
template void Setup<double, float, complex<double>>::initCoord();
template void Setup<double, float, complex<double>>::setup();

