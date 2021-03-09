/// @copyright (c) 2009 CSIRO
/// Australia Telescope National Facility (ATNF)
/// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
/// PO Box 76, Epping NSW 1710, Australia
/// atnf-enquiries@csiro.au
///
/// This file is part of the ASKAP software distribution.
///
/// The ASKAP software distribution is free software: you can redistribute it
/// and/or modify it under the terms of the GNU General Public License as
/// published by the Free Software Foundation; either version 2 of the License,
/// or (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program; if not, write to the Free Software
/// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
///
/// @author Pascal Elahi <pascal.elahi@csiro.au>

#include "common.h"

void configerror(std::string &s) {
    std::cerr<<"Error: "<<s<<endl;
#ifdef _USEMPI 
    MPI_Abort(0,MPI_COMM_WORLD);
#else 
    exit(0);
#endif
}

/// UI
void getinput(int argc, char **argv, struct Options &opt)
{
    std::string s; 
    if(argc != 5){
        s = "Usage: "+std::string(argv[0]);
        s += " <number of samples ["+std::to_string(opt.nSamples)+"]>";
        s += " <number of look up planes ["+std::to_string(opt.wSize)+"]>";
        s += " <number of spectral channels ["+std::to_string(opt.nChan)+"]>";
        s += " <number of iterations to run for bench marking ["+std::to_string(opt.nIterations)+"]> ";
        std::cerr<<s<<endl;
#ifdef _USEMPI 
        MPI_Abort(0,MPI_COMM_WORLD);
#else 
        exit(0);
#endif
    }
    opt.nSamples = atoll(argv[1]);
    opt.wSize = atoi(argv[2]);
    opt.nChan = atoi(argv[3]);
    opt.nIterations = atoi(argv[4]);
    //check if options acceptable 
    if (opt.nSamples <=0) {
        s = "Number of samples <= 0, is "+std::to_string(opt.nSamples);
        configerror(s);
    }
    if (opt.wSize <=0) {
        s = "Number of look-up planes <= 0, is "+std::to_string(opt.wSize);
        configerror(s);
    }
    if (opt.nChan <=0) {
        s = "Number of spectral channels <= 0, is "+std::to_string(opt.nChan);
        configerror(s);
    }
    if (opt.nIterations <=0) {
        s = "Number of iterations <= 0, is "+std::to_string(opt.nIterations);
        configerror(s);
    }
}

// Report on timings
void report_timings(const double time, Options &opt, const int sSize, const double griddings)
{
    double spectralsamplesize = opt.nSamples*opt.nChan;
    cout << "    Number of iterations "<<opt.nIterations<<endl;
    cout << "    Time " << time << " (s) " << endl;
    cout << "    Time per visibility spectral sample " << 1e6*time / spectralsamplesize << " (us) " << endl;
    cout << "    Time per gridding   " << 1e9*time / (spectralsamplesize * static_cast<double>((sSize)*(sSize))) << " (ns) " << endl;
    cout << "    Gridding rate   " << (griddings / 1000000) / time << " (million grid points per second)" << endl;
}

int verify_result(std::string compname, 
    std::vector<Value> ref, std::vector<Value> comp, 
    double abserr, double relerr)
{
    cout << "Verifying "<<compname<<" results ..."<<endl;
    if (ref.size() != comp.size()) {
        cout << "Failed! (Grid sizes differ)" << std::endl;
        return 1;
    }

    for (unsigned long long i = 0; i < ref.size(); ++i) {
        if (fabs(ref[i].real() - comp[i].real()) > abserr) 
        {
            cout << "Failed! Expected " << comp[i].real() << ", got "
                     << comp[i].real() << " at index " << i <<endl;
            return 1;
        }
    }
    cout << "Passed" << endl;
    return 0;
}


/////////////////////////////////////////////////////////////////////////////////
// Initialize W project convolution function
// - This is application specific and should not need any changes.
//
// freq - temporal frequency (inverse wavelengths)
// cellSize - size of one grid cell in wavelengths
// support - Total width of convolution function=2*support+1
// wCellSize - size of one w grid cell in wavelengths
// wSize - Size of lookup table in w
void initC(const std::vector<Coord>& freq, const Coord cellSize,
           const Coord baseline,
           const int wSize, int& support, int& overSample,
           Coord& wCellSize, std::vector<Value>& C)
{
    cout << "Initializing W projection convolution function" << endl;
    // DAM -- I don't really understand the following equation. baseline*freq is the array size in wavelengths,
    // but I don't know why the sqrt is used and why there is a multiplication with cellSize rather than a division.
    // In the paper referred to in ../README.md they suggest using rms(w)*FoV for the width (in wavelengths), which
    // would lead to something more like:
    // support = max( 3, ceil( 0.5 * scale*baseline*freq[0] / (cellSize*cellSize) ) )
    // where "scale" reduces the maximum baseline length to the RMS (1/sqrt(3) for uniformaly distributed
    // visibilities, 1/(2+log10(n)/2) or so for n baselines with a Gaussian radial profile).
    support = static_cast<int>(1.5 * sqrt(std::abs(baseline) * static_cast<Coord>(cellSize)
                                          * freq[0]) / cellSize);

    cout << "FoV = " << 180./3.14159265/cellSize << " deg" << endl;

    overSample = 8;
    cout << "Support = " << support << " pixels" << endl;
    wCellSize = 2 * baseline * freq[0] / wSize;
    cout << "W cellsize = " << wCellSize << " wavelengths" << endl;

    // Convolution function. This should be the convolution of the
    // w projection kernel (the Fresnel term) with the convolution
    // function used in the standard case. The latter is needed to
    // suppress aliasing. In practice, we calculate entire function
    // by Fourier transformation. Here we take an approximation that
    // is good enough.
    const int sSize = 2 * support + 1;

    const int cCenter = (sSize - 1) / 2;

    C.resize(sSize*sSize*overSample*overSample*wSize);
    cout << "Size of convolution function = " << sSize*sSize*overSample
         *overSample*wSize*sizeof(Value) / (1024*1024) << " MB" << std::endl;
    cout << "Shape of convolution function = [" << sSize << ", " << sSize << ", "
             << overSample << ", " << overSample << ", " << wSize << "]" << std::endl;

    for (int k = 0; k < wSize; k++) {
        double w = double(k - wSize / 2);
        double fScale = sqrt(abs(w) * wCellSize * freq[0]) / cellSize;

        for (int osj = 0; osj < overSample; osj++) {
            for (int osi = 0; osi < overSample; osi++) {
                for (int j = 0; j < sSize; j++) {
                    const double j2 = std::pow((double(j - cCenter) + double(osj) / double(overSample)), 2);

                    for (int i = 0; i < sSize; i++) {
                        const double r2 = j2 + std::pow((double(i - cCenter) + double(osi) / double(overSample)), 2);
                        const int cind = i + sSize * (j + sSize * (osi + overSample * (osj + overSample * k)));

                        if (w != 0.0) {
                            C[cind] = static_cast<Value>(std::cos(r2 / (w * fScale)));
                        } else {
                            C[cind] = static_cast<Value>(std::exp(-r2));
                        }
                    }
                }
            }
        }
    }

    // Now normalise the convolution function
    Real sumC = 0.0;

    for (int i = 0; i < sSize*sSize*overSample*overSample*wSize; i++) {
        sumC += abs(C[i]);
    }

    for (int i = 0; i < sSize*sSize*overSample*overSample*wSize; i++) {
        C[i] *= Value(wSize * overSample * overSample / sumC);
    }
}

// Initialize Lookup function
// - This is application specific and should not need any changes.
//
// freq - temporal frequency (inverse wavelengths)
// cellSize - size of one grid cell in wavelengths
// gSize - size of grid in pixels (per axis)
// support - Total width of convolution function=2*support+1
// wCellSize - size of one w grid cell in wavelengths
// wSize - Size of lookup table in w
void initCOffset(const std::vector<Coord>& u, const std::vector<Coord>& v,
                 const std::vector<Coord>& w, const std::vector<Coord>& freq,
                 const Coord cellSize, const Coord wCellSize,
                 const int wSize, const int gSize, const int support, const int overSample,
                 std::vector<int>& cOffset, std::vector<int>& iu,
                 std::vector<int>& iv)
{
    const int nSamples = u.size();
    const int nChan = freq.size();

    const int sSize = 2 * support + 1;

    // Now calculate the offset for each visibility point
    cOffset.resize(nSamples*nChan);
    iu.resize(nSamples*nChan);
    iv.resize(nSamples*nChan);

    for (int i = 0; i < nSamples; i++) {
        for (int chan = 0; chan < nChan; chan++) {

            const int dind = i * nChan + chan;

            const Coord uScaled = freq[chan] * u[i] / cellSize;
            iu[dind] = int(uScaled);

            if (uScaled < Coord(iu[dind])) {
                iu[dind] -= 1;
            }

            const int fracu = int(overSample * (uScaled - Coord(iu[dind])));
            iu[dind] += gSize / 2;

            const Coord vScaled = freq[chan] * v[i] / cellSize;
            iv[dind] = int(vScaled);

            if (vScaled < Coord(iv[dind])) {
                iv[dind] -= 1;
            }

            const int fracv = int(overSample * (vScaled - Coord(iv[dind])));
            iv[dind] += gSize / 2;

            // The beginning of the convolution function for this point
            Coord wScaled = freq[chan] * w[i] / wCellSize;
            int woff = wSize / 2 + int(wScaled);
            cOffset[dind] = sSize * sSize * (fracu + overSample * (fracv + overSample * woff));
        }
    }

}

// Return a pseudo-random integer in the range 0..2147483647
// Based on an algorithm in Kernighan & Ritchie, "The C Programming Language"
static unsigned long next = 1;
int randomInt()
{
    const unsigned int maxint = std::numeric_limits<int>::max();
    next = next * 1103515245 + 12345;
    return ((unsigned int)(next / 65536) % maxint);
}
