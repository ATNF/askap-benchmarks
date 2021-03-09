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
