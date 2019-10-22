/// @copyright (c) 2011 CSIRO
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
/// @detail testing simple ACC multithreading. Use env variable ACC_NUM_CORES to
/// control the number of threads when compiling with -ta=multicore
///
/// @author Daniel Mitchell <daniel.mitchell@csiro.au>

// System includes
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstddef>
#include <cmath>
#include <sys/stat.h>

// Local includes
#include "tSimpleACC.h"

using namespace std;

int main(int /*argc*/, char** /* argv*/)
{
    cout << "Testing simple reduction" << endl;

    const float data[] = {0,0,1,0,0,0};
    const size_t size = sizeof(data) / sizeof(float);

    const int ftype = 4;
    float threadMaxVal=0.0;
    #pragma acc parallel loop reduction(max:threadMaxVal)
    for (size_t i = 0; i < size; ++i) {
        if (ftype==0) threadMaxVal = data[i]; // GPU: yes, CPU MC: no
        else if (ftype==1) threadMaxVal = max( threadMaxVal, data[i] ); // GPU: yes, CPU MC: no
        else if (ftype==2) threadMaxVal = fmax( threadMaxVal, data[i] ); // GPU: yes, CPU MC: yes
        else if (ftype==3) threadMaxVal = fmaxf( threadMaxVal, data[i] ); // GPU: yes, CPU MC: yes
        else if ((ftype==4) && (data[i]>threadMaxVal)) threadMaxVal = data[i]; // GPU: yes, CPU MC: yes
    }

    cout << "Max value = " << threadMaxVal << endl;

/*
    float f0=0.0, f1=1.0;
    double d0=0.0, d1=1.0;
    cout << "fmax(f0,f1) = " << fmax(f0,f1)  << endl;
    cout << "max(f0,f1)  = " << max(f0,f1)  << endl;
    cout << "max(d0,d1)  = " << max(d0,d1)  << endl;
*/

    return 0;
}
