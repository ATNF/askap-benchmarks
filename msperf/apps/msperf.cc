/// @file msperf.cc
///
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
/// @author Ben Humphreys <ben.humphreys@csiro.au>

// Include package level header file
#include "askap_msperf.h"

// System includes
#include <iostream>
#include <string>
#include <sstream>
#include <mpi.h>

// ASKAPsoft includes
#include "CommandLineParser.h"
#include "Common/ParameterSet.h"
#include "casa/OS/Timer.h"

// Local includes
#include "writers/DataSet.h"

// Using
using LOFAR::ParameterSet;

static ParameterSet getParameterSet(int argc, char *argv[])
{
    cmdlineparser::Parser parser;

    // Command line parameter
    cmdlineparser::FlaggedParameter<std::string> inputsPar("-c", "config.in");

    // This parameter is optional, default will be returned if not present
    parser.add(inputsPar, cmdlineparser::Parser::return_default);

    parser.process(argc, const_cast<char**> (argv));

    const std::string parsetFile = inputsPar;

    LOFAR::ParameterSet parset(parsetFile);

    return parset;
}

static std::string itostr(const int i)
{
    std::stringstream ss;
    std::string str;
    ss << i;
    ss >> str;

    return str;
}

int main(int argc, char *argv[])
{
    // MPI init
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ParameterSet parset = getParameterSet(argc, argv);
    ParameterSet subset(parset.makeSubset("msperf."));

    // Replace in the filename the %w pattern with the rank number
    std::string filename = subset.getString("filename");
    const std::string pattern = "%w";
    filename.replace(filename.find(pattern),pattern.length(), itostr(rank));

    int intTime = subset.getInt32("integrationTime");
    int integrations = subset.getInt32("nIntegrations");

    DataSet data(filename, subset);

    casa::Timer timer;
    casa::Timer total;
    total.mark();
    for (int i = 0; i < integrations; ++i) {
        timer.mark();
        data.add();
        MPI_Barrier(MPI_COMM_WORLD);

        // Report progress
        if (rank == 0) {
            const float realtime = timer.real();
            const float perf = static_cast<float>(intTime) / realtime;
            std::cout << "Wrote integration " << i <<
            " in " << realtime << " seconds"
            << " (" << perf << "x requirement)" << std::endl;
        }
    }

    // Report totals
    if (rank == 0) {
        const float realtime = total.real();
        const float perf = static_cast<float>(intTime * integrations) / realtime;
        std::cout << "Wrote " << integrations << " integrations "
            " in " << realtime << " seconds"
            << " (" << perf << "x requirement)" << std::endl;
    }

    MPI_Finalize();

    return 0;
}
