#!/bin/sh
#
# ASKAP auto-generated file
#

ASKAP_ROOT=/Users/ord006/Soft/askapsdp
export ASKAP_ROOT

PATH=${ASKAP_ROOT}/3rdParty/LOFAR/Common/Common-3.3/install/bin:${ASKAP_ROOT}/3rdParty/apr-util/apr-util-1.3.9/install/bin:${ASKAP_ROOT}/3rdParty/expat/expat-2.0.1/install/bin:${ASKAP_ROOT}/3rdParty/apr/apr-1.3.9/install/bin:${ASKAP_ROOT}/3rdParty/casacore/casacore-2.0.3/install/bin:${ASKAP_ROOT}/3rdParty/wcslib/wcslib-4.18/install/bin:${ASKAP_ROOT}/3rdParty/fftw/fftw-3.3.3/install/bin:${PATH}
export PATH

if [ "${DYLD_LIBRARY_PATH}" !=  "" ]
then
    DYLD_LIBRARY_PATH=${ASKAP_ROOT}/Code/Components/CP/benchmarks/current/mpiperf:${ASKAP_ROOT}/3rdParty/LOFAR/Common/Common-3.3/install/lib:${ASKAP_ROOT}/3rdParty/boost/boost-1.56.0/install/lib:${ASKAP_ROOT}/3rdParty/log4cxx/log4cxx-0.10.0/install/lib:${ASKAP_ROOT}/3rdParty/apr-util/apr-util-1.3.9/install/lib:${ASKAP_ROOT}/3rdParty/expat/expat-2.0.1/install/lib:${ASKAP_ROOT}/3rdParty/apr/apr-1.3.9/install/lib:${ASKAP_ROOT}/3rdParty/cmdlineparser/cmdlineparser-0.1.1/install/lib:${ASKAP_ROOT}/3rdParty/casa-components/casa-components-1.6.0/install/lib:${ASKAP_ROOT}/3rdParty/casacore/casacore-2.0.3/install/lib:${ASKAP_ROOT}/3rdParty/wcslib/wcslib-4.18/install/lib:${ASKAP_ROOT}/3rdParty/cfitsio/cfitsio-3.35/install/lib:${ASKAP_ROOT}/3rdParty/fftw/fftw-3.3.3/install/lib:${DYLD_LIBRARY_PATH}
else
    DYLD_LIBRARY_PATH=${ASKAP_ROOT}/Code/Components/CP/benchmarks/current/mpiperf:${ASKAP_ROOT}/3rdParty/LOFAR/Common/Common-3.3/install/lib:${ASKAP_ROOT}/3rdParty/boost/boost-1.56.0/install/lib:${ASKAP_ROOT}/3rdParty/log4cxx/log4cxx-0.10.0/install/lib:${ASKAP_ROOT}/3rdParty/apr-util/apr-util-1.3.9/install/lib:${ASKAP_ROOT}/3rdParty/expat/expat-2.0.1/install/lib:${ASKAP_ROOT}/3rdParty/apr/apr-1.3.9/install/lib:${ASKAP_ROOT}/3rdParty/cmdlineparser/cmdlineparser-0.1.1/install/lib:${ASKAP_ROOT}/3rdParty/casa-components/casa-components-1.6.0/install/lib:${ASKAP_ROOT}/3rdParty/casacore/casacore-2.0.3/install/lib:${ASKAP_ROOT}/3rdParty/wcslib/wcslib-4.18/install/lib:${ASKAP_ROOT}/3rdParty/cfitsio/cfitsio-3.35/install/lib:${ASKAP_ROOT}/3rdParty/fftw/fftw-3.3.3/install/lib
fi
export DYLD_LIBRARY_PATH

exec ${ASKAP_ROOT}/Code/Components/CP/benchmarks/current/mpiperf/apps/mpiperf "$@"
