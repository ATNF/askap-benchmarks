# Copyright (c) 2017 CSIRO
# Australia Telescope National Facility (ATNF)
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# PO Box 76, Epping NSW 1710, Australia
# atnf-enquiries@csiro.au
#
# This file is part of the ASKAP software distribution.
#
# The ASKAP software distribution is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
#

#CXX=g++
#CFLAGS=-O3 -fstrict-aliasing -fcx-limited-range -Wall -Wextra -fopenmp

CXX=pgc++

USE=gpu
ifeq ($(USE),cpu)
    # Multicore CPU version
    CFLAGS=-fast -O3 -acc -ta=multicore
    LDFLAGS=
else ifeq ($(USE),gpu)
    # GPU version
    ifneq ("$(wildcard /opt/pgi)","")
        CFLAGS=-fast -O3 -acc -ta=nvidia,managed -DGPU
        LDFLAGS=-Mcudalib=cufft
    else
        CFLAGS=-fast -O3 -acc -ta=tesla,cc60,managed -DGPU
        LDFLAGS=-Mcudalib=cufft
    endif
else
    # Multicore CPU version
    CFLAGS=-fast -O3
    LDFLAGS=
endif

all:		tConvolveACC tConvolveDegrid tConvolveCmplxMult

Stopwatch.o:	Stopwatch.cc Stopwatch.h
		$(CXX) $(CFLAGS) -c Stopwatch.cc

tConvolveACC:	tConvolveACC.cc Stopwatch.o
		$(CXX) $(CFLAGS) -c tConvolveACC.cc
		$(CXX) $(CFLAGS) $(LDFLAGS) -o tConvolveACC tConvolveACC.o Stopwatch.o

tConvolveCmplxMult:	tConvolveCmplxMult.cc Stopwatch.o
		$(CXX) $(CFLAGS) -c tConvolveCmplxMult.cc
		$(CXX) $(CFLAGS) $(LDFLAGS) -o tConvolveCmplxMult tConvolveCmplxMult.o Stopwatch.o

tConvolveDegrid:	tConvolveDegrid.cc Stopwatch.o
		$(CXX) $(CFLAGS) -c tConvolveDegrid.cc
		$(CXX) $(CFLAGS) $(LDFLAGS) -o tConvolveDegrid tConvolveDegrid.o Stopwatch.o

clean:
		rm -f *.o tConvolveACC tConvolveDegrid

