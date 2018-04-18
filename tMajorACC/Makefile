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
    CFLAGS_EXTRA=-acc -ta=multicore
else ifeq ($(USE),gpu)
    # GPU version
    #CFLAGS_EXTRA=-acc -ta=tesla,cc60,managed -DGPU
    #CFLAGS_EXTRA=-acc -ta=tesla,cc60 -DGPU
    CFLAGS_EXTRA=-acc -ta=tesla,cc60 -DGPU -DVERIFY
    LDFLAGS_EXTRA=-Mcudalib=cufft
else
    # Multicore CPU version
endif

CFLAGS=-fast -O3 $(CFLAGS_EXTRA)
LDFLAGS=-L$(FFTW_LIB) -lfftw3f $(LDFLAGS_EXTRA)

all:		tMajorACC

Stopwatch.o:	Stopwatch.cc Stopwatch.h
		$(CXX) $(CFLAGS) -c Stopwatch.cc

tMajorACC:	tMajorACC.cc Stopwatch.o
		$(CXX) $(CFLAGS) -c tMajorACC.cc
		$(CXX) $(CFLAGS) $(LDFLAGS) -o tMajorACC tMajorACC.o Stopwatch.o

clean:
		rm -f *.o tMajorACC

