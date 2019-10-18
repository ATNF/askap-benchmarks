# Copyright (c) 2019 CSIRO
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


CXX=pgc++

USE=gpu
ifeq ($(USE),cpu)
    # Multicore CPU version
    CFLAGS=-fast -O3 -acc -ta=multicore
    LDFLAGS=
else ifeq ($(USE),gpu)
    # GPU version
    ifneq ("$(wildcard /opt/pgi)","")
        #CFLAGS=-fast -O3 -acc -ta=nvidia,cc60,managed
        CFLAGS=-O1 -acc -ta=nvidia,cc60,managed
        #LDFLAGS=-Mcudalib=cufft
    else
        CFLAGS=-fast -O3 -acc -ta=tesla,cc60,managed
        #LDFLAGS=-Mcudalib=cufft
    endif
else
    # Serial CPU version
    CFLAGS=-fast -O3
    LDFLAGS=
endif

EXENAME = tConvolveACC
OBJS = tConvolveACC.o Stopwatch.o Benchmark.o

all:		$(EXENAME)

%.o:		%.cc %.h Benchmark.h
		$(CXX) $(CFLAGS) -c $<

$(EXENAME):	$(OBJS)
		$(CXX) -o $(EXENAME) $(OBJS) $(CFLAGS) $(LDFLAGS)

clean:
		rm -f $(EXENAME) *.o

