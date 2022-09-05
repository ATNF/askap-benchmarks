.PHONY: all clean

F90=gfortran 
OMPFLAGS=-fopenmp -DUSEOPENMP
FOMPFLAGS=-fopenmp 
LIBDIR=$(shell pwd)/../../lib/
LIB=-lprofile_util_omp
INCDIR=$(shell pwd)/../../include/
FLAGS=-L$(LIBDIR) -I$(INCDIR) -Wl,-rpath=$(LIBDIR)

all: bin/openmpvec_cpp bin/openmpvec_c bin/openmpvec_fort

clean:
	rm -f bin/openmpvec_cpp bin/openmpvec_c bin/openmpvec_fort

bin/openmpvec_cpp: openmpvec.cpp
	$(CXX) $(OMPFLAGS) $(FLAGS) openmpvec.cpp -o bin/openmpvec_cpp $(LIB) 

bin/openmpvec_c: openmpvec.c
	$(CXX) $(OMPFLAGS) $(FLAGS) openmpvec.c -o bin/openmpvec_c $(LIB)

bin/openmpvec_fort: openmpvec.f90
	$(F90) $(FOMPFLAGS) $(FLAGS) openmpvec.f90 -o bin/openmpvec_fort $(LIB)

	
