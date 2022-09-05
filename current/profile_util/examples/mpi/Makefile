.PHONY: all clean

LIBDIR=$(shell pwd)/../../lib/
LIB=-lprofile_util_mpi 
INCDIR=$(shell pwd)/../../include/
FLAGS=-L$(LIBDIR) -I$(INCDIR) -Wl,-rpath=$(LIBDIR) -D_MPI -std=c++17
CXXFLAGS?=

all: bin/mpivec_cpp bin/mpicomm_cpp bin/mpi-compute

clean:
	rm -f bin/mpivec_cpp
	rm -f bin/mpicomm_cpp

bin/mpivec_cpp: mpivec.cpp
	$(MPICXX) $(FLAGS) mpivec.cpp -o bin/mpivec_cpp $(LIB)

bin/mpicomm_cpp: mpicomm.cpp
	$(MPICXX) $(FLAGS) $(CXXFLAGS) mpicomm.cpp -o bin/mpicomm_cpp $(LIB)

bin/mpi-compute: mpi-compute.cpp
	$(MPICXX) $(FLAGS) $(CXXFLAGS) mpi-compute.cpp -o bin/mpi-compute $(LIB)

