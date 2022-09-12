# Simple Makefile

# for building with hip 
#hipcc ../main.cpp ../src/GridderCPU.cpp ../src/GridderGPU.hip ../src/gridKernelGPU.hip ../src/Setup.cpp ../utilities/MaxError.cpp ../utilities/PrintVector.cpp ../utilities/RandomVectorGenerator.cpp -o askapGrid -std=c++14 -Xcompiler -fopenmp

CXX ?= hipcc
CXXFLAGS = -fPIC -std=c++17 -O2
COMPILERFLAGS = $(CXXFLAGS) $(EXTRAFLAGS)
VERSION ?= 1

.PHONY: all clean gridding degridding hogbomclean

all: gridding degridding hogbomclean

gridding:
	@echo "Making gridding" ;
	$(MAKE) -C gridding/version$(VERSION)/build all
degridding:
	@echo "Making degridding" ;
	$(MAKE) -C degridding/version$(VERSION)/build all
hogbomclean:
	@echo "Making hogbom clean" ;
	$(MAKE) -C clean/version$(VERSION)/build all

clean:
	@echo "Clean builds" ;
	$(MAKE) -C gridding/version$(VERSION)/build clean ;
	$(MAKE) -C degridding/version$(VERSION)/build clean ;
	$(MAKE) -C clean/version$(VERSION)/build clean ;
