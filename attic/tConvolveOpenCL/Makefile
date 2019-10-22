CXX = g++
CFLAGS = -g -Wall -O0 -fstrict-aliasing -fPIC
INCLUDE=-I/home/hum092/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc
LIBPATH=-L/home/hum092/NVIDIA_GPU_Computing_SDK/OpenCL/common/lib/Linux64

EXENAME = tConvolveOpenCL
EXEOBJS = $(EXENAME).o Stopwatch.o OpenCLGridder.o
EXELIBS = -lOpenCL

all:		$(EXENAME)

%.o:		%.cc %.h
			$(CXX) $(CFLAGS) $(INCLUDE) -c $<

$(EXENAME):	$(EXEOBJS)
			$(CXX) $(CFLAGS) $(LIBPATH) -o $(EXENAME) $(EXEOBJS) $(EXELIBS)

clean: 
		rm -f $(EXENAME) *.o

