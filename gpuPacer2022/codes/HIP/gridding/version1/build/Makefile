CXX = hipcc
CXXFLAGS = -std=c++17 -fopenmp -O2 #-Xcompiler

# Target executable name:
EXE = bin/askapGrid.exe


#hipcc ../main.cpp ../src/GridderCPU.cpp ../src/GridderGPU.hip ../src/gridKernelGPU.hip ../src/Setup.cpp ../utilities/MaxError.cpp ../utilities/PrintVector.cpp ../utilities/RandomVectorGenerator.cpp -o askapGrid -std=c++14 -Xcompiler -fopenmp

OBJS = obj/main.o \
obj/GridderCPU.o \
obj/GridderGPU.o \
obj/gridKernelGPU.o \
obj/Setup.o \
obj/MaxError.o \
obj/PrintVector.o \
obj/RandomVectorGenerator.o \
obj/RunInfo.o

all : $(EXE)

$(EXE) : $(OBJS)
	$(CXX) $(CXXFLAGS) -v $(OBJS) -o $@ -I../src/ -I../utilities 2> make.$(shell date -Iseconds).log 

obj/%.o: ../src/%.cpp ../src/*.h ../src/*.cuh ../utilities/*.h
	$(CXX) $(CXXFLAGS) -I../src/ -I../utilities -c $< -o $@

obj/%.o: ../utilities/%.cpp ../src/*.h ../src/*.cuh ../utilities/*.h
	$(CXX) $(CXXFLAGS) -I../src/ -I../utilities -c $< -o $@

obj/%.o: ../src/%.hip ../src/*.h ../src/*.cuh ../utilities/*.h
	$(CXX) $(CXXFLAGS) -I../src/ -I../utilities -c $< -o $@

obj/%.o: ../%.cpp ../src/*.h ../src/*.cuh ../utilities/*.h
	$(CXX) $(CXXFLAGS) -I../src/ -I../utilities -c $< -o $@

clean:
	rm $(OBJS) $(EXE)