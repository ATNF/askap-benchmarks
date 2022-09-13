CXX = hipcc
CXXFLAGS ?=  -v -Xcompiler -fopenmp -O2

# Target executable name:
EXE = bin/askapDegrid.exe

OBJS = obj/main.o \
obj/DegridderCPU.o \
obj/DegridderGPU.o \
obj/degridKernelGPU.o \
obj/Setup.o \
obj/MaxError.o \
obj/PrintVector.o \
obj/RandomVectorGenerator.o \
obj/RunInfo.o

all : $(EXE)

$(EXE) : $(OBJS)
	$(CXX) $(CXXFLAGS) -v $(OBJS) -o $@ -I../src/ -I../utilities 2> make.$(shell date -Iseconds).log 

obj/%.o: ../src/%.cpp ../src/*.h ../utilities/*.h
	$(CXX) $(CXXFLAGS) -I../src/ -I../utilities -c $< -o $@

obj/%.o: ../utilities/%.cpp ../src/*.h ../utilities/*.h
	$(CXX) $(CXXFLAGS) -I../src/ -I../utilities -c $< -o $@

obj/%.o: ../src/%.cu ../src/*.h ../utilities/*.h
	$(CXX) $(CXXFLAGS) -I../src/ -I../utilities -c $< -o $@

obj/%.o: ../%.cpp ../src/*.h ../utilities/*.h
	$(CXX) $(CXXFLAGS) -I../src/ -I../utilities -c $< -o $@

clean:
	rm -f $(OBJS) $(EXE)
