CXX = hipcc
CXXFLAGS = -std=c++17 -fopenmp -O2 #-Xcompiler

# Target executable name:
EXE = bin/askapClean.exe

OBJS = obj/main.o \
obj/HogbomCuda.o \
obj/HogbomGolden.o \
obj/MaxError.o \
obj/ImageProcess.o \
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