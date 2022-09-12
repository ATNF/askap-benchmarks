# Simple Makefile

OUTPUTFILEBASE=libprofile_util
CXXFLAGS = -fPIC -std=c++17 -O2
OMPFLAGS ?= -fopenmp
EXTRAFLAGS ?= 
COMPILER ?=$(CXX)
COMPILERFLAGS = $(CXXFLAGS) $(EXTRAFLAGS)

BUILDTYPE ?= serial
DEVICETYPE= cpu  
BUILDNAME ?=

OBJS = obj/mem_util.o obj/time_util.o obj/thread_affinity_util.o 
LIB = lib/$(OUTPUTFILEBASE)$(BUILDNAME)

$(LIB).so: $(OBJS)
	@echo "Making $(BUILDTYPE) for $(DEVICETYPE) library"
	$(CXX) -shared $(OBJS) -o $(LIB).so
	rm $(OBJS)

$(OBJS): obj/%.o : src/%.cpp include/profile_util.h

obj/%.o: src/%.cpp include/profile_util.h
	$(COMPILER) $(COMPILERFLAGS) -Iinclude/ -c $< -o $@

clean:
	rm -f $(LIB).so
