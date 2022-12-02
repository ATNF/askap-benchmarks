# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /pawsey/centos7.6/devel/gcc/4.8.5/cmake/3.18.0/bin/cmake

# The command to remove a file.
RM = /pawsey/centos7.6/devel/gcc/4.8.5/cmake/3.18.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /group/director2196/ocekmer/askap-benchmarks/gpuPacer2022/codes/gridding/CUDA

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /group/director2196/ocekmer/askap-benchmarks/gpuPacer2022/codes/gridding/CUDA/build

# Include any dependencies generated for this target.
include CMakeFiles/askap.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/askap.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/askap.dir/flags.make

CMakeFiles/askap.dir/main.cpp.o: CMakeFiles/askap.dir/flags.make
CMakeFiles/askap.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/group/director2196/ocekmer/askap-benchmarks/gpuPacer2022/codes/gridding/CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/askap.dir/main.cpp.o"
	/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/askap.dir/main.cpp.o -c /group/director2196/ocekmer/askap-benchmarks/gpuPacer2022/codes/gridding/CUDA/main.cpp

CMakeFiles/askap.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/askap.dir/main.cpp.i"
	/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /group/director2196/ocekmer/askap-benchmarks/gpuPacer2022/codes/gridding/CUDA/main.cpp > CMakeFiles/askap.dir/main.cpp.i

CMakeFiles/askap.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/askap.dir/main.cpp.s"
	/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /group/director2196/ocekmer/askap-benchmarks/gpuPacer2022/codes/gridding/CUDA/main.cpp -o CMakeFiles/askap.dir/main.cpp.s

# Object files for target askap
askap_OBJECTS = \
"CMakeFiles/askap.dir/main.cpp.o"

# External object files for target askap
askap_EXTERNAL_OBJECTS =

askap: CMakeFiles/askap.dir/main.cpp.o
askap: CMakeFiles/askap.dir/build.make
askap: utilities/libUtilitiesCMake.a
askap: src/Solvers/libSourceCodeCMake.a
askap: /pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/lib64/libgomp.so
askap: /lib64/libpthread.so
askap: CMakeFiles/askap.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/group/director2196/ocekmer/askap-benchmarks/gpuPacer2022/codes/gridding/CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable askap"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/askap.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/askap.dir/build: askap

.PHONY : CMakeFiles/askap.dir/build

CMakeFiles/askap.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/askap.dir/cmake_clean.cmake
.PHONY : CMakeFiles/askap.dir/clean

CMakeFiles/askap.dir/depend:
	cd /group/director2196/ocekmer/askap-benchmarks/gpuPacer2022/codes/gridding/CUDA/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /group/director2196/ocekmer/askap-benchmarks/gpuPacer2022/codes/gridding/CUDA /group/director2196/ocekmer/askap-benchmarks/gpuPacer2022/codes/gridding/CUDA /group/director2196/ocekmer/askap-benchmarks/gpuPacer2022/codes/gridding/CUDA/build /group/director2196/ocekmer/askap-benchmarks/gpuPacer2022/codes/gridding/CUDA/build /group/director2196/ocekmer/askap-benchmarks/gpuPacer2022/codes/gridding/CUDA/build/CMakeFiles/askap.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/askap.dir/depend

