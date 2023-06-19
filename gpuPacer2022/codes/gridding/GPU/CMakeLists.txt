cmake_minimum_required(VERSION 3.18)

# set the project name
project(askap
		VERSION 1.0
        LANGUAGES CXX)

# specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

macro(askap_option optname optdesc status)
	option(ASKAP_${optname} "${optdesc}" "${status}")
endmacro()

askap_option(USE_HIP "Attempt to use HIP instead of CUDA" OFF)
askap_option(USE_AMD "Attempt to use HIP AMD" ON)

# set the GPU langugae and standard
if (ASKAP_USE_HIP)
    # Find hip
    find_package(HIP REQUIRED)
    # Link with HIP
    enable_language(HIP)
    if(NOT DEFINED CMAKE_HIP_STANDARD)
        set(CMAKE_HIP_STANDARD 17)
        set(CMAKE_HIP_STANDARD_REQUIRED ON)
    endif()
    add_definitions("-DUSEHIP")
    add_compile_options("-fPIE")
    target_link_libraries(hip::device)
    if (ASKAP_USE_AMD)
        add_definitions("-D__HIP_PLATFORM_AMD__")
        set(GPU_TARGETS "gfx90a" CACHE STRING "GPU targets to compile for")
    endif()
else()
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    add_definitions("-DUSECUDA")
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
endif()

configure_file(askapConfig.h.in askapConfig.h)

# add the include library
add_subdirectory("utilities")
add_subdirectory("src/Solvers")

# add the executable
add_executable(askap main.cpp)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
message(STATUS "OpenMP found")
target_link_libraries(askap PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(askap PUBLIC UtilitiesCMake)
target_link_libraries(askap PUBLIC SourceCodeCMake)

target_include_directories(askap PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           "${PROJECT_BINARY_DIR}/utilities"
                           "${PROJECT_BINARY_DIR}/src"
                           "${PROJECT_BINARY_DIR}/src/Solvers"
                           "${PROJECT_BINARY_DIR}/data"
                           )
