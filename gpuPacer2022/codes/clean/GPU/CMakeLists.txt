cmake_minimum_required(VERSION 3.24)

# set the project name
project(askap
		VERSION 1.0
        LANGUAGES CXX)

# specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

askap_option(USE_HIP "Attempt to use HIP instead of CUDA" OFF)

# set the GPU langugae and standard
if (ASKAP_USE_HIP)
    enable_language(HIP)
    add_definitions(USEHIP)
    if(NOT DEFINED CMAKE_HIP_STANDARD)
        set(CMAKE_HIP_STANDARD 17)
        set(CMAKE_HIP_STANDARD_REQUIRED ON)
    endif()
else()
    enable_language(CUDA)
    add_definitions(USECUDA)
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