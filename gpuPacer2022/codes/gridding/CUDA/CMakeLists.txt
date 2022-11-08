cmake_minimum_required(VERSION 3.8)

# set the project name
project(askap
		VERSION 1.0
        LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
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