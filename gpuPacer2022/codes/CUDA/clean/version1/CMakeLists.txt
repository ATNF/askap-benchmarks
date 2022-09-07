cmake_minimum_required(VERSION 3.8)

# set the project name
project(YS_Clean
		VERSION 1.0
        LANGUAGES CXX CUDA)

# specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

configure_file(YS_CleanConfig.h.in YS_CleanConfig.h)

# add the include library
add_subdirectory("utilities")
add_subdirectory("src")
# add_subdirectory("data")

# add the executable
add_executable(YS_Clean main.cpp)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
message(STATUS "OpenMP found")
target_link_libraries(YS_Clean PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(YS_Clean PUBLIC UtilitiesCMake)
target_link_libraries(YS_Clean PUBLIC SourceCodeCMake)

target_include_directories(YS_Clean PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           "${PROJECT_BINARY_DIR}/utilities"
                           "${PROJECT_BINARY_DIR}/src"
                           "${PROJECT_BINARY_DIR}/data"
                           )