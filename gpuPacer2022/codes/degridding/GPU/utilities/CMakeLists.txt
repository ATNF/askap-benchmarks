# Add source to this project's executable.
add_library (UtilitiesCMake 
    PrintVector.cpp
    RandomVectorGenerator.cpp
    Setup.cpp 
    MaxError.cpp 
    MemUtil.cpp 
    ThreadAffinityUtil.cpp 
    TimeUtil.cpp 
    WarmupGPU.hip
    WarmupGPU.cu
    )