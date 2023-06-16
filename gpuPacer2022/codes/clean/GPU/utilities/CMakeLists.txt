# Add source to this project's executable.
add_library (UtilitiesCMake 
    ImageProcess.cpp 
    MaxError.cpp 
    MemUtil.cpp 
    ThreadAffinityUtil.cpp 
    TimeUtil.cpp 
    WarmupGPU.hip
    WarmupGPU.cu
    )