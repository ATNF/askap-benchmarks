# Add source to this project's executable.
add_library (SourceCodeCMake 
    DegridderCPU.cpp 
    DegridderGPUTiled.hip
    DegridderGPULessIdle.hip 
    DegridderGPUSequential.hip 
    DegridderGPUInterleaved.hip 
    DegridderGPUWarpShuffle.hip
    DegridderGPUTiled.cu
    DegridderGPULessIdle.cu 
    DegridderGPUSequential.cu 
    DegridderGPUInterleaved.cu 
    DegridderGPUWarpShuffle.cu 
    DegridderGPUTiled.cu
    )