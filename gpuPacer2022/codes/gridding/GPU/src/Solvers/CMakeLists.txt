# Add source to this project's executable.
add_library (SourceCodeCMake 
    GridderCPU.cpp 
    GridderGPUOlder.hip 
    GridderGPUAtomicTiled.hip 
    GridderGPUAtomic.hip
    GridderGPUOlder.cu 
    GridderGPUAtomicTiled.cu 
    GridderGPUAtomic.cu
    )