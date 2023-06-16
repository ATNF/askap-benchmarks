# Add source to this project's executable.
add_library (SourceCodeCMake 
    Golden.cpp 
    GPUOlder.hip 
    GPUPS.hip 
    GPUPSFullUnroll.hip 
    GPUPSLastWUnrolled.hip
    GPUOlder.cu 
    GPUPS.cu 
    GPUPSFullUnroll.cu 
    GPUPSLastWUnrolled.cu
    )