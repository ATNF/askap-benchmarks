hipcc ../main.cpp ../src/Solvers/GridderCPU.cpp ../src/Solvers/GridderGPUAtomic.cu ../src/Solvers/GridderGPUAtomicTiled.cu ../src/Solvers/GridderGPUOlder.cu ../utilities/WarmupGPU.cu ../utilities/MaxError.cpp ../utilities/Setup.cpp ../utilities/RandomVectorGenerator.cpp ../utilities/PrintVector.cpp -o askapDegridder -std=c++17 -Xcompiler -fopenmp
