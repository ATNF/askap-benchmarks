hipcc ../main.cpp ../src/Solvers/DegridderCPU.cpp ../src/Solvers/DegridderGPUInterleaved.cu ../src/Solvers/DegridderGPULessIdle.cu ../src/Solvers/DegridderGPUSequential.cu ../src/Solvers/DegridderGPUTiled.cu ../src/Solvers/DegridderGPUWarpShuffle.cu ../utilities/WarmupGPU.cu ../utilities/MaxError.cpp ../utilities/Setup.cpp ../utilities/RandomVectorGenerator.cpp ../utilities/PrintVector.cpp -o askapDegridder -std=c++17 -Xcompiler -fopenmp
