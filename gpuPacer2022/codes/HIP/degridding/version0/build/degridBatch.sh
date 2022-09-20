#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq
#SBATCH --time=00:10:00
#SBATCH --account=director2196

module load cuda/11.4.2 gcc/11.1.0
hipcc ../main.cpp ../src/DegridderCPU.cpp ../src/DegridderGPU.cu ../src/degridKernelGPU.cu ../src/Setup.cpp ../utilities/MaxError.cpp ../utilities/PrintVector.cpp ../utilities/RandomVectorGenerator.cpp -o degridder -std=c++17 -Xcompiler -fopenmp -O2
srun ./degridder
