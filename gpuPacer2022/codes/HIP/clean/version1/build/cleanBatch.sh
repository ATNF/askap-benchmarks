#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq
#SBATCH --time=00:10:00
#SBATCH --account=director2196

module load hip/4.3.0
hipcc ../main.cpp ../src/HogbomCuda.hip ../src/HogbomGolden.cpp ../utilities/MaxError.cpp ../utilities/ImageProcess.cpp -o askapClean -std=c++14 -Xcompiler -fopenmp
srun ./askapClean
