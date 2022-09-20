#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq
#SBATCH --time=00:10:00
###SBATCH --account=director2196
#SBATCH --account=pawsey0007

module load cuda/11.4.2 gcc/11.1.0

make clean
CXXFLAGS="-v -Xcompiler -fopenmp -O2" make all

srun nvprof ./bin/askapDegrid.exe
