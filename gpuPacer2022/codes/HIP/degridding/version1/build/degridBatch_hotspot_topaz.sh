#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq
#SBATCH --time=00:10:00
###SBATCH --account=director2196
#SBATCH --account=pawsey0007

module load hip/4.3.0

make clean
make all

srun nvprof ./bin/askapDegrid.exe
