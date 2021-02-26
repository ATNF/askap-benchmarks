#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --account=pawsey0001
#SBATCH --export=NONE

# load modules 
module load gcc 
module load cuda

# compile and build
make clean
make

# run
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}
./tConvolveCuda 
