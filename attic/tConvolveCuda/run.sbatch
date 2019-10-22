#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --account=pawsey0233
#SBATCH --export=NONE
 
module load broadwell gcc/5.4.0 cuda mvapich
./tConvolveCuda 
