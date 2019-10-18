#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --account=pawsey0233
#SBATCH --job-name=ConvolveOpenACC
#SBATCH --export=NONE

srun ./tConvolveACC
srun ./tConvolveCmplxMult
srun ./tConvolveDegrid

