#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --account=pawsey0233
#SBATCH --job-name=CleanOpenACC
#SBATCH --export=NONE

srun ./tHogbomCleanACC

