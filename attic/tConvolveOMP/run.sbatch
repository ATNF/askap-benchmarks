#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuq-dev
#SBATCH --job-name=tConvolveOMP
#SBATCH --export=NONE

export OMP_NUM_THREADS=4
echo OMP_NUM_THREADS = $OMP_NUM_THREADS
srun ./tConvolveOMP

export OMP_NUM_THREADS=8
echo OMP_NUM_THREADS = $OMP_NUM_THREADS
srun ./tConvolveOMP

export OMP_NUM_THREADS=16
echo OMP_NUM_THREADS = $OMP_NUM_THREADS
srun ./tConvolveOMP

