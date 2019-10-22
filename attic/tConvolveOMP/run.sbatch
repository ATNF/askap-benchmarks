#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=tConvolveOMP
#SBATCH --export=NONE

export OMP_NUM_THREADS=4
echo OMP_NUM_THREADS = $OMP_NUM_THREADS
srun ./tConvolveOMP

export OMP_NUM_THREADS=10
echo OMP_NUM_THREADS = $OMP_NUM_THREADS
srun ./tConvolveOMP

export OMP_NUM_THREADS=20
echo OMP_NUM_THREADS = $OMP_NUM_THREADS
srun ./tConvolveOMP

