#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --account=pawsey0233
#SBATCH --job-name=ConvolveOpenACC
#SBATCH --export=NONE

export ACC_NUM_CORES=1
echo ACC_NUM_CORES = $ACC_NUM_CORES
./tConvolveCmplxMultCPU

export ACC_NUM_CORES=14
echo ACC_NUM_CORES = $ACC_NUM_CORES
./tConvolveCmplxMultCPU

export ACC_NUM_CORES=28
echo ACC_NUM_CORES = $ACC_NUM_CORES
./tConvolveCmplxMultCPU

export ACC_NUM_CORES=56
echo ACC_NUM_CORES = $ACC_NUM_CORES
./tConvolveCmplxMultCPU

