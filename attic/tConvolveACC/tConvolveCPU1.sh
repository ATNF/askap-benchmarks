#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --account=pawsey0233
#SBATCH --job-name=ConvolveOpenACC
#SBATCH --export=NONE

#export ACC_NUM_CORES=28
export ACC_NUM_CORES=56
echo 
echo ACC_NUM_CORES = $ACC_NUM_CORES

echo 
echo "--------------------------------------------------------"
echo without srun
./tConvolveCmplxMultCPU

echo 
echo "--------------------------------------------------------"
echo with srun
srun ./tConvolveCmplxMultCPU

