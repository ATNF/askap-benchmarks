#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --account=pawsey0233
#SBATCH --job-name=CleanOpenACC
#SBATCH --export=NONE

export ACC_NUM_CORES=28
echo
echo ACC_NUM_CORES = $ACC_NUM_CORES

echo
echo "--------------------------------------------------------"
echo without srun
./tHogbomCleanCPU

echo
echo "--------------------------------------------------------"
echo with srun
srun ./tHogbomCleanCPU

