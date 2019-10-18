#!/bin/bash -l
#SBATCH --job-name=mpi-test
#SBATCH --time=1:00:00
#SBATCH --nodes=8
#SBATCH --tasks=8

source  ../../init_package_env.sh

srun -N 8 -n 8 mpiperf -c config.in > mpiperf.log


