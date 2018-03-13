#!/bin/bash
#SBATCH --partition=askap
#SBATCH --nodelist=galaxy-ingest07,galaxy-ingest15
#SBATCH --export=NONE
#SBATCH --account=askaprt
#SBATCH --time=11:00:00

module load sandybridge gcc mvapich 

source ../../init_package_env.sh

# This will run on the one node
#srun --export=all -n 2 -N 1  /bin/hostname

# This will run on two nodes, i.e. one task on each node
srun --export=all -n 2 -N  2 mpithread -c early_science.in
