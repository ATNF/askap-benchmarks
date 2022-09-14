#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --sockets-per-node=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=workq
#SBATCH --time=00:10:00
###SBATCH --account=director2196
#SBATCH --account=pawsey0007

module load rocm/4.5.0 gcc/11.2.0

make clean
CXXFLAGS="-v -Xcompiler -fopenmp -O2" make all

srun rocprof --stats --sys-trace ./bin/askapDegrid.exe
