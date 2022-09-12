#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --exclusive
#SBATCH -p askaprt

nmpi=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
basedir=/scratch/pawsey0001/pelahi/mpitests/
logdir=${basedir}/nodes-${SLURM_NNODES}.nmpi-${nmpi}
mkdir -p ${logdir}
fname=${logdir}/mpi.${SLURM_JOB_ID}

#have rank 0 display all mpich info 
export MPICH_ENV_DISPLAY=1
#have mpi report info 
export MPICH_MEMORY_REPORT=1
# file to which all mpich report info is written 
export MPICH_MEMORY_REPORT_FILE=${fname}.mpi_report

# test resetting buffer sizes 
# defaults are 
#export MPICH_GATHERV_MAX_TMP_SIZE=536870912
#export MPICH_SCATTERV_MAX_TMP_SIZE=536870912
#export MPICH_ALLREDUCE_BLK_SIZE=716800

# here can override if desired
#export MPICH_ALLREDUCE_BLK_SIZE=716800

srun ./bin/mpicomm_cpp 1> ${fname}.log 2> ${fname}.err
