#!/bin/bash 

CXX=nvcc
MPICXX=nvcc
if [ ! -z $1 ]; then
    CXX=$1
fi
if [ ! -z $2]; then
    MPICXX=$2
fi

devicetype=cuda
# first is serial
buildtypes=("CUDA Serial" "CUDA OpenMP" "CUDA MPI" "CUDA MPI+OpenMP")
buildnames=("_cuda" "_cuda_omp" "_cuda_mpi" "_cuda_mpi_omp")
compilers=(${CXX} ${CXX} ${MPICXX} ${MPICXX})
extraflags=("" "${OMPFLAGS}" "-D_MPI" "-D_MPI ${OMPFLAGS}")


for ((i=0;i<4;i++)) 
do 
    echo "BUILDNAME=${buildnames[$i]} BUILDNAME=${buildnames[$i]} DEVICETYPE=${devicetype}"
    make BUILDNAME=${buildnames[$i]} BUILDNAME=${buildnames[$i]} DEVICETYPE=${devicetype} clean
    make BUILDNAME=${buildnames[$i]} BUILDNAME=${buildnames[$i]} DEVICETYPE=${devicetype} CXX=${compilers[$i]} COMPILER=${compilers[$i]} EXTRAFLAGS="${extraflags[$i]}" 
done

