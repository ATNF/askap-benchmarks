#!/bin/bash 

CXX=hipcc
MPICXX=hipcc
if [ ! -z $1 ]; then
    CXX=$1
fi
if [ ! -z $2]; then
    MPICXX=$2
fi

devicetype=hip
# first is serial
buildtypes=("HIP Serial" "HIP OpenMP" "HIP MPI" "HIP MPI+OpenMP")
buildnames=("_hip" "_hip_omp" "_hip_mpi" "_hip_mpi_omp")
compilers=(${CXX} ${CXX} ${MPICXX} ${MPICXX})
extraflags=("" "${OMPFLAGS}" "-D_MPI" "-D_MPI ${OMPFLAGS}")


for ((i=0;i<4;i++)) 
do 
    echo "BUILDNAME=${buildnames[$i]} BUILDNAME=${buildnames[$i]} DEVICETYPE=${devicetype}"
    make BUILDNAME=${buildnames[$i]} BUILDNAME=${buildnames[$i]} DEVICETYPE=${devicetype} clean
    make BUILDNAME=${buildnames[$i]} BUILDNAME=${buildnames[$i]} DEVICETYPE=${devicetype} CXX=${compilers[$i]} COMPILER=${compilers[$i]} EXTRAFLAGS="${extraflags[$i]}" 
done

