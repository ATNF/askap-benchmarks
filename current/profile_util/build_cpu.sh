#!/bin/bash 

CXX=CC
MPICXX=CC
if [ ! -z $1 ]; then
    CXX=$1
fi
if [ ! -z $2]; then
    MPICXX=$2
fi

devicetype=cpu
OMPFLAGS=-fopenmp
# first is serial
buildtypes=("Serial" "OpenMP" "MPI" "MPI+OpenMP")
buildnames=(" " "_omp" "_mpi" "_mpi_omp")
compilers=(${CXX} ${CXX} ${MPICXX} ${MPICXX})
extraflags=("" "${OMPFLAGS}" "-D_MPI" "-D_MPI ${OMPFLAGS}")

for ((i=0;i<4;i++)) 
do 
    echo "BUILDTYPE=${buildtypes[$i]} BUILDNAME=${buildnames[$i]} DEVICETYPE=${devicetype}"
    make BUILDTYPE=${buildtypes[$i]} BUILDNAME=${buildnames[$i]} DEVICETYPE=${devicetype} clean
    make BUILDTYPE=${buildtypes[$i]} BUILDNAME=${buildnames[$i]} DEVICETYPE=${devicetype} CXX=${compilers[$i]} COMPILER=${compilers[$i]}  EXTRAFLAGS="${extraflags[$i]}" 
done

