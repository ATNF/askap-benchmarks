#!/bin/bash 

CXX=g++
MPICXX=mpic++
if [ ! -z $1 ]; then
    CXX=$1
fi
if [ ! -z $2]; then
    MPICXX=$2
fi

CUDACXX=nvcc
CUDAMPICXX=mpic++
if [ ! -z $3 ]; then
    CUDACXX=$3
fi
if [ ! -z $4]; then
    CUDAMPICXX=$4
fi

HIPCXX=hipcc
HIPMPICXX=hipcc
if [ ! -z $5 ]; then
    HIPCXX=$5
fi
if [ ! -z $6]; then
    HIPMPICXX=$6
fi

./build_cpu.sh ${CXX} ${MPICXX}
./build_cuda.sh ${CUDACXX} ${CUDAMPICXX}
./build_hip.sh ${HIPCXX} ${HIPMPICXX}