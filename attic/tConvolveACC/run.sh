#!/bin/bash

export OMP_NUM_THREADS=4
numactl --membind 0 --cpunodebind 0 ./tConvolveOMP
