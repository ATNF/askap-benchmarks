#!/bin/sh
#
# The "-np 32" is for running on giant which has 32 cores.
#
mpirun -np 32 ./tConvolveMPI
