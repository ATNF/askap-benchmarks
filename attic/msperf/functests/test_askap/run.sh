#!/bin/bash

source ../../init_package_env.sh

rm -rf *.ms

mpirun -np 16 msperf.sh -c config.in
