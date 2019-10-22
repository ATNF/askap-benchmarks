#!/bin/bash

source ../../init_package_env.sh

rm -rf *.ms

mpirun -np 2 msperf.sh -c config.in
