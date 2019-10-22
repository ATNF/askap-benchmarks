#!/bin/bash

source ../../init_package_env.sh

mpirun -np 6 mpiperf -c config.in
