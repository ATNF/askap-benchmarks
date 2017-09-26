#!/bin/bash

source ../../init_package_env.sh

mpirun -np 8 mpithread -c config.in 1> test.log
