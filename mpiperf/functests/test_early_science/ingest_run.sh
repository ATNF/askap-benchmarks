#!/bin/bash

source ../../init_package_env.sh

mpiexec --hostfile ingest_machines -np 8 /astro/askap/sord/askapingest/Code/Components/CP/benchmarks/current/mpiperf/apps/mpiperf.sh -c config.in > mpiperf.log
