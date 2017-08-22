#!/bin/bash

source ../../init_package_env.sh

mpiexec --hostfile ingest_ip -np 6 /astro/askap/sord/askapingest/Code/Components/CP/benchmarks/current/mpiperf/apps/mpiperf.sh -c config.in > mpiperf.log
