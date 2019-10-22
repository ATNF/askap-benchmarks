#!/bin/bash

source ../../init_package_env.sh

mpiexec --hostfile ingest_ip -np 6 /astro/askap/sord/askapingest/Code/Components/CP/benchmarks/current/mpiperf/apps/mpithread.sh -c early_science.in > mpiperf.log
