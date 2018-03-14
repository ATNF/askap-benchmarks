#!/bin/sh
numactl --membind 0 --cpunodebind 0 ./tConvolveMT 4 > instance1.out &
numactl --membind 1 --cpunodebind 1 ./tConvolveMT 4 > instance2.out
