Running the tConvolveMPI benchmark
==================================

The tConvolveMPI benchmark program measures the performance of a convolutional resampling
algorithm as used in radio astronomy data processing. The benchmark is configured to reflect
the computing needs of the Australian Square Kilometer Array Pathfinder (ASKAP) central
processor.

The benchmark distributes work to multiple cores or multiple nodes via Message Passing
Interface (MPI) much like the ASKAP software, and while it is possible to benchmark an
entire cluster the aim of the benchmark is primarily to benchmark a single compute node.

Platform Requirements
---------------------
Building and execution of the benchmark requires:

* A host system with 512MB of RAM per CPU core
* A C++ compiler (e.g. GCC)
* Make
* MPI (e.g. OpenMPI, MPICH)

By default, the Makefile uses GNU C++ compiler and flags.

Building
--------
Given the file tConvolveMPI.tgz, the benchmark can be extracted and compiled as follows:

    $ tar zxvf tConvolveMPI.tgz 
    tConvolveMPI/
    tConvolveMPI/tConvolveMPI.cc
    tConvolveMPI/Benchmark.h
    tConvolveMPI/Benchmark.cc
    tConvolveMPI/Stopwatch.h
    tConvolveMPI/tConvolveMPI.h
    tConvolveMPI/Makefile
    tConvolveMPI/Makefile.intel
    tConvolveMPI/Stopwatch.cc

    $ cd tConvolveMPI

    $ make
    mpicxx -O3 -fstrict-aliasing -Wall -c tConvolveMPI.cc
    mpicxx -O3 -fstrict-aliasing -Wall -c Stopwatch.cc
    mpicxx -O3 -fstrict-aliasing -Wall -c Benchmark.cc
    mpicxx -o tConvolveMPI tConvolveMPI.o Stopwatch.o Benchmark.o

Execution
---------
Once built, the benchmark can be executed on a single node using the command below.
The number of processes (argument to –np) should generally be set to the number of
cores available on the node.

For example, a dual-socket quad-core system would use a value of 8. For a chip multithreading
(CMT) or hyper-threading enabled system optimum performance may be obtained by specifying
one process for each virtual-core. For example, a dual-socket quad-core hyper-threading
enabled system would use a value of 16.

    $ mpirun -np 4 tConvolveMPI
    Initializing W projection convolution function
    Support = 64 pixels
    W cellsize = 566.034 wavelengths
    Initializing W projection convolution function
    Support = 64 pixels
    W cellsize = 566.034 wavelengths
    Initializing W projection convolution function
    Support = 64 pixels
    W cellsize = 566.034 wavelengths
    Initializing W projection convolution function
    Support = 64 pixels
    W cellsize = 566.034 wavelengths
    Size of convolution function = 268 MB
    Shape of convolution function = [129, 129, 8, 8, 33]
    Size of convolution function = 268 MB
    Shape of convolution function = [129, 129, 8, 8, 33]
    Size of convolution function = 268 MB
    Shape of convolution function = [129, 129, 8, 8, 33]
    Size of convolution function = 268 MB
    Shape of convolution function = [129, 129, 8, 8, 33]
    +++++ Forward processing (MPI) +++++
        Number of processes: 4
        Time 17.66 (s) 
        Time per visibility spectral sample 110.375 (us) 
        Time per gridding   6.63271 (ns) 
        Gridding rate   603.071 (million grid points per second)
    +++++ Reverse processing (MPI) +++++
        Number of processes: 4
        Time 18.91 (s) 
        Time per visibility spectral sample 118.188 (us) 
        Time per degridding 7.10219 (ns) 
        Degridding rate 563.207 (million grid points per second)
    Done

NUMA Awareness
--------------
For systems of non-uniform memory architecture (NUMA) such as multi-socket AMD Opteron
or Intel Nehalem based platforms, memory locality awareness may be required to obtain
optimum performance. Such support may be provided by the MPI implementation in
cooperation with the operating system. For example to enable this support with OpenMPI
the `-mca mpi_paffinity_alone 1` argument can be added to the mpirun command. For example:

    $ mpirun -mca mpi_paffinity_alone 1 -np 4 tConvolveMPI

Use of optimized BLAS library
-----------------------------
The tConvolveMPI benchmark can optionally make use of an optimized BLAS library and can
be configured to use the `cblas_caxpy()` function for gridding and `cblas_cdotu_sub()`
function for degridding.

Included is an example makefile (Makefile.intel) which will use the Intel Compiler and
Intel Math Kernel library (MKL). This example can be easily adapted for other compilers
and/or BLAS library implementations.

Submitting Results
------------------
The full output from the execution of the benchmark should be submitted. Where an
optimized BLAS library is used both the results of the non-BLAS and BLAS enabled runs
should be submitted.

Where modification to either code or makefile is required, these modifications should
be submitted along with the results.

Interpretation of Results
-------------------------
The numbers of interest are the "gridding rate" and "degridding rate", where in both
cases a higher number is better. These numbers are a measure of throughput and can be
used to determine a price/performance or price/power ratio for the system under test.

The algorithm has a low flop/byte ratio and its performance is typically limited by both
memory latency and memory bandwidth. It is hoped this benchmark can be used to arrive at
an optimum node configuration with respect to selection of CPU and memory configuration.
