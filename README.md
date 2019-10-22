ASKAP Benchmark Packages
========================

These benchmark packages are available for use to benchmark key algorithms
required to image data from the Australian SKA Pathfinder (ASKAP).

The packages in _attic_ were used to benchmark a variety of platforms for the
original ASKAP Science Data Processor, and were made widely available to
vendors.

The packages in _current_ are more closely aligned with the current processing
approach and parameters, and are availavle for on-going benchmarking and
acceptance testing.

Hogbom Clean Benchmark (tHogbomClean)
-------------------------------------
The tHogbomClean benchmark implements the kernel of the
[Hogbom Clean](http://cdsads.u-strasbg.fr/abs/1974A%26AS...15..417H)
deconvolution algorithm. This benchmark is quite minimal and actually omits the
final step, convolution of the model with the clean beam, but this involves the
similar operations to the other steps as far as the CPU is concerned.

Execution of the tHogbomClean benchmarks will require the existence of the
point spread function (PSF) image and the dirty image (the image to be cleaned)
in the working directory. These are available in the _data_ directory.

The following tHogbomClean benchmarks are available in the _current_ directory

### tHogbomCleanOMP
This implementation uses OpenMP to utilize multiple cores in a single
shared-memory system.

### tHogbomCleanACC
This implementation uses OpenACC to utilize multiple cores in either a single
shared-memory CPU system or a single GPU.

Note that older, unmantained versions of these benchmarks are available for a
range of platforms in the _attic_ sub-directory.

Convolutional Resamping Benchmark (tConvolve)
---------------------------------------------
This benchmark suite include parallel implementations of
[Tim Cornwell's](http://www.atnf.csiro.au/people/tim.cornwell/) original
[tConvolveBLAS](http://wfit.googlecode.com/svn-history/r1088/wfit/doc/code/tConvolveBLAS.cc)
benchmark. The tConvolve benchmark programs measures the performance of a
convolutional resampling algorithm as used in radio astronomy data processing.
The benchmark is configured to reflect the computing needs of the Australian
Square Kilometer Array Pathfinder (ASKAP) Science Data Processor. A more
detailed description and some analysis of this algorithm is found in
[SKA Memo 132](http://www.skatelescope.org/uploaded/59116_132_Memo_Humphreys.pdf).

The following tHogbomClean benchmarks are available in the _current_ directory

### tConvolveMPI
The implementation distributes work to multiple CPU cores or multiple nodes via
Message Passing Interface (MPI) much like the ASKAP software, and while it is
possible to benchmark an entire cluster the aim of the benchmark is primarily
to benchmark a single compute node.

### tConvolveACC
This implementation uses OpenACC to utilize multiple cores in either a single
shared-memory CPU system or a single GPU.

Note that older, unmantained versions of these benchmarks are available for a
range of platforms in the _attic_ sub-directory.

Combined Benchmark (tMajor)
---------------------------
This benchmark combines Convolutional Resamping and Hogbom Clean in a single
major cycle / minor cycle imaging and deconvolution loop. It is currently under
construction

### tMajorACC
Currently under construction
**Todo**: update this document

Other Benchmarks
----------------
**Todo**: update this document

### msperf
Benchmark to measure the performance of a filesystem.

### mpiperf
Benchmark to measure the performance of a mpi gather.

Instructions
------------

### tConvolveMPI

```text
$ srun -N 1 -n  1 ./tConvolveMPI > tConvolveMPI_np01.out
$ srun -N 1 -n  2 ./tConvolveMPI > tConvolveMPI_np02.out
$ srun -N 1 -n  4 ./tConvolveMPI > tConvolveMPI_np04.out
$ srun -N 1 -n  8 ./tConvolveMPI > tConvolveMPI_np08.out
$ srun -N 1 -n 12 ./tConvolveMPI > tConvolveMPI_np12.out
$ srun -N 1 -n 16 ./tConvolveMPI > tConvolveMPI_np16.out
$ srun -N 1 -n 20 ./tConvolveMPI > tConvolveMPI_np20.out
$ srun -N 1 -n 24 ./tConvolveMPI > tConvolveMPI_np24.out
```

```text
$ grep 't2   Gridding rate (per node)' tConvolveMPI_np??.out | grep 'Mpix/sec'
tConvolveMPI_np01.out: t2   Gridding rate (per node)   254.954 (Mpix/sec)
tConvolveMPI_np02.out: t2   Gridding rate (per node)   255.748 (Mpix/sec)
tConvolveMPI_np04.out: t2   Gridding rate (per node)   204.217 (Mpix/sec)
tConvolveMPI_np08.out: t2   Gridding rate (per node)   171.032 (Mpix/sec)
tConvolveMPI_np12.out: t2   Gridding rate (per node)   157.875 (Mpix/sec)
tConvolveMPI_np16.out: t2   Gridding rate (per node)   145.817 (Mpix/sec)
tConvolveMPI_np20.out: t2   Gridding rate (per node)   129.284 (Mpix/sec)
tConvolveMPI_np24.out: t2   Gridding rate (per node)   113.235 (Mpix/sec)
```

```text
$ grep 't2   Degridding rate (per node)' tConvolveMPI_np??.out | grep 'Mpix/sec'
tConvolveMPI_np01.out: t2   Degridding rate (per node) 177.311 (Mpix/sec)
tConvolveMPI_np02.out: t2   Degridding rate (per node) 177.311 (Mpix/sec)
tConvolveMPI_np04.out: t2   Degridding rate (per node) 148.723 (Mpix/sec)
tConvolveMPI_np08.out: t2   Degridding rate (per node) 123.266 (Mpix/sec)
tConvolveMPI_np12.out: t2   Degridding rate (per node) 113.863 (Mpix/sec)
tConvolveMPI_np16.out: t2   Degridding rate (per node) 107.454 (Mpix/sec)
tConvolveMPI_np20.out: t2   Degridding rate (per node) 103.525 (Mpix/sec)
tConvolveMPI_np24.out: t2   Degridding rate (per node) 97.8489 (Mpix/sec)
```

### tHogbomCleanOMP

```text
$ cp ../../data/*.img .
$ export OMP_NUM_THREADS=1
$ srun -N 1 -n  1 -c 1 ./tHogbomCleanOMP > tHogbomCleanOMP_nt01.out
$ export OMP_NUM_THREADS=4
$ srun -N 1 -n  1 -c 4 ./tHogbomCleanOMP > tHogbomCleanOMP_nt04.out
$ export OMP_NUM_THREADS=8
$ srun -N 1 -n  1 -c 8 ./tHogbomCleanOMP > tHogbomCleanOMP_nt08.out
$ export OMP_NUM_THREADS=12
$ srun -N 1 -n  1 -c 12 ./tHogbomCleanOMP > tHogbomCleanOMP_nt12.out
$ export OMP_NUM_THREADS=16
$ srun -N 1 -n  1 -c 16 ./tHogbomCleanOMP > tHogbomCleanOMP_nt16.out
$ export OMP_NUM_THREADS=20
$ srun -N 1 -n  1 -c 20 ./tHogbomCleanOMP > tHogbomCleanOMP_nt20.out
$ export OMP_NUM_THREADS=24
$ srun -N 1 -n  1 -c 24 ./tHogbomCleanOMP > tHogbomCleanOMP_nt24.out
```

Note that when the number of threads is greater than 12, the OpenMP cleaning
rate can vary significantly. This is presumably due to NUMA issues. The speedups
shown below are the maximum values achieved in across several repeated runs.

```text
$ grep speedup tHogbomCleanOMP_nt??.out
tHogbomCleanOMP_nt01.out:    Number of threads =  1, speedup = 0.958801
tHogbomCleanOMP_nt04.out:    Number of threads =  4, speedup = 3.36842
tHogbomCleanOMP_nt08.out:    Number of threads =  8, speedup = 6.375
tHogbomCleanOMP_nt12.out:    Number of threads = 12, speedup = 8.93103
tHogbomCleanOMP_nt16.out:    Number of threads = 16, speedup = 10.24
tHogbomCleanOMP_nt20.out:    Number of threads = 20, speedup = 10.7083
tHogbomCleanOMP_nt24.out:    Number of threads = 24, speedup = 12.1905
```

