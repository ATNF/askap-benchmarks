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
```

```text
$ grep 't2   Degridding rate (per node)' tConvolveMPI_np??.out | grep 'Mpix/sec'
tConvolveMPI_np01.out: t2   Degridding rate (per node) 177.311 (Mpix/sec)
tConvolveMPI_np02.out: t2   Degridding rate (per node) 177.311 (Mpix/sec)
tConvolveMPI_np04.out: t2   Degridding rate (per node) 148.723 (Mpix/sec)
tConvolveMPI_np08.out: t2   Degridding rate (per node) 123.266 (Mpix/sec)
tConvolveMPI_np12.out: t2   Degridding rate (per node) 113.863 (Mpix/sec)
tConvolveMPI_np16.out: t2   Degridding rate (per node) 107.454 (Mpix/sec)
```

