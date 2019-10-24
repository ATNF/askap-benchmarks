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

The following tHogbomClean benchmarks are available in the _current_ directory.
Cleaning generally takes place on a single process, and as such multi-threading
is a natural approach to parallelism.

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

The following tConvolve benchmarks are available in the _current_ directory.
As gridding is independent for each frequency, Taylor term, etc., data-parallelism
is a natural approach.

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

The following examples were generated on a single node of Magnus at the Pawsey
Supercomputing Centre. 

### tHogbomCleanOMP

```text
$ cd current/tHogbomCleanOMP
$ cp ../../data/dirty_4096.img dirty.img
$ cp ../../data/psf_4096.img psf.img
$ export OMP_PROC_BIND=true
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
rate can vary significantly unless ```text OMP_PROC_BIND``` is set.

```text
$ grep speedup tHogbomCleanOMP_nt??.out
tHogbomCleanOMP_nt01.out:    Number of threads = 1,  speedup = 0.96139
tHogbomCleanOMP_nt04.out:    Number of threads = 4,  speedup = 3.3542
tHogbomCleanOMP_nt08.out:    Number of threads = 8,  speedup = 5.88056
tHogbomCleanOMP_nt12.out:    Number of threads = 12, speedup = 7.66545
tHogbomCleanOMP_nt16.out:    Number of threads = 16, speedup = 9.79861
tHogbomCleanOMP_nt20.out:    Number of threads = 20, speedup = 11.258
tHogbomCleanOMP_nt24.out:    Number of threads = 24, speedup = 12.158
```

### tConvolveMPI

```text
$ cd current/tConvolveMPI
$ srun -N 1 -n  1 ./tConvolveMPI > tConvolveMPI_np01.out
$ srun -N 1 -n  4 ./tConvolveMPI > tConvolveMPI_np04.out
$ srun -N 1 -n  8 ./tConvolveMPI > tConvolveMPI_np08.out
$ srun -N 1 -n 12 ./tConvolveMPI > tConvolveMPI_np12.out
$ srun -N 1 -n 16 ./tConvolveMPI > tConvolveMPI_np16.out
$ srun -N 1 -n 20 ./tConvolveMPI > tConvolveMPI_np20.out
$ srun -N 1 -n 24 ./tConvolveMPI > tConvolveMPI_np24.out
```

```text
$ grep 'Continuum gridding performance' tConvolveMPI_np??.out
tConvolveMPI_np01.out:    Continuum gridding performance:   0.778712 (Mvis/sec) / 0.093312 (Mpix/sec) = 8.34526x requirement
tConvolveMPI_np04.out:    Continuum gridding performance:   0.63073  (Mvis/sec) / 0.093312 (Mpix/sec) = 6.75937x requirement
tConvolveMPI_np08.out:    Continuum gridding performance:   0.518894 (Mvis/sec) / 0.093312 (Mpix/sec) = 5.56085x requirement
tConvolveMPI_np12.out:    Continuum gridding performance:   0.469241 (Mvis/sec) / 0.093312 (Mpix/sec) = 5.02874x requirement
tConvolveMPI_np16.out:    Continuum gridding performance:   0.419676 (Mvis/sec) / 0.093312 (Mpix/sec) = 4.49756x requirement
tConvolveMPI_np20.out:    Continuum gridding performance:   0.369281 (Mvis/sec) / 0.093312 (Mpix/sec) = 3.95749x requirement
tConvolveMPI_np24.out:    Continuum gridding performance:   0.324967 (Mvis/sec) / 0.093312 (Mpix/sec) = 3.48259x requirement
```

```text
$ grep 'Continuum degridding performance' tConvolveMPI_np??.out
tConvolveMPI_np01.out:    Continuum degridding performance:   0.550931 (Mvis/sec) / 0.093312 (Mpix/sec) = 5.90418x requirement
tConvolveMPI_np04.out:    Continuum degridding performance:   0.461288 (Mvis/sec) / 0.093312 (Mpix/sec) = 4.9435x requirement
tConvolveMPI_np08.out:    Continuum degridding performance:   0.383865 (Mvis/sec) / 0.093312 (Mpix/sec) = 4.11378x requirement
tConvolveMPI_np12.out:    Continuum degridding performance:   0.349371 (Mvis/sec) / 0.093312 (Mpix/sec) = 3.74412x requirement
tConvolveMPI_np16.out:    Continuum degridding performance:   0.330492 (Mvis/sec) / 0.093312 (Mpix/sec) = 3.54179x requirement
tConvolveMPI_np20.out:    Continuum degridding performance:   0.318689 (Mvis/sec) / 0.093312 (Mpix/sec) = 3.4153x requirement
tConvolveMPI_np24.out:    Continuum degridding performance:   0.300729 (Mvis/sec) / 0.093312 (Mpix/sec) = 3.22284x requirement
```

```text
$ grep 'Spectral gridding performance' tConvolveMPI_np??.out
tConvolveMPI_np01.out:    Spectral gridding performance:    5.62871 (Mvis/sec) / 0.164003 (Mpix/sec) = 34.3208x requirement
tConvolveMPI_np04.out:    Spectral gridding performance:    5.54713 (Mvis/sec) / 0.164003 (Mpix/sec) = 33.8234x requirement
tConvolveMPI_np08.out:    Spectral gridding performance:    4.97081 (Mvis/sec) / 0.164003 (Mpix/sec) = 30.3093x requirement
tConvolveMPI_np12.out:    Spectral gridding performance:    4.20607 (Mvis/sec) / 0.164003 (Mpix/sec) = 25.6463x requirement
tConvolveMPI_np16.out:    Spectral gridding performance:    3.544   (Mvis/sec) / 0.164003 (Mpix/sec) = 21.6094x requirement
tConvolveMPI_np20.out:    Spectral gridding performance:    3.03771 (Mvis/sec) / 0.164003 (Mpix/sec) = 18.5223x requirement
tConvolveMPI_np24.out:    Spectral gridding performance:    2.658   (Mvis/sec) / 0.164003 (Mpix/sec) = 16.207x requirement
```

```text
$ grep 'Spectral degridding performance' tConvolveMPI_np??.out
tConvolveMPI_np01.out:    Spectral degridding performance:    4.61147 (Mvis/sec) / 0.164003 (Mpix/sec) = 28.1182x requirement
tConvolveMPI_np04.out:    Spectral degridding performance:    4.50296 (Mvis/sec) / 0.164003 (Mpix/sec) = 27.4566x requirement
tConvolveMPI_np08.out:    Spectral degridding performance:    4.11561 (Mvis/sec) / 0.164003 (Mpix/sec) = 25.0948x requirement
tConvolveMPI_np12.out:    Spectral degridding performance:    3.61087 (Mvis/sec) / 0.164003 (Mpix/sec) = 22.0171x requirement
tConvolveMPI_np16.out:    Spectral degridding performance:    3.16324 (Mvis/sec) / 0.164003 (Mpix/sec) = 19.2877x requirement
tConvolveMPI_np20.out:    Spectral degridding performance:    2.77357 (Mvis/sec) / 0.164003 (Mpix/sec) = 16.9117x requirement
tConvolveMPI_np24.out:    Spectral degridding performance:    2.40725 (Mvis/sec) / 0.164003 (Mpix/sec) = 14.6781x requirement
```

### tConvolveACC

Note that the performance numbers quoted here are the same as those quoted in
tConvolveMPI. That is, assuming ~ 12,000 and 15,000 separate processing units
for continuum and spectral line observing respectively. If GPUs were used for
gridding instead of multi-core CPUs, the reduction in processing units would
need to be accounted for in the performance numbers (where here it is assumed
that each GPU is a single processing unit or MPI process).

