ASKAP Benchmark Packages
========================

These benchmark packages were used to benchmark a variety of platforms for the Australian
SKA Pathfinder (ASKAP) Science Data Processor. These benchmarks have been made widely
available to vendors.

Hogbom Clean Benchmark (tHogbomClean)
-------------------------------------
The tHogbomClean benchmark implements the kernel of the [Hogbom Clean](http://cdsads.u-strasbg.fr/abs/1974A%26AS...15..417H)
deconvolution algorithm. This simple benchmark was written to benchmark the Intel Xeon
Phi accelerator, previously known as Many Integrated Cores (MIC).  This benchmark is
quite minimal and actually omits the final step, convolution of the model with the
clean beam, but this involves the similar operations to the other steps as far as
the CPU is concerned.

Execution of both the tHogbomClean benchmarks will require the existence of the point spread
function (PSF) image and the dirty image (the image to be cleaned) the working directory.
These can be downloaded from here:

* http://www.atnf.csiro.au/people/Ben.Humphreys/dirty.img
* http://www.atnf.csiro.au/people/Ben.Humphreys/psf.img

### tHogbomCleanMIC
This original implementation of the benchmark targets the Intel Xeon Phi accelerator.

### tHogbomCleanOMP
This implementation uses OpenMP to utilize multiple cores in a single shared-memory system.

### tHogbomCleanCuda
This is implemented in NVIDIA CUDA and executes on a single NVIDIA GPU. Note, a
more portable version of this benchmark implemented by Mark Harris of NVIDIA can be found
here: https://github.com/harrism/tHogbomCleanHemi


Convolutional Resamping Benchmark (tConvolve)
---------------------------------------------
This benchmark suite include parallel implementations of [Tim Cornwell's](http://www.atnf.csiro.au/people/tim.cornwell/)
original [tConvolveBLAS](http://wfit.googlecode.com/svn-history/r1088/wfit/doc/code/tConvolveBLAS.cc)
benchmark. The tConvolve benchmark programs measures the performance of a convolutional
resampling algorithm as used in radio astronomy data processing. The benchmark is
configured to reflect the computing needs of the Australian Square Kilometer Array
Pathfinder (ASKAP) Science Data Processor. A more detailed description and some analysis
of this algorithm is found in [SKA Memo 132](http://www.skatelescope.org/uploaded/59116_132_Memo_Humphreys.pdf).

### tConvolveMPI
The implementation distributes work to multiple CPU cores or multiple nodes via Message
Passing Interface (MPI) much like the ASKAP software, and while it is possible to
benchmark an entire cluster the aim of the benchmark is primarily to benchmark a single
compute node.

### tConvolveCuda
This is implemented in NVIDIA CUDA and executes on a single NVIDIA GPU.

### tConvolveHIP 
This is the HIP implementation and executes on a single NVIDIA/AMD GPU.

### tConvolveOpenCL
This is the OpenCl implementation and executes on a single NVIDIA/AMD GPU.

### tConvolveHIPCPU
This is the HIP CPU implementation and executes on CPUs

### tConvolveCommon
This contains common functions used in all implementations of the convolution function.
