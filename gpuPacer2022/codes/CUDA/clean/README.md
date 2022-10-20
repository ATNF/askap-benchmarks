# Deconvolution
## Introduction
This section of the repo includes the serial and accelerated codes for deconvolution, which use the Hogbom's CLEAN algorithm. This algorithm is summarized as follows:
- Start with the dirty image
- Find the brightest pixel
- Subtract a fraction (around 0.1) of the dirty beam from the dirty image at the location of that brightest pixel.
- The subtracted image is called the residual image.
- Find the new brightest pixel.
- Continue the loop until a threshold is reached.

## The United Versions
- All the solver classes are located in the folder src/Solvers/. 
- The strategy and factory patterns were applied to have a single version with various solvers.
- In ***main()***, there are two solvers introduced
    1. Reference solver: The solver, which was verified to be correct before (e.g. CPU version "Golden") that we are going to use to verify the newly developed kernel.
    2. Test solver: The one that we develop and verify using the reference solver.
- Every GPU solver introduced here can be used as a reference solver.
- The reference and test solvers can be assigned via the file: Parameters.h.

## Solvers
### Golden
- The serial CPU solver

### CudaOlder
- Older version from the previous hackathon
- Uses the shared memory with a classical reduction approach

### CudaPS
- Parallel sweep inplemented (aka a tree approach or sequential addressing)
- A new, 2-step reduction was adapted to extract the max index
- A grid-stride data load implemented

### CudaPSLastWUnrolled
- The last warp unrolled
    - As reduction proceeds, # of active threads decreases
    - Only 1 warp left when s <= 32
- Hence, no need for **__syncthreads** for the last warp

### CudaPSFullUnroll
- Complete unrolling
    - Use template
