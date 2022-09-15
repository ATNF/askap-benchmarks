# Deconvolution
- All versions are tested and verified on Topaz and Mulan.
## Version 1
- Preliminary model

## Version 2
- Parallel sweep inplemented (aka a tree approach or sequential addressing)
- A new, 2-step reduction was adapted to extract the max index
- A grid-stride data load implemented

## Version 3
- The last warp unrolled
    - As reduction proceeds, # of active threads decreases
    - Only 1 warp left when s <= 32
- Hence, no need for **__syncthreads** for the last warp

## Version 4
- Complete unrolling

