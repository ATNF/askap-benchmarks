# Deconvolution
## Version 0
- Initial commit
- Tested in a local computer
- Results verified

## Version 1
- CMake implemented
- Tested in Topaz
- CPU version gives incorrect results
- CUDA version works well

## Version 2
- Parallel sweep inplemented (aka a tree approach or sequential addressing)
- A new, 2-step reduction was adapted to extract the max index
- A grid-stride data load implemented