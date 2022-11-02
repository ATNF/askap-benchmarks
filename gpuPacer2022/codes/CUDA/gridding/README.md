# Gridding
## Solvers
### cpu
- CPU version
- Using a single node, single thread (a serial version)

### gpuOlder
- Developed during the Nvidia-CUDA hackathon in 2017.

### gpuAtomic
- Atomic additions were added to the older version
- ***This solver gives the best performance so far***

### gpuAtomicTiled
- Developed with the aim of improving ***gpuAtomic*** version by improving the kernel launch configuration.
- Slower than ***gpuAtomic***



