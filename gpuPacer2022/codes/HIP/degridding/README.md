# Degridding
## Version 0
- Initial commit
- Tested in a local computer
- GPU and CPU results are different

## Version 1
- From Mitch's branch

## Version 2
- A new launch configuration implemented
    - kernel<<<***gridSize***, ***blockSize***>>>(...)
    - ***gridSize***
        - Number of data to be produced after degridding
        - Each block will priduce a single datum
    - ***blockSize***
        - dim3(blockSize1, blockSize2)
        - Each block size equals to ***BLOCK_SIZE*** (32).
        - Blocks here are 2 dimensional
            - threads in the 1st direction are responsible for the loop **suppU**
            - threads in the 2nd direction are responsible for the loop **suppV**
- Hence, the inner suppV loop is eliminated.
- A **block-stride loop** is employed for **data load** from **global** to **shared** memory.
- A parallel sweep is implemented.
- More modifications are needed for further performance improvements
- Tests will be performed on Topaz and Mulan for different data size 

## Version 3 ***NOT WORKING FOR BIG DATA***
- ***cind*** and ***gind*** are not shared anymore
- Avoided 1 ***__syncthreads***

## Version 4
- ***cind*** and ***gind*** are shared again
- Half of the threads in parallel sweep were idle in Versions 2 and 3
- The idle ones are now working on the imaginary part
- Stolen and adapted from **Mitch**'s branch

