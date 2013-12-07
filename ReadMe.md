LU decomposition using CUDA
=====================================================

1) A parallel implementation of LU decomposition 

2) Two versions : 1) Using global memory alone 2) Using shared memory for pivot row

3) For both the implementations single thread scales the pivot row
 
4) Global memory : Blocks with one thread each  are launched for reduction.  

5) Shared memory : Blocks with static size are used. Thread with id==0 copies the pivot row into shared memory and after that rest of the threads
                   in the block start reducing.


