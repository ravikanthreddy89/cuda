Instructions for running the tests
======================================================

1) The file with the name "globla.cu" contains the globla memory implementation
2) The file with the name "shared_mem.cu" contains shared memory implementation
3) The slurm script file contains test run commands.
4) NOTE : make sure the object file name is "out_10", so that script file
finds the right object file to run.
5) For shared mmory the block size(number of threads per block)is static and
matrix sizes for testing should be multiples of 512.
