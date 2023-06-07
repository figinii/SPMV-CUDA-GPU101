## GPU101 PiA Project
<p align="justify">
This repository contains my implementation of an algorithm that speeds up the calculation of a Sparse Matrix-Vector Multiplication through parallelizing the process.
It is implemented using C CUDA.
</p>

## Usage

You only need NVCC to compile the source
the various programs, which you should already have installed if you did already setup your maching for CUDA.
To compile all the examples simply type
```
make
```
Within the scope of this folder.
Note that the examples are all compiled using the -O3 flag, you have to use this flag also when compiling the GPU version of the code using nvcc.
All the parameters regarding input settings CANNOT BE CHANGED.
The Smith-Waterman algorithm is the only one that generates all the inputs at runtime, while for SYMGS and SPMV you should also use a sparse matrix as input (the program will not run without it).
SPMV and SYMGS GPU implementation will not produce the same results as for the CPU implementation, even when implementing a correct design, this is due to how floating point operations are implemented on GPU, to assure correctness then, check that the error rate is below a certain threshold (<0.1% error rate should be fine).
