## GPU101 PiA Project
<p align="justify">
This repository contains my implementation of an algorithm that speeds up the calculation of a Sparse Matrix-Vector Multiplication through parallelizing the process.
It is implemented using C CUDA.
</p>

## Contents
- the "spmv" folder contains two source code.
    - the "spmv-csr.c" file is the prompt for my project; the original algorithm to calculate the SPMV operation on CPU
    - the "spmv-parallelized.cu" file is my acceleration of the algorithm in GPU
- the "doc" folder contains a pdf explaining the project in detail

## Usage

To compile the "spmv-parallelized.cu" you only need NVCC
```
nvcc spmv-parallelized.cu -o spmv
```
