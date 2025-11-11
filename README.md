# Sepa Matteo PARCO-Computing-2026-243283

This repository contains code and materials related to parallel computing with a focus on sparse matrix-vector multiplication (SpMV) optimized for shared-memory parallelization. It is part of the coursework or research activities in the "Introduction to Parallel Computing" academic year 2025-2026.

**Written By: Sepa Matteo (243283)**

---

## About

Parallel computing project exploring efficient implementations of sparse matrix operations, particularly SpMV, which is a fundamental linear algebra operation used extensively in scientific computing and machine learning. Sparse matrices are those with mostly zero entries, requiring specialized storage and algorithms for efficiency.

The project includes:
- Reading and converting matrix files to a compressed sparse row (CSR) format.
- Implementing sequential and parallel SpMV routines.
- Benchmarking performance using multiple sparsity patterns and threading configurations.

---

## Features

- Implementation of COO (Coordinate) and CSR (Compressed Sparse Row) sparse matrix storage formats.
- Sequential and OpenMP-based parallel SpMV algorithms.
- Support for reading matrix data from Matrix Market (.mtx) files.
- Performance benchmarking over multiple matrices covering different sparsity degrees.
- Tools for cache profiling and performance analysis.

---

## Getting Started

1. **Clone the repository:**

```bash
git clone https://github.com/sepamatteo/PARCO-Computing-2026-243283.git
cd PARCO-Computing-2026-243283
```


2. **Build the code:**
```bash
make
```

3. **Run SpMV on example datasets:**

- Download desired .mtx files from https://sparse.tamu.edu/

using

```wget <url.matrix_name>.tar.gz```

- Extract inside the ```\data``` directory
 
using 

- ```gzip -d <matrix_name>.tar.gz```
- ```tar -xvf <matrix_name>.tar```

## Documentation

The PDF report is available in the ```/docs``` directory

## Usage

1. **Using runner script**

Run ```src/runner.sh``` for automated multiple instance execution and benchmarking

**Runner script flags:**
- ```--verbose``` enables verbose mode
- ```--coo``` runs coo implementation
- ```--seq-csr``` runs sequential csr implementation
- ```--par-csr``` runs parallel csr implementation
- ```--show-plot``` shows plot after benchmark
- ```--cachegrind``` runs selected implementations with cachegrind monitoring
- ```--python``` to run the python benchmark data analysis script
- ```--matrix``` select matrix file if not default is used
- ```--threads``` select number of threads to run in the parallel csr implementation

2. **Manually running**

Manually run ```/outputs/<executable> ../data/<matrix_name>/<matrix_name>.mtx```

## Running on the cluster

There are 3 PBS scripts in the ```/jobs``` directory:

- ```CompileSpMV.pbs``` compiles and links the source using the MAKEFILE
- ```RunSpMV.pbs``` runs the multiplication and benchmark using the ```runner.sh``` script, modify the arguments passed to the script to adjust the running parameters
- ```SpMVCachegrind``` runs with cachegrind for cache miss analysis

Run the desired script using ```qsub <scriptName>.pbs```

## License

This project is licensed under the GPL-3.0 License.

---