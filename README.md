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
- ```tar -xvf <matrix_name.tar>```

## Usage

1. **Using runner script**

Modify and run ```src/runner.sh``` for automated multiple instance execution and benchmarking

2. **Manually running**

Manually run ```/outputs/<executable> ../data/<matrix_name>/<matrix_name>.mtx```

## License

This project is licensed under the GPL-3.0 License.

---