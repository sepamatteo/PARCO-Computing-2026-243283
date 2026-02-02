# Distributed MPI + OpenMP SpMV – Deliverable 2  

![HiCREST + UniTrento](https://www2.almalaurea.it/img/logofull/70062.png)  

**Introduction to Parallel Computing**   a.y. 2025–2026 – University of Trento \
**Deliverable:** 2 – Distributed-memory SpMV using MPI-X \
**Written by: Matteo Sepa (243283)**   

## Project Overview

This project implements a **distributed Sparse Matrix-Vector Multiplication (SpMV)** using:

- **MPI** for distributed-memory parallelism (cyclic row distribution of the matrix, cyclic column distribution of the vector)
- **OpenMP** for shared-memory intra-node parallelism on local rows
- **CSR** storage format
- Ghost (halo) exchange via two-phase `MPI_Alltoallv` to handle non-local vector accesses
- One-time construction of communication pattern + fast per-iteration value exchange
- Detailed performance metrics collection


## Features

- Cyclic (1D block-cyclic) distribution of matrix rows
- Cyclic distribution of vector columns → requires ghost communication
- Static ghost pattern construction (`build_ghost_structure`)
- Efficient ghost value exchange (`exchange_ghost_values`)
- Precomputed column access metadata
- Hybrid MPI + OpenMP parallelism (configurable threads per rank)
- Warm-up + timed benchmark iterations
- Comprehensive metrics reporting

## Directory Structure
```text
.
|
├─ data/                       # Contains matrix market formatted matrices
├─ docs/
|  └─ report/
├─ include/
|  ├─ communication.hpp
│  ├─ distribution.hpp
|  ├─ matrix_gen.hpp
│  ├─ main_mpi.hpp              
│  ├─ matrix_io.hpp
│  ├─ metrics.hpp
│  ├─ mmio.h
|  └─spmv_local.hpp
├─ jobs/
|  └─ mpi.pbs                   # PBS script
├─ outputs/
|  └─ spmv_mpi                  # Executable
├─ src/
│  ├─ communication.cpp
│  ├─ distribution.cpp
│  ├─ main_mpi.cpp              # Main function
|  ├─ matrix_gen.cpp
│  ├─ matrix_io.cpp
│  ├─ metrics.cpp
│  ├─ mmio.c
|  └─spmv_local.cpp
├─ MAKEFILE
└─ README.md

```
### Prerequisites
- Tested on C++11 standard compiler 
- MPI implementation (OpenMPI or MPICH)
- OpenMP support

Tested on HPC Cluster

## Build Instructions

```bash
# Clone the repository
git clone https://github.com/sepamatteo/PARCO-Computing-2026-243283.git
cd PARCO-Computing-2026-243283/deliverable_2

# Build
make clean && make

The executable will be named spmv_mpi.

```

# Usage
**ALWAYS** run by modifying the provided PBS script  ```jobs/mpi.pbs```

### Most common options
``` bash
mpirun ./spmv_mpi ../data/<matrix>/<matrix>.mtx -np <P> [options]

Options:
  --threads T / -t      Number of OpenMP threads per MPI rank (default: 1)
  --verbose / -v          Print warm-up and benchmark progress
```

# Report
The full report for Deliverable 2 is available at ```docs/report/report.pdf```

The report is a LaTeX-based report in IEEEtran format \
It contains methodology, data distribution strategy, ghost exchange explanation, hybrid OpenMP scheduling analysis, strong scaling results, load balance discussion, and conclusions.

# References / Acknowledgments

- Matrix Market I/O library (mmio.h / mmio.c)
- SuiteSparse Matrix Collection: https://sparse.tamu.edu
- Course slides & material by Prof. Flavio Vella

# License
This project is licensed under the GPL-3.0 License.