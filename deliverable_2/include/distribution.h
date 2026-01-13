#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include <mpi.h>
#include <vector>
#include <iostream>
#include <mpi.h>
#include <cassert>

void distribute_matrix(int rank, int size, int M, int nz_global,
                       const std::vector<int>& global_row_ptr,
                       const std::vector<int>& global_col_idx,
                       const std::vector<double>& global_values,
                       std::vector<int>& local_row_ptr,
                       std::vector<int>& local_col_idx,
                       std::vector<double>& local_values,
                       int& local_M, int& local_nnz);

void init_local_vector(int rank, int size, int N,
                       std::vector<double>& local_x,
                       int& local_col_count);

#endif