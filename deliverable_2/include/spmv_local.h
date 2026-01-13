#ifndef SPMV_LOCAL_H
#define SPMV_LOCAL_H

#include <vector>
#include <unordered_map>
#include <iostream>
#include <mpi.h>

void compute_local_spmv(int rank, int size, int local_M,
                        const std::vector<int>& local_row_ptr,
                        const std::vector<int>& local_col_idx,
                        const std::vector<double>& local_values,
                        const std::vector<double>& local_x,
                        const std::unordered_map<int, double>& ghost_x,
                        std::vector<double>& y_local);

#endif