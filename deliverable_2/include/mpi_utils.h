#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include <vector>
#include <mpi.h>

// Assumes local_row_ptr/col_idx/values from main_mpi.cpp (your Deliv1 layout)
// x: full global vector< double > (local copy, updated with ghosts)
// ghosts: temp vector for received remote x values (size local_nnz worst-case)
void exchange_ghost_x(const std::vector<int>& local_col_idx, int N, int rank, int size,
                      std::vector<double>& x, std::vector<double>& ghosts_out);

// Gather distributed y_local (local_M elems) to root's full y (M elems)
void gather_y_to_root(const std::vector<double>& y_local, int local_M, int M,
                      int rank, int root, std::vector<double>& y_global);

// Local SpMV (exact reuse of your deliv1 vector logic, single-thread)
void local_spmv(const std::vector<int>& row_ptr, const std::vector<int>& col_idx,
                const std::vector<double>& values, const std::vector<double>& x_in,
                std::vector<double>& y_out);

#endif
