#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

/* -----------------------------------------------------------------------------
 Interface for distributing a global CSR matrix across MPI processes
 using a **1D block-cyclic row distribution** (simple cyclic by row index).

 Main strategy:
   - Rows are assigned to processes in round-robin fashion:
   - row i belongs to process (i % size)
   - Each process receives roughly M / size rows
   - Nonzeros (nnz) are distributed accordingly — no 2D block distribution

 Vector x is distributed **cyclically by columns** (j % size),
 which is why ghost communication is needed during SpMV.

 This header declares the main distribution function and any related types.

 Typical usage flow:
   1. rank 0 reads matrix → global CSR
   2. distribute_matrix() scatters rows & nnz to all processes
   3. each process builds local CSR structures
 -----------------------------------------------------------------------------*/

#include <mpi.h>
#include <vector>
#include <iostream>
#include <mpi.h>
#include <cassert>

/**
 * @brief Distributes global CSR matrix to all MPI processes using cyclic row distribution
 *
 * Only rank 0 needs to provide the full global CSR arrays.
 * All other ranks pass empty/zero-initialized vectors — they will be filled by this function.
 *
 * Distribution policy:
 *   - Rows are distributed cyclically: row r goes to process (r % size)
 *   - Each process gets approximately M / size rows
 *   - Corresponding nonzeros are sent with the rows
 *   - row_ptr is relative to the local matrix (starts at 0 for first local row)
 *
 * After this call:
 *   - local_row_ptr.size() == local_M + 1
 *   - local_col_idx.size() == local_nnz
 *   - local_values.size()  == local_nnz
 *   - column indices remain **global** (not renumbered locally)
 *
 * @param rank              This process's MPI rank
 * @param size              Total number of MPI processes
 * @param M                 Global number of rows
 * @param nz_global         Global number of nonzeros
 * @param global_row_ptr    Full row pointers (only meaningful on rank 0)
 * @param global_col_idx    Full column indices  (only meaningful on rank 0)
 * @param global_values     Full nonzero values   (only meaningful on rank 0)
 * @param local_row_ptr     [out] Local CSR row pointers
 * @param local_col_idx     [out] Local column indices (global numbering)
 * @param local_values      [out] Local nonzero values
 * @param local_M           [out] Number of rows this process owns
 * @param local_nnz         [out] Number of nonzeros this process owns
*/
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
