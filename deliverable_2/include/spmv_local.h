#ifndef SPMV_LOCAL_H
#define SPMV_LOCAL_H

#include <vector>
#include <iostream>
#include <mpi.h>

#include "../include/communication.h"

/**
 * @brief Computes local portion of sparse matrix-vector product:  y_local = A_local * x
 *
 * Assumptions about data layout:
 *   - Rows are distributed by blocks → local_M = number of local matrix rows
 *   - Vector x is distributed cyclically → column j belongs to rank (j % size)
 *   - All non-local columns appearing in local_col_idx[] must be present in ghost_values[]
 *     (prepared by exchange_ghost_values() using the pattern built in build_ghost_structure())
 *
 * Performance notes:
 *   - Uses OpenMP parallelization over local rows
 *   - Uses unordered_map for O(1) average-case ghost lookup
 *     → good when number of ghosts is not extremely large and keys are well-distributed
 *   - Schedule(guided,64) helps with very imbalanced row nnz counts
 *
 * @param rank              MPI rank (mainly for error messages)
 * @param size              number of MPI processes (used to compute column owners)
 * @param local_M           number of matrix rows owned by this process
 * @param local_row_ptr     CSR row pointers (size = local_M + 1)
 * @param local_col_idx     CSR column indices — **global** numbering!
 * @param local_values      CSR nonzero values
 * @param local_x           local part of input vector x (cyclic distribution)
 * @param ghost_values      received values of remote x entries (order matches ghost.ghost_cols)
 * @param ghost             ghost communication metadata (contains ghost_cols list)
 * @param y_local           [out] result vector — only local rows (size = local_M)
 */
void compute_local_spmv(int rank, int size, int local_M,
                        const std::vector<int>& local_row_ptr,
                        const std::vector<int>& local_col_idx,
                        const std::vector<double>& local_values,
                        const std::vector<double>& local_x,
                        const std::vector<double>& ghost_values,
                        const GhostExchange& ghost,
                        std::vector<double>& y_local);


#endif
