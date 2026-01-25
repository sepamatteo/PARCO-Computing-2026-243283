#ifndef COMMUNICATION_H
#define COMMUNICATION_H

/*
 * @file communication.h
 * @brief Ghost exchange pattern construction and vector value communication
 *        for distributed SpMV with 1D cyclic column distribution of vector x
*/

#include <mpi.h>
#include <vector>
#include <set>
#include <unordered_map>
#include <iostream>
#include <algorithm>

void exchange_ghosts(int rank, int size, int N, int local_nnz,
                     const std::vector<int>& local_col_idx,
                     const std::vector<double>& local_x,
                     std::unordered_map<int, double>& ghost_x);

/**
 * Structure that describes the ghost communication pattern.
 * Filled once during setup phase — reused in every SpMV iteration.
*/
struct GhostExchange {
    // Ghost columns needed from each rank (global indices)
    std::vector<std::vector<int>> ghosts_from_rank;

    // Communication metadata
    std::vector<int> send_counts, recv_counts;
    std::vector<int> send_disp, recv_disp;

    // Flat list of ghost columns (same order as recv buffer)
    std::vector<int> ghost_cols;
};

/**
 * @brief One-time construction of ghost communication pattern
 *
 * Determines — for every remote MPI rank — which global columns this process
 * needs to receive in every SpMV iteration (because it has nonzeros pointing to them).
 *
 * Uses 1D cyclic distribution of vector x:
 *     column j belongs to rank (j % size)
 *
 * Output: fully filled GhostExchange structure ready for repeated use in exchange_ghost_values()
 *
 * @param rank          this MPI process rank
 * @param size          total number of MPI processes
 * @param N             global number of columns
 * @param local_col_idx CSR column indices of local matrix rows (global numbering)
 * @param ghost         [out] communication metadata structure (filled by this function)
*/
void build_ghost_structure(
    int rank, int size, int N,
    const std::vector<int>& local_col_idx,
    GhostExchange& ghost
);

/**
 * @brief Exchange ghost values of vector x using precomputed pattern
 *
 * Uses two MPI_Alltoallv phases:
 *   1. Exchange list of requested column indices
 *   2. Exchange the actual double values
 *
 * @param rank          this process rank
 * @param size          number of processes
 * @param ghost         precomputed ghost communication pattern
 * @param local_x       local part of vector x (cyclic distribution)
 * @param ghost_values  [out] received ghost values (in order of ghost.ghost_cols)
*/
void exchange_ghost_values(
    int rank, int size,
    const GhostExchange& ghost,
    const std::vector<double>& local_x,
    std::vector<double>& ghost_values
);


#endif
