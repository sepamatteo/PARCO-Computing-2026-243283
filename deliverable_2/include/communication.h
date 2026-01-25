#ifndef COMMUNICATION_H
#define COMMUNICATION_H

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

struct GhostExchange {
    // Ghost columns needed from each rank (global indices)
    std::vector<std::vector<int>> ghosts_from_rank;

    // Communication metadata
    std::vector<int> send_counts, recv_counts;
    std::vector<int> send_disp, recv_disp;

    // Flat list of ghost columns (same order as recv buffer)
    std::vector<int> ghost_cols;
};

void build_ghost_structure(
    int rank, int size, int N,
    const std::vector<int>& local_col_idx,
    GhostExchange& ghost
);

void exchange_ghost_values(
    int rank, int size,
    const GhostExchange& ghost,
    const std::vector<double>& local_x,
    std::vector<double>& ghost_values
);


#endif
