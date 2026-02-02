#include "../include/communication.hpp"

#include <cstddef>
#include <mpi.h>
#include <vector>
#include <set>
#include <iostream>
#include <cassert>

/*
 * Build ghost communication pattern ONCE.
 * Identifies which global x entries are needed from which ranks.
 */
void build_ghost_structure(
    int rank, int size, int N,
    const std::vector<int>& local_col_idx,
    GhostExchange& ghost
) {
    std::vector<std::set<int>> needed(size);

    // Discover required ghost columns
    for (int j : local_col_idx) {
        if (j < 0 || j >= N) {
            std::cerr << "Rank " << rank << ": invalid column index " << j << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int owner = j % size;
        if (owner != rank) {
            needed[owner].insert(j);
        }
    }

    // Convert to vectors
    ghost.ghosts_from_rank.resize(size);
    for (int p = 0; p < size; ++p) {
        ghost.ghosts_from_rank[p].assign(
            needed[p].begin(),
            needed[p].end()
        );
    }

    // Build send counts and displacements
    ghost.send_counts.resize(size);
    ghost.send_disp.resize(size + 1, 0);

    for (int p = 0; p < size; ++p) {
        ghost.send_counts[p] = ghost.ghosts_from_rank[p].size();
        ghost.send_disp[p + 1] = ghost.send_disp[p] + ghost.send_counts[p];
    }

    // Exchange counts
    ghost.recv_counts.resize(size);
    MPI_Alltoall(
        ghost.send_counts.data(), 1, MPI_INT,
        ghost.recv_counts.data(), 1, MPI_INT,
        MPI_COMM_WORLD
    );

    ghost.recv_disp.resize(size + 1, 0);
    for (int p = 0; p < size; ++p) {
        ghost.recv_disp[p + 1] = ghost.recv_disp[p] + ghost.recv_counts[p];
    }

    // Build flat ghost column list (same order as receive buffer)
    int total_ghosts = ghost.send_disp[size];
    ghost.ghost_cols.resize(total_ghosts);

    int offset = 0;
    for (int p = 0; p < size; ++p) {
        for (int j : ghost.ghosts_from_rank[p]) {
            ghost.ghost_cols[offset++] = j;
        }
    }

    assert(offset == total_ghosts);
    
    ghost.ghost_map.reserve(ghost.ghost_cols.size());
    for (size_t i = 0; i < ghost.ghost_cols.size(); ++i) {
        ghost.ghost_map[ghost.ghost_cols[i]] = static_cast<int>(i);
    }
}

/*
 * Exchange ghost vector values.
 * Only values are communicated — structure is reused.
 */
void exchange_ghost_values(
    int rank, int size,
    const GhostExchange& ghost,
    const std::vector<double>& local_x,
    std::vector<double>& ghost_values
) {
    // Step 1: receive requests for local x entries
    int total_recv_req = ghost.recv_disp[size];
    std::vector<int> recv_req_buf(total_recv_req);

    std::vector<int> send_req_buf = ghost.ghost_cols;

    MPI_Alltoallv(
        send_req_buf.data(),
        ghost.send_counts.data(),
        ghost.send_disp.data(),
        MPI_INT,
        recv_req_buf.data(),
        ghost.recv_counts.data(),
        ghost.recv_disp.data(),
        MPI_INT,
        MPI_COMM_WORLD
    );

    // Step 2: pack local values requested by others
    std::vector<double> send_val_buf(total_recv_req);

    for (int i = 0; i < total_recv_req; ++i) {
        int j = recv_req_buf[i];
        int local_idx = (j - rank) / size;

        if (local_idx < 0 || local_idx >= (int)local_x.size()) {
            std::cerr << "Rank " << rank
                      << ": invalid local index for column " << j << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        send_val_buf[i] = local_x[local_idx];
    }

    // Step 3: exchange values
    int total_send_vals = ghost.send_disp[size];
    ghost_values.resize(total_send_vals);

    MPI_Alltoallv(
        send_val_buf.data(),
        ghost.recv_counts.data(),
        ghost.recv_disp.data(),
        MPI_DOUBLE,
        ghost_values.data(),
        ghost.send_counts.data(),
        ghost.send_disp.data(),
        MPI_DOUBLE,
        MPI_COMM_WORLD
    );
}
