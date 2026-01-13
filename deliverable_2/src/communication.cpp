#include "../include/communication.h"

void exchange_ghosts(int rank, int size, int N, int local_nnz,
                     const std::vector<int>& local_col_idx,
                     const std::vector<double>& local_x,
                     std::unordered_map<int, double>& ghost_x) {
    // Collect needed remote columns
    std::vector<std::set<int>> needed_sets(size);
    for (int k = 0; k < local_nnz; ++k) {
        int j = local_col_idx[k];
        if (j < 0 || j >= N) {
            std::cerr << "Rank " << rank << ": invalid col " << j << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int owner = j % size;
        if (owner != rank) {
            needed_sets[owner].insert(j);
        }
    }
    std::vector<std::vector<int>> needed_from(size);
    for (int p = 0; p < size; ++p) {
        needed_from[p].assign(needed_sets[p].begin(), needed_sets[p].end());
    }
    // Exchange requests
    std::vector<int> send_counts(size);
    for (int p = 0; p < size; ++p) send_counts[p] = needed_from[p].size();
    std::vector<int> recv_counts(size);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> send_disp(size + 1, 0);
    for (int p = 0; p < size; ++p) send_disp[p + 1] = send_disp[p] + send_counts[p];
    std::vector<int> recv_disp(size + 1, 0);
    for (int p = 0; p < size; ++p) recv_disp[p + 1] = recv_disp[p] + recv_counts[p];
    int total_send_req = send_disp[size];
    int total_recv_req = recv_disp[size];
    std::vector<int> send_req_buf(total_send_req);
    for (int p = 0; p < size; ++p) {
        std::copy(needed_from[p].begin(), needed_from[p].end(), send_req_buf.begin() + send_disp[p]);
    }
    std::vector<int> recv_req_buf(total_recv_req);
    MPI_Alltoallv(send_req_buf.data(), send_counts.data(), send_disp.data(), MPI_INT,
                  recv_req_buf.data(), recv_counts.data(), recv_disp.data(), MPI_INT,
                  MPI_COMM_WORLD);
    // Exchange values
    std::vector<int> send_val_counts = recv_counts;
    std::vector<int> recv_val_counts = send_counts;
    std::vector<int> send_val_disp = recv_disp;
    std::vector<int> recv_val_disp = send_disp;
    int total_send_val = total_recv_req;
    std::vector<double> send_val_buf(total_send_val);
    for (int idx = 0; idx < total_send_val; ++idx) {
        int j = recv_req_buf[idx];
        int local_idx = (j - rank) / size;
        if (local_idx < 0 || local_idx >= static_cast<int>(local_x.size())) {
            std::cerr << "Rank " << rank << ": bad local_idx for j=" << j << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        send_val_buf[idx] = local_x[local_idx];
    }
    int total_recv_val = total_send_req;
    std::vector<double> recv_val_buf(total_recv_val);
    MPI_Alltoallv(send_val_buf.data(), send_val_counts.data(), send_val_disp.data(), MPI_DOUBLE,
                  recv_val_buf.data(), recv_val_counts.data(), recv_val_disp.data(), MPI_DOUBLE,
                  MPI_COMM_WORLD);
    // Build ghost map
    ghost_x.clear();
    int offset = 0;
    for (int p = 0; p < size; ++p) {
        if (p == rank) continue;
        for (size_t i = 0; i < needed_from[p].size(); ++i) {
            int j = needed_from[p][i];
            ghost_x[j] = recv_val_buf[offset + i];
        }
        offset += needed_from[p].size();
    }
}