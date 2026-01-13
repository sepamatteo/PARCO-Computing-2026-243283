#include "../include/distribution.h"

void distribute_matrix(int rank, int size, int M, int nz_global,
                       const std::vector<int>& global_row_ptr,
                       const std::vector<int>& global_col_idx,
                       const std::vector<double>& global_values,
                       std::vector<int>& local_row_ptr,
                       std::vector<int>& local_col_idx,
                       std::vector<double>& local_values,
                       int& local_M, int& local_nnz) {
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            // Count local rows and nnz
            local_M = 0;
            local_nnz = 0;
            for (int i = r; i < M; i += size) {
                local_M++;
                local_nnz += global_row_ptr[i + 1] - global_row_ptr[i];
            }
            local_row_ptr.assign(local_M + 1, 0);
            local_col_idx.resize(local_nnz);
            local_values.resize(local_nnz);
            int local_row_idx = 0;
            int nnz_idx = 0;
            for (int gi = r; gi < M; gi += size) {
                int start = global_row_ptr[gi];
                int end = global_row_ptr[gi + 1];
                int cnt = end - start;
                local_row_ptr[local_row_idx + 1] = local_row_ptr[local_row_idx] + cnt;
                for (int k = start; k < end; ++k) {
                    if (nnz_idx >= local_nnz) {
                        std::cerr << "Rank 0 → rank " << r << ": nnz overflow!\n";
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    local_col_idx[nnz_idx] = global_col_idx[k];
                    local_values[nnz_idx] = global_values[k];
                    nnz_idx++;
                }
                local_row_idx++;
            }
            assert(nnz_idx == local_nnz && "nnz count mismatch!");
            // Send to other ranks
            int meta[2] = {local_M, local_nnz};
            MPI_Send(meta, 2, MPI_INT, r, 0, MPI_COMM_WORLD);
            if (r != 0) {
                MPI_Send(local_row_ptr.data(), local_M + 1, MPI_INT, r, 1, MPI_COMM_WORLD);
                MPI_Send(local_col_idx.data(), local_nnz, MPI_INT, r, 2, MPI_COMM_WORLD);
                MPI_Send(local_values.data(), local_nnz, MPI_DOUBLE, r, 3, MPI_COMM_WORLD);
            }
        }
    } else {
        int meta[2];
        MPI_Recv(meta, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_M = meta[0];
        local_nnz = meta[1];
        local_row_ptr.resize(local_M + 1);
        local_col_idx.resize(local_nnz);
        local_values.resize(local_nnz);
        MPI_Recv(local_row_ptr.data(), local_M + 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_col_idx.data(), local_nnz, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_values.data(), local_nnz, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void init_local_vector(int rank, int size, int N,
                       std::vector<double>& local_x,
                       int& local_col_count) {
    local_col_count = (N + size - 1 - rank) / size;  // Cyclic distribution
    local_x.assign(local_col_count, 1.0);  // Unit vector for testing
}