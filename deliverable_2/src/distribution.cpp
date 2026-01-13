#include "../include/distribution.h"

void distribute_matrix(int rank, int size, int M, int nz_global,
                       const std::vector<int>& global_row_ptr,
                       const std::vector<int>& global_col_idx,
                       const std::vector<double>& global_values,
                       std::vector<int>& local_row_ptr,
                       std::vector<int>& local_col_idx,
                       std::vector<double>& local_values,
                       int& local_M, int& local_nnz) {
    
    // Step 1: Calculate distribution metadata on rank 0
    std::vector<int> row_counts(size);
    std::vector<int> nnz_counts(size);
    std::vector<int> row_displs(size + 1, 0);
    std::vector<int> nnz_displs(size + 1, 0);
    
    if (rank == 0) {
        // Calculate how many rows and nnz each rank gets
        for (int r = 0; r < size; ++r) {
            row_counts[r] = 0;
            nnz_counts[r] = 0;
            for (int i = r; i < M; i += size) {
                row_counts[r]++;
                nnz_counts[r] += global_row_ptr[i + 1] - global_row_ptr[i];
            }
        }
        
        // Calculate displacements
        for (int r = 0; r < size; ++r) {
            row_displs[r + 1] = row_displs[r] + row_counts[r];
            nnz_displs[r + 1] = nnz_displs[r] + nnz_counts[r];
        }
    }
    
    // Step 2: Broadcast counts to all ranks
    MPI_Bcast(row_counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnz_counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Each rank knows its local size
    local_M = row_counts[rank];
    local_nnz = nnz_counts[rank];
    
    // Step 3: Prepare send buffers on rank 0
    std::vector<int> send_row_ptr;
    std::vector<int> send_col_idx;
    std::vector<double> send_values;
    
    if (rank == 0) {
        // Total elements to send
        int total_rows = row_displs[size];
        int total_nnz = nnz_displs[size];
        
        send_row_ptr.resize(total_rows + size); // +size for the extra element per rank
        send_col_idx.resize(total_nnz);
        send_values.resize(total_nnz);
        
        // Pack data for each rank
        for (int r = 0; r < size; ++r) {
            int row_offset = row_displs[r];
            int nnz_offset = nnz_displs[r];
            int local_row_idx = 0;
            int local_nnz_idx = 0;
            
            // For each row owned by rank r (cyclic distribution)
            for (int gi = r; gi < M; gi += size) {
                int start = global_row_ptr[gi];
                int end = global_row_ptr[gi + 1];
                int cnt = end - start;
                
                // Store row_ptr entry (relative offsets)
                send_row_ptr[row_offset + local_row_idx] = local_nnz_idx;
                
                // Copy column indices and values
                for (int k = start; k < end; ++k) {
                    send_col_idx[nnz_offset + local_nnz_idx] = global_col_idx[k];
                    send_values[nnz_offset + local_nnz_idx] = global_values[k];
                    local_nnz_idx++;
                }
                local_row_idx++;
            }
            // Store final row_ptr entry
            send_row_ptr[row_offset + local_row_idx] = local_nnz_idx;
        }
    }
    
    // Step 4: Allocate receive buffers
    local_row_ptr.resize(local_M + 1);
    local_col_idx.resize(local_nnz);
    local_values.resize(local_nnz);
    
    // Step 5: Scatter data using MPI_Scatterv
    // For row_ptr: each rank gets row_counts[rank] + 1 elements
    std::vector<int> row_ptr_counts(size);
    std::vector<int> row_ptr_displs(size + 1, 0);
    
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            row_ptr_counts[r] = row_counts[r] + 1;
            if (r > 0) {
                row_ptr_displs[r] = row_ptr_displs[r - 1] + row_counts[r - 1] + 1;
            }
        }
    }
    
    MPI_Bcast(row_ptr_counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_ptr_displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnz_displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Scatter row_ptr
    MPI_Scatterv(send_row_ptr.data(), row_ptr_counts.data(), row_ptr_displs.data(), MPI_INT,
                 local_row_ptr.data(), local_M + 1, MPI_INT,
                 0, MPI_COMM_WORLD);
    
    // Scatter col_idx
    MPI_Scatterv(send_col_idx.data(), nnz_counts.data(), nnz_displs.data(), MPI_INT,
                 local_col_idx.data(), local_nnz, MPI_INT,
                 0, MPI_COMM_WORLD);
    
    // Scatter values
    MPI_Scatterv(send_values.data(), nnz_counts.data(), nnz_displs.data(), MPI_DOUBLE,
                 local_values.data(), local_nnz, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
}

void init_local_vector(int rank, int size, int N,
                       std::vector<double>& local_x,
                       int& local_col_count) {
    local_col_count = (N + size - 1 - rank) / size;  // Cyclic distribution
    local_x.assign(local_col_count, 1.0);  // Unit vector for testing
}