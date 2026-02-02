#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cassert>
#include <omp.h>

#include "../include/matrix_io.h"
#include "../include/matrix_gen.h"
#include "../include/distribution.h"
#include "../include/communication.h"
#include "../include/spmv_local.h"
#include "../include/metrics.h"

#define WARMUP_ITERS 3
#define BENCHMARK_ITERS 10

int main(int argc, char ** argv) {
    MPI_Init( & argc, & argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, & rank);
    MPI_Comm_size(MPI_COMM_WORLD, & size);

    // ===== Argument Parsing =====
    std::string matrix_filename;
    bool use_synthetic = false;
    int base_M = 0;
    double density = 0.0;
    bool verbose = false;
    int num_threads = 1;

    // First arg after program name is usually filename, but check for flags
    int arg_idx = 1;
    if (argc > 1 && argv[1][0] != '-') {
        matrix_filename = argv[arg_idx++];
    }

    while (arg_idx < argc) {
        std::string arg = argv[arg_idx++];
        if (arg == "--synthetic") {
            use_synthetic = true;
            if (arg_idx + 1 < argc) {
                base_M = std::atoi(argv[arg_idx++]);
                density = std::atof(argv[arg_idx++]);
                if (base_M <= 0 || density <= 0.0 || density > 1.0) {
                    if (rank == 0) std::cerr << "Invalid base_M or density\n";
                    MPI_Finalize();
                    return 1;
                }
            } else {
                if (rank == 0) std::cerr << "Usage: --synthetic base_M density\n";
                MPI_Finalize();
                return 1;
            }
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--threads" || arg == "-t") {
            if (arg_idx < argc) {
                num_threads = std::atoi(argv[arg_idx++]);
                if (num_threads <= 0) {
                    if (rank == 0) std::cerr << "Invalid threads\n";
                    MPI_Finalize();
                    return 1;
                }
            } else {
                if (rank == 0) std::cerr << "Usage: --threads T\n";
                MPI_Finalize();
                return 1;
            }
        } else if (matrix_filename.empty() && arg[0] != '-') {
            matrix_filename = arg; // Fallback if filename after flags
        } else {
            if (rank == 0) std::cerr << "Unknown arg: " << arg << "\n";
            MPI_Finalize();
            return 1;
        }
    }

    // Checks if the selection is not for synthetic matrix if the filename is provided
    if (!use_synthetic && matrix_filename.empty()) {
        if (rank == 0) std::cerr << "Usage: mpirun -np P ./spmv_mpi <matrix.mtx> [--synthetic base_M density] [--threads T] [--verbose]\n";
        MPI_Finalize();
        return 1;
    }

    // if synthetic matrix is used ignore matrix filename
    if (use_synthetic && !matrix_filename.empty()) {
        if (rank == 0) std::cerr << "Warning: Ignoring <matrix.mtx> since --synthetic is used\n";
        matrix_filename.clear(); 
    }

    omp_set_num_threads(num_threads);

    if (rank == 0 && verbose) {
        std::cout << "OMP threads per MPI rank: " << num_threads << std::endl;
        if (use_synthetic) {
            std::cout << "Using synthetic matrix: base_M=" << base_M << ", density=" << density << std::endl;
        }
    }

    // ===== GLOBAL MATRIX DATA =====
    int M = 0, N = 0, nz_global = 0;
    std::vector <int> global_row_ptr, global_col_idx;
    std::vector <double> global_values;

    if (use_synthetic) {
        if (rank == 0) {
            M = base_M * size; // Scale with P for weak scaling
            N = M; // Square matrix
            nz_global = generate_synthetic_matrix(M, density, 42,
                global_row_ptr, global_col_idx, global_values);
        }
    } else {
        if (rank == 0) {
            read_matrix_market(matrix_filename, M, N, nz_global,
                global_row_ptr, global_col_idx, global_values);
        }
    }

    // ===== BROADCAST DIMENSIONS =====
    int dims[3] = {
        M,
        N,
        nz_global
    };
    MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
    M = dims[0];
    N = dims[1];
    nz_global = dims[2];

    // ===== DISTRIBUTED MATRIX =====
    std::vector <int> local_row_ptr, local_col_idx;
    std::vector <double> local_values;
    int local_M = 0, local_nnz = 0;
    distribute_matrix(rank, size, M, nz_global,
        global_row_ptr, global_col_idx, global_values,
        local_row_ptr, local_col_idx, local_values,
        local_M, local_nnz);

    // Free global matrix memory on rank 0 (no longer needed)
    if (rank == 0) {
        global_row_ptr.clear();
        global_row_ptr.shrink_to_fit();
        global_col_idx.clear();
        global_col_idx.shrink_to_fit();
        global_values.clear();
        global_values.shrink_to_fit();
    }

    // ===== Local vector x (cyclic distribution) =====
    std::vector <double> local_x;
    int local_col_count = 0;
    init_local_vector(rank, size, N, local_x, local_col_count);

    GhostExchange ghost;
    build_ghost_structure(
        rank, size, N,
        local_col_idx,
        ghost
    );

    // ===== Precompute column access metadata (OPTIMIZATION #3) =====
    std::vector <char> col_is_local(local_col_idx.size());
    std::vector <int> col_access_idx(local_col_idx.size());

    for (size_t k = 0; k < local_col_idx.size(); ++k) {
        int j = local_col_idx[k];
        if (j % size == rank) {
            col_is_local[k] = 1;
            col_access_idx[k] = (j - rank) / size;
        } else {
            col_is_local[k] = 0;
            col_access_idx[k] = ghost.ghost_map.at(j); // SAFE: built once
        }
    }

    double best_time_s = 1e9;
    double total_time_all = 0.0;
    double total_comm_time = 0.0;

    // ===== Warm Up (not timed) =====
    if (rank == 0 && verbose) {
        std::cout << "Running " << WARMUP_ITERS << " warm-up iterations...\n";
    }
    for (int iter = 0; iter < WARMUP_ITERS; ++iter) {
        std::vector <double> ghost_values;
        exchange_ghost_values(rank, size, ghost, local_x, ghost_values);

        std::vector <double> y_local;
        compute_local_spmv(rank, size, local_M, local_row_ptr,
            local_col_idx, local_values,
            local_x, ghost_values,
            col_is_local, col_access_idx,
            y_local);
    }

    // ===== Benchmark (timed) =====
    if (rank == 0 && verbose) {
        std::cout << "Starting benchmark (" << BENCHMARK_ITERS << " iterations)...\n";
    }

    for (int iter = 0; iter < BENCHMARK_ITERS; ++iter) {
        auto start_total = std::chrono::steady_clock::now();

        // ===== Communication phase =====
        auto start_comm = std::chrono::steady_clock::now();
        std::vector <double> ghost_values;
        exchange_ghost_values(rank, size, ghost, local_x, ghost_values);
        auto end_comm = std::chrono::steady_clock::now();

        // ===== Computation phase ======
        std::vector <double> y_local;
        compute_local_spmv(rank, size, local_M, local_row_ptr,
            local_col_idx, local_values,
            local_x, ghost_values,
            col_is_local, col_access_idx,
            y_local);

        auto end_total = std::chrono::steady_clock::now();

        double t_total_local = std::chrono::duration <double> (end_total - start_total).count();
        double t_comm_local = std::chrono::duration <double> (end_comm - start_comm).count();

        // Reduce maximum time across ranks (bottleneck time)
        double max_time_total;
        MPI_Reduce( & t_total_local, & max_time_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        double max_comm_time;
        MPI_Reduce( & t_comm_local, & max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            total_time_all += max_time_total;
            total_comm_time += max_comm_time;
            if (max_time_total < best_time_s) best_time_s = max_time_total;
        }
    }

    size_t mem_local =
        local_row_ptr.size() * sizeof(int) +
        local_col_idx.size() * sizeof(int) +
        local_values.size() * sizeof(double) +
        local_x.size() * sizeof(double) +
        ghost.ghost_cols.size() * sizeof(int) +
        ghost.ghost_map.size() * (sizeof(int) + sizeof(char)); // approx

    collect_and_print_metrics(
        MPI_COMM_WORLD,
        rank, size,
        use_synthetic ? "synthetic" : matrix_filename, // Use "synthetic" as filename for metrics
        M, N, nz_global,
        local_M, local_nnz,
        static_cast <int> (ghost.ghost_cols.size()),
        mem_local,
        best_time_s,
        total_time_all,
        total_comm_time,
        BENCHMARK_ITERS
    );

    MPI_Finalize();
    return 0;
}