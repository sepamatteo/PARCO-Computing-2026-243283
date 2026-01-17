#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cassert>
#include <unordered_map>
#include <iomanip>
#include <omp.h>

#include "../include/matrix_io.h"
#include "../include/distribution.h"
#include "../include/communication.h"
#include "../include/spmv_local.h"

#define WARMUP_ITERS 3
#define BENCHMARK_ITERS 10

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) std::cerr << "Usage: mpirun -np P ./spmv_mpi <matrix.mtx> [--threads T] [--verbose ; -v]\n";
        MPI_Finalize();
        return 1;
    }

    std::string matrix_filename = argv[1];
    bool verbose = false;
    int num_threads = 1;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg  == "--threads" || arg == "-t") {
            if (i + 1 < argc) {
                num_threads = std::atoi(argv[++i]);
            }
        }
    }

    omp_set_num_threads(num_threads);

    if (rank == 0 && verbose) {
        std::cout << "OMP threads per MPI rank: " << num_threads << std::endl;
    }

    // ===== GLOBAL MATRIX DATA =====
    int M = 0, N = 0, nz_global = 0;
    std::vector<int> global_row_ptr, global_col_idx;
    std::vector<double> global_values;

    if (rank == 0) {
        read_matrix_market(matrix_filename, M, N, nz_global,
                           global_row_ptr, global_col_idx, global_values);
    }

    // ===== BROADCAST DIMENSIONS =====
    int dims[3] = {M, N, nz_global};
    MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
    M = dims[0]; N = dims[1]; nz_global = dims[2];

    // ===== DISTRIBUTED MATRIX =====
    std::vector<int> local_row_ptr, local_col_idx;
    std::vector<double> local_values;
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
    std::vector<double> local_x;
    int local_col_count = 0;
    init_local_vector(rank, size, N, local_x, local_col_count);

    double best_time_s     = 1e9;
    double total_time_all  = 0.0;
    double total_comm_time = 0.0;

    // ===== Warm Up (not timed) =====
    if (rank == 0 && verbose) {
        std::cout << "Running " << WARMUP_ITERS << " warm-up iterations...\n";
    }
    for (int iter = 0; iter < WARMUP_ITERS; ++iter) {
        std::unordered_map<int, double> ghost_x;
        exchange_ghosts(rank, size, N, local_nnz, local_col_idx, local_x, ghost_x);
        std::vector<double> y_local;
        compute_local_spmv(rank, size, local_M, local_row_ptr, local_col_idx, local_values,
                           local_x, ghost_x, y_local);
    }

    // ===== Benchmark (timed) =====
    if (rank == 0 && verbose) {
        std::cout << "Starting benchmark (" << BENCHMARK_ITERS << " iterations)...\n";
    }

    for (int iter = 0; iter < BENCHMARK_ITERS; ++iter) {
        auto start_total = std::chrono::steady_clock::now();

        // ===== Communication phase =====
        auto start_comm = std::chrono::steady_clock::now();
        std::unordered_map<int, double> ghost_x;
        exchange_ghosts(rank, size, N, local_nnz, local_col_idx, local_x, ghost_x);
        auto end_comm = std::chrono::steady_clock::now();

        // ===== Computation phase ======
        std::vector<double> y_local;
        compute_local_spmv(rank, size, local_M, local_row_ptr, local_col_idx, local_values,
                           local_x, ghost_x, y_local);

        auto end_total = std::chrono::steady_clock::now();

        double t_total_local = std::chrono::duration<double>(end_total - start_total).count();
        double t_comm_local  = std::chrono::duration<double>(end_comm - start_comm).count();

        // Reduce maximum time across ranks (bottleneck time)
        double max_time_total;
        MPI_Reduce(&t_total_local, &max_time_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        double max_comm_time;
        MPI_Reduce(&t_comm_local, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            total_time_all  += max_time_total;
            total_comm_time += max_comm_time;
            if (max_time_total < best_time_s) best_time_s = max_time_total;
        }
    }

    // ===== Final statistics (rank 0 only) =====
    if (rank == 0) {
        double avg_time_s = total_time_all / BENCHMARK_ITERS;
        double avg_comm_s = total_comm_time / BENCHMARK_ITERS;

        // ===== FLOPs: standard 2 per nonzero =====
        long long flops_per_spmv = 2LL * nz_global;

        double gflops_best = (flops_per_spmv / best_time_s) / 1e9;
        double gflops_avg  = (flops_per_spmv / avg_time_s) / 1e9;

        // ===== Rough bandwidth estimate =====
        double bytes_per_spmv_approx = 12.0 * nz_global + 16.0 * M;  // 8B val + 4B col + ~16B y access
        double gbs_avg = (bytes_per_spmv_approx / avg_time_s) / 1e9;

        double arith_intensity = static_cast<double>(flops_per_spmv) / bytes_per_spmv_approx;

        double comm_fraction = (avg_comm_s / avg_time_s) * 100.0;

        std::cout << "\n=== Distributed MPI CSR SpMV Benchmark Results ===\n";
        std::cout << "Matrix              : " << matrix_filename << "\n";
        std::cout << "Dimensions          : " << M << " x " << N << "   (nnz = " << nz_global << ")\n";
        std::cout << "Processes           : " << size << "\n";
        std::cout << "Best max-time       : " << std::fixed << std::setprecision(3)
                  << best_time_s * 1000 << " ms\n";
        std::cout << "Avg max-time        : " << std::setprecision(3)
                  << avg_time_s * 1000 << " ms\n";
        std::cout << "Performance (best)  : " << std::setprecision(2) << gflops_best << " GFLOPS (system)\n";
        std::cout << "Performance (avg)   : " << std::setprecision(2) << gflops_avg  << " GFLOPS (system)\n";
        std::cout << "Effective Bandwidth : " << std::setprecision(2) << gbs_avg << " GB/s (approx)\n";
        std::cout << "Arithmetic Intensity: " << std::setprecision(3) << arith_intensity << " FLOP/byte\n";
        std::cout << "Communication frac. : " << std::setprecision(1) << comm_fraction << "%\n";
        std::cout << "==============================================\n";
    }

    MPI_Finalize();
    return 0;
}
