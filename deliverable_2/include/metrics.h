#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include <string>
#include <mpi.h>

struct SpMVStatistics {
    // Timing
    double best_time_s     = 1e9;
    double avg_time_s      = 0.0;
    double avg_comm_s      = 0.0;
    double comm_fraction   = 0.0;

    // Performance
    double gflops_best     = 0.0;
    double gflops_avg      = 0.0;
    double speedup         = 0.0;
    double efficiency      = 0.0;

    // Load balance
    int    rows_min = 0, rows_max = 0;
    long long rows_sum = 0;
    int    nnz_min = 0, nnz_max = 0;
    long long nnz_sum = 0;

    // Ghosts / communication
    int    ghosts_min = 0, ghosts_max = 0;
    long long ghosts_sum = 0;
    double comm_volume_mb = 0.0;   // total over all ranks

    // Memory
    double mem_min_mb = 0.0;
    double mem_max_mb = 0.0;

    // Matrix/problem size (for reference)
    int    M = 0;
    int    N = 0;
    long long nz_global = 0;
    int    nprocs = 0;
    std::string matrix_filename;
};

void collect_and_print_metrics(
    MPI_Comm comm,
    int rank,
    int size,
    const std::string& matrix_filename,
    int M, int N, long long nz_global,
    int local_M,
    int local_nnz,
    int local_ghosts,
    size_t mem_local_bytes,
    double best_time_s_local,
    double total_time_all_local,
    double total_comm_time_local,
    int benchmark_iters
);

void print_final_statistics(const SpMVStatistics& stats);

#endif // METRICS_H
