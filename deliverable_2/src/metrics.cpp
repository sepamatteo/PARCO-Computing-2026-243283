#include "../include/metrics.h"
#include <iostream>
#include <iomanip>
#include <cmath>

void collect_and_print_metrics(
    MPI_Comm comm,
    int rank,
    int size,
    const std::string& matrix_filename,
    int M_in, int N_in, long long nz_global_in,
    int local_M,
    int local_nnz,
    int local_ghosts,
    size_t mem_local_bytes,
    double best_time_s_local,
    double total_time_all_local,
    double total_comm_time_local,
    int benchmark_iters
) {
    int my_rank;
    MPI_Comm_rank(comm, &my_rank);

    SpMVStatistics stats;
    stats.matrix_filename = matrix_filename;
    stats.M          = M_in;
    stats.N          = N_in;
    stats.nz_global  = nz_global_in;
    stats.nprocs     = size;

    // --- Timing ---
    double max_time_total;
    MPI_Reduce(&best_time_s_local, &stats.best_time_s, 1, MPI_DOUBLE, MPI_MIN, 0, comm);

    double max_time;
    MPI_Reduce(&total_time_all_local, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    stats.avg_time_s = max_time / benchmark_iters;

    double max_comm;
    MPI_Reduce(&total_comm_time_local, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    stats.avg_comm_s = max_comm / benchmark_iters;

    stats.comm_fraction = (stats.avg_comm_s / stats.avg_time_s) * 100.0;

    // --- Performance ---
    long long flops_per_spmv = 2LL * stats.nz_global;
    stats.gflops_best = (flops_per_spmv / stats.best_time_s) / 1e9;
    stats.gflops_avg  = (flops_per_spmv / stats.avg_time_s)  / 1e9;

    // Speedup & efficiency (you'll need to provide T1 separately or hardcode it)
    // For now we leave it as placeholder (common practice: replace manually)
    stats.speedup    = 1.0;     // ← replace with real T1 / Tp
    stats.efficiency = stats.speedup / size;

    // --- Load balance ---
    MPI_Reduce(&local_M,     &stats.rows_min, 1, MPI_INT, MPI_MIN, 0, comm);
    MPI_Reduce(&local_M,     &stats.rows_max, 1, MPI_INT, MPI_MAX, 0, comm);
    MPI_Reduce(&local_M,     &stats.rows_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);

    MPI_Reduce(&local_nnz,   &stats.nnz_min,  1, MPI_INT, MPI_MIN, 0, comm);
    MPI_Reduce(&local_nnz,   &stats.nnz_max,  1, MPI_INT, MPI_MAX, 0, comm);
    MPI_Reduce(&local_nnz,   &stats.nnz_sum,  1, MPI_LONG_LONG, MPI_SUM, 0, comm);

    MPI_Reduce(&local_ghosts,&stats.ghosts_min,1, MPI_INT, MPI_MIN, 0, comm);
    MPI_Reduce(&local_ghosts,&stats.ghosts_max,1, MPI_INT, MPI_MAX, 0, comm);
    MPI_Reduce(&local_ghosts,&stats.ghosts_sum,1, MPI_LONG_LONG, MPI_SUM, 0, comm);

    // --- Communication volume ---
    double comm_bytes_local = 2.0 * local_ghosts * sizeof(double);
    double comm_bytes_total;
    MPI_Reduce(&comm_bytes_local, &comm_bytes_total, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    stats.comm_volume_mb = comm_bytes_total / 1e6;

    // --- Memory ---
    size_t mem_min, mem_max;
    MPI_Reduce(&mem_local_bytes, &mem_min, 1, MPI_UNSIGNED_LONG, MPI_MIN, 0, comm);
    MPI_Reduce(&mem_local_bytes, &mem_max, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, comm);
    stats.mem_min_mb = mem_min / (1024.0 * 1024.0);
    stats.mem_max_mb = mem_max / (1024.0 * 1024.0);

    // Only rank 0 prints
    if (my_rank == 0) {
        print_final_statistics(stats);
    }
}

void print_final_statistics(const SpMVStatistics& s) {
    std::cout << "\n=== Distributed MPI CSR SpMV Benchmark Results ===\n";
    std::cout << "Matrix              : " << s.matrix_filename << "\n";
    std::cout << "Dimensions          : " << s.M << " x " << s.N
              << "   (nnz = " << s.nz_global << ")\n";
    std::cout << "Processes           : " << s.nprocs << "\n\n";

    std::cout << "Timing per SpMV\n";
    std::cout << "  Best max-time     : " << s.best_time_s * 1000 << " ms\n";
    std::cout << "  Avg  max-time     : " << s.avg_time_s  * 1000 << " ms\n\n";

    std::cout << "Performance\n";
    std::cout << "  GFLOPS (best)     : " << s.gflops_best << "\n";
    std::cout << "  GFLOPS (avg)      : " << s.gflops_avg  << "\n";
    std::cout << "  Speedup           : " << s.speedup    << "×\n";
    std::cout << "  Efficiency        : " << s.efficiency * 100 << " %\n\n";

    std::cout << "Load balance\n";
    std::cout << "  Rows per rank     : min=" << s.rows_min
              << "  avg=" << (s.rows_sum / s.nprocs)
              << "  max=" << s.rows_max << "\n";
    std::cout << "  NNZ per rank      : min=" << s.nnz_min
              << "  avg=" << (s.nnz_sum / s.nprocs)
              << "  max=" << s.nnz_max << "\n\n";

    std::cout << "Communication\n";
    std::cout << "  Ghost entries     : min=" << s.ghosts_min
              << "  avg=" << (s.ghosts_sum / s.nprocs)
              << "  max=" << s.ghosts_max << "\n";
    std::cout << "  Comm volume       : " << s.comm_volume_mb << " MB per SpMV\n";
    std::cout << "  Comm fraction     : " << s.comm_fraction << " %\n\n";

    std::cout << "Memory footprint\n";
    std::cout << "  Per-rank memory   : min=" << s.mem_min_mb
              << " MB   max=" << s.mem_max_mb << " MB\n";

    std::cout << "==============================================\n";
}