#include <iostream>
#include <vector>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <omp.h>
#include <algorithm>
#include <iomanip>
#include <fstream>

#define BLOCK_SIZE 10
#define NUM_THREADS 16
#define WARMUP_ITERS 3
#define BENCHMARK_ITERS 10

extern "C" {
#include "../include/mmio.h"
#include <valgrind/callgrind.h>
}

int main(int argc, char* argv[]) {
    bool verbose = false;
    std::string matrix_filename;
    // validate command-line arguments and print usage if incorrect
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " matrix_file.mtx" << std::endl;
        return 1;
    }
    
    // check if verbose
    for(int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--verbose" || std::string(argv[i]) == "-v") {
            verbose = true;
        } else {
            matrix_filename = argv[i];
        }
    }
    // check if no matrix file is passed
    if (matrix_filename.empty()) {
        std::cerr << "No matrix file specified." << std::endl;
        return 1;
    }
    
    // reads mtx file passed as argument
    FILE* f = fopen(matrix_filename.c_str(), "r");
    if (f == nullptr) {
        std::cerr << "Could not open file: " << matrix_filename << std::endl;
        return 1;
    }
    
    // reads matrix banner
    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) {
        std::cerr << "Could not process Matrix Market banner." << std::endl;
        fclose(f);
        return 1;
    }

    // check matrix type: must be real, sparse matrix
    if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode) || mm_is_complex(matcode)) {
        std::cerr << "Only real-valued sparse matrices supported." << std::endl;
        fclose(f);
        return 1;
    }

    // reads matrix size
    int M, N, nz;
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
        std::cerr << "Could not read matrix size." << std::endl;
        fclose(f);
        return 1;
    }

    // creates coo row_index, col_index, and values vector
    std::vector<int> row_coo(nz), col_coo(nz);
    std::vector<double> val_coo(nz);
    for (int i = 0; i < nz; ++i) {
        int r, c;
        double v;
        if (fscanf(f, "%d %d %lf", &r, &c, &v) != 3) {
            std::cerr << "Error reading matrix entry." << std::endl;
            fclose(f);
            return 1;
        }
        // Convert to 0-based indexing
        row_coo[i] = r - 1;
        col_coo[i] = c - 1;
        val_coo[i] = v;
    }
    fclose(f);

    // ================= COO -> CSR conversion =================
    std::vector<int> row_ptr(M + 1, 0);
    std::vector<int> col_idx(nz);
    std::vector<double> values(nz);

    // count nonzeros per row
    for (int i = 0; i < nz; ++i)
        row_ptr[row_coo[i] + 1]++;

    // prefix sum for row_ptr
    for (int i = 0; i < M; ++i)
        row_ptr[i + 1] += row_ptr[i];

    // insert values and columns in correct position
    std::vector<int> fill(M, 0);
    for (int i = 0; i < M; ++i) fill[i] = row_ptr[i];
    for (int i = 0; i < nz; ++i) {
        int r = row_coo[i];
        int dest = fill[r]++;
        col_idx[dest] = col_coo[i];
        values[dest] = val_coo[i];
    }

    // generate random monodimensional array
    std::vector<double> x(N), y(M, 0.0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    for (int i = 0; i < N; ++i) {
        x[i] = dis(gen);
    }
    //
    
    // ================= Warm-up (3 iterations, not timed) =================
    if (verbose) {
        std::cout << "Running 3 warm-up iterations for parallel CSR SpMV..." << std::endl;
    }
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        #pragma omp parallel
        {
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < M; ++i) y[i] = 0.0;
            
            #pragma omp for schedule(guided, BLOCK_SIZE) nowait
            for (int r = 0; r < M; ++r) {
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int k = row_ptr[r]; k < row_ptr[r+1]; ++k) {
                    sum += values[k] * x[col_idx[k]];
                }
                y[r] = sum;
            }
        }
    }

    // this will clear the file content
    std::ofstream ofs("../benchmarks/Parallel_CSR_exec_times.txt", std::ofstream::out | std::ofstream::trunc);
    ofs.close();
    
    
    // writes execution times to file
    std::ofstream outfile("../benchmarks/Parallel_CSR_exec_times.txt", std::ios_base::app);
    if (!outfile.is_open()) {
        std::cerr << "Warning: unable to open exec_time file for writing \n";
    }
    
    // ================= SET THREAD COUNT =================
    int num_threads = NUM_THREADS;
    if (getenv("OMP_NUM_THREADS") != nullptr) {
        num_threads = atoi(getenv("OMP_NUM_THREADS"));
    }
    omp_set_num_threads(num_threads);
    if (verbose) {
        std::cout << "Using: " << num_threads << " threads\n";
    }
    
    // toggles callgrind (set to false) collection here
    CALLGRIND_TOGGLE_COLLECT;
    
    //for (int j = 0; j < M; ++j) y[j] = 0.0;
    
    for (int i = 0; i < BENCHMARK_ITERS; ++i) {
        // ================= Parallel SpMV =================
        // starts timing
        auto start = std::chrono::steady_clock::now();
        
        #pragma omp parallel
        {
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < M; i++) {
                y[i] = 0.0;
            }
            
            #pragma omp for schedule(guided, BLOCK_SIZE) nowait
            for (int r = 0; r < M; ++r) {
                double sum = 0.0;
                
                #pragma omp simd reduction(+:sum)
                for (int k = row_ptr[r]; k < row_ptr[r+1]; ++k) {
                    sum += values[k] * x[col_idx[k]];
                }
                y[r] = sum;
            }
        }
        // =====================================
        // stops timing
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double, std::milli>(end - start);
        
        if (verbose) {
            std::cout << "Multiplication took " << elapsed.count() << " ms" << std::endl;
        }
        // writes to benchmark file
        outfile << elapsed.count() << "\n";
    }
    // toggles callgrind collect (set to true) here
    CALLGRIND_TOGGLE_COLLECT;

    outfile.close();
    
    if (verbose) {
        std::vector<double> times_ms(BENCHMARK_ITERS);
        std::ifstream times_file("../benchmarks/Parallel_CSR_exec_times.txt");
        for (int i = 0; i < BENCHMARK_ITERS; ++i) {
            times_file >> times_ms[i];
        }
        times_file.close();
        
        double best_time_ms = *std::min_element(times_ms.begin(), times_ms.end());
        double best_time_s  = best_time_ms / 1000.0;
        
        long long flops_per_spmv   = 2LL * nz;                                     // 1 mul + 1 add per nonzero
        double    bytes_per_spmv   = 12.0 * nz + 16.0 * M;                         // 8B val + 4B col_idx + ~16B for y (zero+write)
        
        double gflops = flops_per_spmv / best_time_s / 1e9;
        double gbs    = bytes_per_spmv / best_time_s / 1e9;
        double arith_intensity = static_cast<double>(flops_per_spmv) / bytes_per_spmv;
        
        std::cout << "\n=== Parallel CSR SpMV Benchmark Results ===\n";
        std::cout << "Matrix             : " << matrix_filename << "\n";
        std::cout << "Dimensions         : " << M << " x " << N << "   (nnz = " << nz << ")\n";
        std::cout << "Threads            : " << num_threads << "\n";
        std::cout << "Best time          : " << std::fixed << std::setprecision(3) 
                  << best_time_ms << " ms\n";
        std::cout << "Performance        : " << std::setprecision(2) 
                  << gflops << " GFLOPS\n";
        std::cout << "Effective Bandwidth: " << std::setprecision(2) 
                  << gbs << " GB/s\n";
        std::cout << "Arithmetic Intensity: " << std::setprecision(3) 
                  << arith_intensity << " FLOP/byte\n";
        std::cout << "==========================================\n";
    }

    return 0;
}