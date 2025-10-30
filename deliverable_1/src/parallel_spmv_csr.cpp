#include <iostream>
#include <vector>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <omp.h>

#define BLOCK_SIZE 64
#define NUM_THREADS 16

extern "C" {
#include "../include/mmio.h"
}

int main(int argc, char* argv[]) {
    bool verbose = false;
    std::string matrix_filename;
    // validate command-line arguments and print usage if incorrect
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " matrix_file.mtx" << std::endl;
        return 1;
    }
    
    for(int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--verbose" || std::string(argv[i]) == "-v") {
            verbose = true;
        } else {
            matrix_filename = argv[i];
        }
    }
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

    // COO to CSR conversion
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
    
    // Warm-up (3 iterations, not timed)
    if (verbose) {
        std::cout << "Running 3 warm-up iterations for parallel CSR SpMV..." << std::endl;
    }
    for (int i = 0; i < 3; ++i) {
        #pragma omp parallel for
        for (int j = 0; j < M; ++j) y[j] = 0.0;
    
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, BLOCK_SIZE) nowait
            for (int r = 0; r < M; ++r) {
                double sum = 0.0;
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
    
    // SET THREAD COUNT
    int num_threads = NUM_THREADS;
    if (getenv("OMP_NUM_THREADS") != nullptr) {
        num_threads = atoi(getenv("OMP_NUM_THREADS"));
    }
    omp_set_num_threads(num_threads);
    if (verbose) {
        std::cout << "Using: " << num_threads << " threads\n";
    }
    
    for (int i = 0; i < 10; ++i) {
        // ====== Parallel SpMV =================
        #pragma omp parallel for
        for (int j = 0; j < M; ++j) y[j] = 0.0;
        
        auto start = std::chrono::steady_clock::now();
        
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, BLOCK_SIZE) nowait
            for (int r = 0; r < M; ++r) {
                double sum = 0.0;
                for (int k = row_ptr[r]; k < row_ptr[r+1]; ++k) {
                    sum += values[k] * x[col_idx[k]];
                }
                y[r] = sum;
            }
        }
        // =====================================
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double, std::milli>(end - start);
        
        if (verbose) {
            std::cout << "Multiplication took " << elapsed.count() << " ms" << std::endl;
        }
        
        outfile << elapsed.count() << "\n";
    }

    outfile.close();

    return 0;
}