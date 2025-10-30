#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <fstream>

#define BLOCK_SIZE 64

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

    for (int i = 1; i < argc; ++i) {
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
    
    // reads .mtx file passed as argument
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

    // creates row_index, col_index, and values vectors
    std::vector<int> row_idx(nz);
    std::vector<int> col_idx(nz);
    std::vector<double> values(nz);

    for (int i = 0; i < nz; ++i) {
        int r, c;
        double val;
        if (fscanf(f, "%d %d %lf", &r, &c, &val) != 3) {
            std::cerr << "Error reading matrix entry." << std::endl;
            fclose(f);
            return 1;
        }
        row_idx[i] = r - 1; // convert 1-based to 0-based
        col_idx[i] = c - 1;
        values[i] = val;
    }
    fclose(f);

    // generate random monodimensional array
    std::vector<double> x(N), y(M, 0.0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    for (int i = 0; i < N; ++i) {
        x[i] = dis(gen);
    }

    // this will clear the file content
    std::ofstream ofs("../benchmarks/COO_exec_times.txt", std::ofstream::out | std::ofstream::trunc);
    ofs.close();
    
    // writes execution times to file
    std::ofstream outfile("../benchmarks/COO_exec_times.txt", std::ios_base::app);
    if (!outfile.is_open()) {
        std::cerr << "Warning: unable to open exec_time file for writing \n";
        exit(1);
    }
    
    // 3 warm-up runs
    if (verbose) {
        std::cout << "Running 3 warm-up iterations for COO SpMV... \n";
    }
    for (int warmup = 0; warmup < 3; ++warmup) {
        std::fill(y.begin(), y.end(), 0.0); // Reset y
        for (int block_start = 0; block_start < nz; block_start += BLOCK_SIZE) {
            int block_end = std::min(block_start + BLOCK_SIZE, nz);
            #pragma omp simd
            for (int k = block_start; k < block_end; ++k) {
                y[row_idx[k]] += values[k] * x[col_idx[k]];
            }
        }
    }
    
    // 10 runs of SpMV multiplication
    for (int i = 0; i < 10; ++i) {
        std::fill(y.begin(), y.end(), 0.0); // reset result vector
        auto start = std::chrono::high_resolution_clock::now();
    
        for (int block_start = 0; block_start < nz; block_start += BLOCK_SIZE) {
            int block_end = std::min(block_start + BLOCK_SIZE, nz);
            #pragma omp simd
            for (int k = block_start; k < block_end; ++k) {
                y[row_idx[k]] += values[k] * x[col_idx[k]];
            }
        }
    
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
    
        if (verbose) {
            std::cout << "Multiplication took " << elapsed.count() << " ms" << std::endl;
        }
        outfile << elapsed.count() << "\n";
    }
    outfile.close();
    
    return 0;
}
