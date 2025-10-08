#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

extern "C" {
#include "../include/mmio.h"
}

int main(int argc, char* argv[]) {
    // validate command-line arguments and print usage if incorrect
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " matrix_file.mtx" << std::endl;
        return 1;
    }

    // reads .mtx file passed as argument
    FILE* f = fopen(argv[1], "r");
    if (f == nullptr) {
        std::cerr << "Could not open file: " << argv[1] << std::endl;
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

    // create dense vector with all ones
    std::vector<double> x(N, 1.0);
    std::vector<double> y(M, 0.0);

    // this will clear the file content
    std::ofstream ofs("../benchmarks/COO_exec_times.txt", std::ofstream::out | std::ofstream::trunc);
    ofs.close();
    
    // writes execution times to file
    std::ofstream outfile("../benchmarks/COO_exec_times.txt", std::ios_base::app);
    if (!outfile.is_open()) {
        std::cerr << "Warning: unable to open exec_time file for writing \n";
        exit(1);
    }
    
    // 10 runs of SpMV multiplication
    for (int i = 0; i < 10; ++i) {
        // starting measurment
        auto start = std::chrono::high_resolution_clock::now();
        // perform sequential SpMV multiplication
        for (int k = 0; k < nz; ++k) {
            y[row_idx[k]] += values[k] * x[col_idx[k]];
        }
        // ending measurment
        auto end = std::chrono::high_resolution_clock::now();
    
        // output result vector y
        for (int i = 0; i < M; ++i) {
            std::cout << "y[" << i << "] = " << y[i] << std::endl;
        }
        
        std::chrono::duration<double, std::milli> elapsed = end - start;
        
        std::cout << std::endl;
        std::cout << "==========================================================" << std::endl;
        std::cout << "Multiplication took " << elapsed.count() << "ms" << std::endl;
        std::cout << "==========================================================" << std::endl;
        //std::cout << std::chrono::high_resolution_clock::is_steady;
        outfile << elapsed.count() << "\n";
    }
    outfile.close();
    
    return 0;
}
