#include <iostream>
#include <vector>
#include <chrono>
#include <cstdio>
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
    
    // reads mtx file passed as argument
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

    // create dense vector with all ones
    std::vector<double> x(N, 1.0), y(M, 0.0);

    // this will clear the file content
    std::ofstream ofs("..benchmarks/CSR_exec_times.txt", std::ofstream::out | std::ofstream::trunc);
    ofs.close();
    
    // 10 runs of SpMV multiplication
    for (int i = 0; i < 10 ; ++i) {
        // starting measurment
        auto start = std::chrono::steady_clock::now();
        // perform sequential SpMV multiplication
        for (int i = 0; i < M; ++i)
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
                y[i] += values[k] * x[col_idx[k]];
        // ending measurment
        auto end = std::chrono::steady_clock::now();
    
        // output result vector y
        for (int i = 0; i < M; ++i)
            std::cout << "y[" << i << "] = " << y[i] << std::endl;
    
        auto elapsed = std::chrono::duration<double, std::milli>(end - start);
        std::cout << "CSR multiplication took " << elapsed.count() << " ms" << std::endl;
    
        // writes execution times to file
        std::ofstream outfile("benchmarks/CSR_exec_times.txt", std::ios_base::app);
        if (outfile.is_open()) {
            outfile << elapsed.count() << "\n";
            outfile.close();
        } else {
            std::cerr << "Warning: unable to open exec_time file for writing \n";
        }
    }
    
    return 0;
}
