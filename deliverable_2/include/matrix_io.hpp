#ifndef MATRIX_IO_HPP
#define MATRIX_IO_HPP

#include <vector>
#include <string>
#include <cstdio>
#include <iostream>
#include <cassert>
#include <mpi.h>

extern "C" {
#include "../include/mmio.h"
}

/**
 * @brief Reads a Matrix Market file (.mtx) and converts it to CSR format
 *
 * Only called by rank 0. The file is expected to contain a sparse real
 * general matrix in coordinate format (i j value).
 *
 * Input indices are 1-based → converted to 0-based in output arrays.
 *
 * @param filename      Path to the .mtx file
 * @param M             [out] number of rows
 * @param N             [out] number of columns
 * @param nz_global     [out] number of non-zero entries
 * @param row_ptr       [out] CSR row pointers (size = M+1)
 * @param col_idx       [out] CSR column indices (0-based, size = nz_global)
 * @param values        [out] CSR nonzero values (size = nz_global)
 * */
void read_matrix_market(const std::string& filename, int& M, int& N, int& nz_global,
                        std::vector<int>& row_ptr, std::vector<int>& col_idx,
                        std::vector<double>& values);

#endif
