#ifndef MATRIX_IO_H
#define MATRIX_IO_H

#include <vector>
#include <string>
#include <cstdio>
#include <iostream>
#include <cassert>
#include <mpi.h>

extern "C" {
#include "../include/mmio.h"
}

void read_matrix_market(const std::string& filename, int& M, int& N, int& nz_global,
                        std::vector<int>& row_ptr, std::vector<int>& col_idx,
                        std::vector<double>& values);

#endif