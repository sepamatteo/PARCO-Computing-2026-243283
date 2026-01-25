#ifndef SPMV_LOCAL_H
#define SPMV_LOCAL_H

#include <vector>
#include <iostream>
#include <mpi.h>

#include "../include/communication.h"

void compute_local_spmv(int rank, int size, int local_M,
                        const std::vector<int>& local_row_ptr,
                        const std::vector<int>& local_col_idx,
                        const std::vector<double>& local_values,
                        const std::vector<double>& local_x,
                        const std::vector<double>& ghost_values,
                        const GhostExchange& ghost,
                        std::vector<double>& y_local);


#endif
