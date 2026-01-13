#ifndef COMMUNICATION_H
#define COMMUNICATION_H

#include <mpi.h>
#include <vector>
#include <set>
#include <unordered_map>
#include <iostream>
#include <algorithm>

void exchange_ghosts(int rank, int size, int N, int local_nnz,
                     const std::vector<int>& local_col_idx,
                     const std::vector<double>& local_x,
                     std::unordered_map<int, double>& ghost_x);

#endif