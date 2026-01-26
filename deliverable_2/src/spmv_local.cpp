#include "../include/spmv_local.h"
#include "../include/communication.h"

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <unordered_map>

void compute_local_spmv(int rank, int size, int local_M,
                        const std::vector<int>& local_row_ptr,
                        const std::vector<int>& local_col_idx,
                        const std::vector<double>& local_values,
                        const std::vector<double>& local_x,
                        const std::vector<double>& ghost_values,
                        const GhostExchange& ghost,
                        std::vector<double>& y_local)
{
    y_local.assign(local_M, 0.0);

    // Build ghost column-to-index map (O(num_ghosts), done once per call)
    /*std::unordered_map<int, int> ghost_map;
    for (size_t t = 0; t < ghost.ghost_cols.size(); ++t) {
        ghost_map[ghost.ghost_cols[t]] = static_cast<int>(t);
        }*/

    // MPI+X: OpenMP parallel over local rows
    #pragma omp parallel for schedule(guided, 64)
    for (int i = 0; i < local_M; ++i) {
        double sum = 0.0;
        int start = local_row_ptr[i];
        int end   = local_row_ptr[i + 1];

        for (int k = start; k < end; ++k) {
            int j = local_col_idx[k];
            double v = local_values[k];

            int owner = j % size;
            double xval;
            if (owner == rank) {
                // Local cyclic x entry
                int lidx = (j - rank) / size;
                xval = local_x[lidx];
            } else {
                // Ghost x entry: O(1) hash lookup
                //auto it = ghost_map.find(j);
                auto it = ghost.ghost_map.find(j);
                if (it == ghost.ghost_map.end()) {
                    std::cerr << "Rank " << rank
                              << ": missing ghost value for column " << j << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                int gidx = it->second;
                xval = ghost_values[gidx];
            }
            sum += v * xval;
        }

        y_local[i] = sum;
    }
}
