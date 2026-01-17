#include "../include/spmv_local.h"

#include <omp.h>

void compute_local_spmv(int rank, int size, int local_M,
                        const std::vector<int>& local_row_ptr,
                        const std::vector<int>& local_col_idx,
                        const std::vector<double>& local_values,
                        const std::vector<double>& local_x,
                        const std::unordered_map<int, double>& ghost_x,
                        std::vector<double>& y_local) {
    y_local.assign(local_M, 0.0);
    #pragma omp parallel for schedule(guided, 64)
    for (int i = 0; i < local_M; ++i) {
        double sum = 0.0;
        int start = local_row_ptr[i];
        int end = local_row_ptr[i + 1];

        for (int k = start; k < end; ++k) {
            int j = local_col_idx[k];
            double v = local_values[k];
            int owner = j % size;

            if (owner == rank) {
                int lidx = (j - rank) / size;
                sum += v * local_x[lidx];
            } else {
                auto it = ghost_x.find(j);
                if (it == ghost_x.end()) {
                    std::cerr << "Rank " << rank << ": missing ghost for col " << j << "\n";
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                sum += v * it->second;
            }
        }
        y_local[i] = sum;
    }
}
