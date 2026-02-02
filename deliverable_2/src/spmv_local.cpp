#include "../include/spmv_local.hpp"

#include <mpi.h>
#include <omp.h>
#include <vector>

void compute_local_spmv(int /*rank*/, int /*size*/, int local_M,
                        const std::vector<int>& local_row_ptr,
                        const std::vector<int>& local_col_idx,
                        const std::vector<double>& local_values,
                        const std::vector<double>& local_x,
                        const std::vector<double>& ghost_values,
                        const std::vector<char>& col_is_local,
                        const std::vector<int>& col_access_idx,
                        std::vector<double>& y_local)
{
    y_local.assign(local_M, 0.0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < local_M; ++i) {
        double sum = 0.0;

        for (int k = local_row_ptr[i]; k < local_row_ptr[i + 1]; ++k) {
            double xval = col_is_local[k]
                ? local_x[col_access_idx[k]]
                : ghost_values[col_access_idx[k]];

            sum += local_values[k] * xval;
        }

        y_local[i] = sum;
    }
}
