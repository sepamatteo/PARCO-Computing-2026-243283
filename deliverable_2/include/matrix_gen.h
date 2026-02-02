#ifndef MATRIX_GEN_H
#define MATRIX_GEN_H

#include <vector>
#include <string>

/**
 * @brief Generates a synthetic sparse matrix in CSR format
 *
 * Generates a random sparse matrix using Erdos-Renyi model.
 * Matrix is square (M == N) for simplicity.
 * Nonzeros are placed uniformly at random with given density.
 * Values are random doubles in [-1.0, 1.0].
 * Ensures no duplicate entries in rows (uses sorting and unique).
 *
 * @param M             Number of rows (and columns)
 * @param density       Sparsity density (fraction of nonzeros, e.g., 0.01)
 * @param seed          Random seed for reproducibility
 * @param row_ptr       [out] CSR row pointers (size = M+1)
 * @param col_idx       [out] CSR column indices (size = nz_global)
 * @param values        [out] CSR nonzero values (size = nz_global)
 * @return              Actual number of nonzeros generated
 */
int generate_synthetic_matrix(int M, double density, int seed,
                              std::vector<int>& row_ptr,
                              std::vector<int>& col_idx,
                              std::vector<double>& values);

#endif
