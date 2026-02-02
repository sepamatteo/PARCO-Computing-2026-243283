#include "../include/matrix_gen.hpp"

#include <random>
#include <algorithm>
#include <vector>
#include <utility>

int generate_synthetic_matrix(int M, double density, int seed,
                              std::vector<int>& row_ptr,
                              std::vector<int>& col_idx,
                              std::vector<double>& values) {
    int N = M;  // Square matrix
    std::mt19937 gen(seed);  // Mersenne Twister for good randomness
    std::uniform_int_distribution<int> dist_col(0, N - 1);
    std::uniform_real_distribution<double> dist_val(-1.0, 1.0);

    row_ptr.assign(M + 1, 0);
    std::vector<std::vector<std::pair<int, double>>> rows(M);  // Per-row list to handle duplicates

    // Expected nnz, but we generate exactly this many attempts (may have fewer after dedup)
    long long int expected_nnz = static_cast<int>(density * static_cast<double>(M) * static_cast<double>(N));
    for (int i = 0; i < expected_nnz; ++i) {
        int r = gen() % M;  // Uniform row selection
        int c = dist_col(gen);
        double v = dist_val(gen);
        rows[r].emplace_back(c, v);
    }

    // Process each row: sort by column, remove duplicates
    int nz_global = 0;
    for (int i = 0; i < M; ++i) {
        auto& row = rows[i];
        std::sort(row.begin(), row.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        auto last = std::unique(row.begin(), row.end(), [](const auto& a, const auto& b) {
            return a.first == b.first;
        });
        row.erase(last, row.end());
        row_ptr[i + 1] = row_ptr[i] + static_cast<int>(row.size());
        nz_global += static_cast<int>(row.size());
    }

    // Flatten into CSR arrays
    col_idx.resize(nz_global);
    values.resize(nz_global);
    int offset = 0;
    for (int i = 0; i < M; ++i) {
        for (const auto& p : rows[i]) {
            col_idx[offset] = p.first;
            values[offset] = p.second;
            ++offset;
        }
    }

    return nz_global;
}