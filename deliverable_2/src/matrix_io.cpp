#include "../include/matrix_io.h"

void read_matrix_market(const std::string& filename, int& M, int& N, int& nz_global,
                        std::vector<int>& row_ptr, std::vector<int>& col_idx,
                        std::vector<double>& values) {
    FILE* f = fopen(filename.c_str(), "r");
    if (!f) {
        std::cerr << "Rank 0: Cannot open " << filename << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0 ||
        !mm_is_matrix(matcode) || !mm_is_sparse(matcode) || mm_is_complex(matcode)) {
        std::cerr << "Rank 0: Invalid Matrix Market type\n";
        fclose(f);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (mm_read_mtx_crd_size(f, &M, &N, &nz_global) != 0) {
        std::cerr << "Rank 0: Cannot read size\n";
        fclose(f);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Read COO (1-based to 0-based)
    std::vector<int> row_coo(nz_global), col_coo(nz_global);
    std::vector<double> val_coo(nz_global);
    for (int i = 0; i < nz_global; ++i) {
        int r, c;
        double v;
        if (fscanf(f, "%d %d %lf", &r, &c, &v) != 3) {
            std::cerr << "Rank 0: Read error at entry " << i << "\n";
            fclose(f);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        row_coo[i] = r - 1;
        col_coo[i] = c - 1;
        val_coo[i] = v;
    }
    fclose(f);
    // COO -> CSR
    row_ptr.assign(M + 1, 0);
    for (int i = 0; i < nz_global; ++i) {
        row_ptr[row_coo[i] + 1]++;
    }
    for (int i = 0; i < M; ++i) {
        row_ptr[i + 1] += row_ptr[i];
    }
    col_idx.resize(nz_global);
    values.resize(nz_global);
    std::vector<int> fill(M, 0);
    for (int i = 0; i < M; ++i) fill[i] = row_ptr[i];
    for (int i = 0; i < nz_global; ++i) {
        int r = row_coo[i];
        int dest = fill[r]++;
        col_idx[dest] = col_coo[i];
        values[dest] = val_coo[i];
    }
    // Check fill
    for (int i = 0; i < M; ++i) {
        assert(fill[i] == row_ptr[i + 1] && "CSR fill mismatch!");
    }
}