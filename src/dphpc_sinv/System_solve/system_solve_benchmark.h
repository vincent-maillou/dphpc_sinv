

double solve_mkl_dgesv(
    double *matrix_dense,
    double *rhs,
    double *reference_solution,
    int matrice_size,
    double tolerance,
    bool flag_verbose);

double solve_cusolver_LU(
    double *matrix_dense_h,
    double *rhs_h,
    double *reference_solution_h,
    int matrice_size,
    double tolerance,
    bool flag_verbose);

double solve_cusparse_CG(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution_h,
    int nnz,
    int matrice_size,
    double tolerance,
    bool flag_verbose);

