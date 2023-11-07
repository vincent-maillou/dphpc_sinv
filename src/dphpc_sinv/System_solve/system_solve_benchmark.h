

double solve_mkl_dgbsv(
    double *matrix_band,
    double *rhs,
    double *reference_solution,
    int matrix_size,
    int kl,
    int ku,
    double tolerance,
    bool flag_verbose);

double solve_mkl_dgesv(
    double *matrix_dense,
    double *rhs,
    double *reference_solution,
    int matrix_size,
    double tolerance,
    bool flag_verbose);

double solve_cusolver_LU(
    double *matrix_dense_h,
    double *rhs_h,
    double *reference_solution_h,
    int matrix_size,
    double tolerance,
    bool flag_verbose);

double solve_cusparse_CG(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution_h,
    int nnz,
    int matrix_size,
    double tolerance,
    bool flag_verbose);

double solve_cusparse_ILU_CG(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution_h,
    int nnz,
    int matrix_size,
    double tolerance,
    bool flag_verbose);

double solve_cusolver_CHOL(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution_h,
    int nnz,
    int matrix_size,
    double tolerance,
    bool flag_verbose);
