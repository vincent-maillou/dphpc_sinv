#include <string> 
#include <omp.h>

#include "utils.h"

#include "mkl.h"
#include "cusolverDn.h"
#include <cusparse.h>



// cusolver has CUSOLVER_STATUS_SUCCESS and not cudaSuccess, but they are the same
// this seems again kinda hacky
#define cudaErrchk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDAassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define cusolverErrchk(ans) { cusolverAssert((ans), __FILE__, __LINE__); }
inline void cusolverAssert(cusolverStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUSOLVER_STATUS_SUCCESS) 
   {
        //Did not find a counter part to cudaGetErrorString in cusolver
        fprintf(stderr,"CUSOLVERassert: %s %d\n", file, line);
        if (abort) exit(code);
   }
}


#define cublasErrchk(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUBLAS_STATUS_SUCCESS) 
   {
        //Did not find a counter part to cudaGetErrorString in cusolver
        fprintf(stderr,"CUBLASassert: %s %d\n", file, line);
        if (abort) exit(code);
   }
}

#define cusparseErrchk(ans) { cusparseAssert((ans), __FILE__, __LINE__); }
inline void cusparseAssert(cusparseStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUSPARSE_STATUS_SUCCESS) 
   {
        //Did not find a counter part to cudaGetErrorString in cusolver
        fprintf(stderr,"CUSPARSEassert: %s %d\n", file, line);
        if (abort) exit(code);
   }
}

cusolverDnHandle_t CreateCusolverDnHandle(int device) {
    if (cudaSetDevice(device) != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device.");
    }
    cusolverDnHandle_t handle;
    cusolverErrchk(cusolverDnCreate(&handle));
    return handle;
}


double solve_mkl_dgesv(
    double *matrix_dense,
    double *rhs,
    double *reference_solution,
    int matrice_size,
    double tolerance,
    bool flag_verbose)
{

    double time = -1.0;

    if(flag_verbose){
        std::printf("Copy data to device\n");
    }

    int ipiv[matrice_size];
    int nrhs = 1;
    int info;
    time = -omp_get_wtime();
    dgesv(&matrice_size, &nrhs, matrix_dense, &matrice_size, ipiv, rhs, &matrice_size, &info);
    time += omp_get_wtime();

    if(info != 0){
        std::printf("Error in MKL dgesv\n");
        std::printf("info: %d\n", info);
        return -1.0;
    }

    if(flag_verbose){
        std::printf("MKL dgesv done\n");
    }

    if(!assert_same_array<double>(rhs, reference_solution, tolerance, matrice_size)){
        std::printf("Error: MKL dgesv solution is not the same as the reference solution\n");
        return -1.0;
    }
    else{
        std::printf("MKL dgesv solution is the same as the reference solution\n");
    }
    return time;
}

double solve_cusolver_LU(
    double *matrix_dense_h,
    double *rhs_h,
    double *reference_solution_h,
    int matrice_size,
    double tolerance,
    bool flag_verbose)
{

    double time = -1.0;
    cudaStream_t stream = NULL;
    cusolverDnHandle_t handle = CreateCusolverDnHandle(0);
    cudaErrchk(cudaStreamCreate(&stream));
    cusolverErrchk(cusolverDnSetStream(handle, stream));



    int info_h = 0;
    int bufferSize = 0;

    double *matrix_dense_d = NULL;
    double *rhs_d = NULL;
    int *ipiv_d = NULL;
    int *info_d = NULL;
    double *buffer = NULL;

    //allocate memory on device
    cudaErrchk(cudaMalloc((void**)&info_d, sizeof(int)))
    cudaErrchk(cudaMalloc((void**)&matrix_dense_d, matrice_size*matrice_size*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&rhs_d, matrice_size*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&ipiv_d, matrice_size*sizeof(int)));


    //copy data to device
    if(flag_verbose){
        std::printf("Copy data to device\n");
    }
    cudaErrchk(cudaMemcpy(matrix_dense_d, matrix_dense_h, matrice_size*matrice_size*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemset(info_d, 0, sizeof(int)));
    cudaErrchk(cudaMemcpy(rhs_d, rhs_h, matrice_size*sizeof(double), cudaMemcpyHostToDevice));


    //figure out extra amount of memory needed
    cusolverErrchk(cusolverDnDgetrf_bufferSize(handle, matrice_size, matrice_size,
                                            (double *)matrix_dense_d,
                                              matrice_size, &bufferSize));
    cudaErrchk(cudaMalloc(&buffer, sizeof(double) * bufferSize));

    //LU factorization
    if(flag_verbose){
        std::printf("LU factorization\n");
    }
    time = -omp_get_wtime();
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaStreamSynchronize(stream));
    cusolverErrchk(cusolverDnDgetrf(handle, matrice_size, matrice_size,
                                matrix_dense_d, matrice_size, buffer, ipiv_d, info_d));
    
    //copy info to host
    cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_h != 0) {
        fprintf(stderr, "Error: LU factorization failed\n");
    }
    else{
        std::printf("LU factorization done\n");
    }

    if(flag_verbose){
        std::printf("Back substitution\n");
    }
    //back substitution
    cusolverErrchk(cusolverDnDgetrs(handle, CUBLAS_OP_N, matrice_size,
                                    1, matrix_dense_d, matrice_size, ipiv_d,
                                    rhs_d, matrice_size, info_d));
    cudaErrchk(cudaStreamSynchronize(stream));
    cudaErrchk(cudaDeviceSynchronize());
    time += omp_get_wtime();


    cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_h != 0) {
        fprintf(stderr, "Error: Back substitution failed\n");
    }
    else{
        std::printf("Back substitution done\n");
    }

    //copy solution to host
    if(flag_verbose){
        std::printf("Copy solution to host\n");
    }
    cudaErrchk(cudaMemcpy(rhs_h, rhs_d, matrice_size*sizeof(double), cudaMemcpyDeviceToHost));

    if(!assert_same_array<double>(rhs_h, reference_solution_h, tolerance, matrice_size)){
        std::printf("Error: CuSolver solution is not the same as the reference solution\n");
        return -1.0;
    }
    else{
        std::printf("CuSolver solution is the same as the reference solution\n");
    }


    if (info_d) {
        cudaErrchk(cudaFree(info_d));
    }
    if (buffer) {
        cudaErrchk(cudaFree(buffer));
    }
    if (matrix_dense_d) {
        cudaErrchk(cudaFree(matrix_dense_d));
    }
    if (ipiv_d) {
        cudaErrchk(cudaFree(ipiv_d));
    }


    if (handle) {
        cusolverErrchk(cusolverDnDestroy(handle));
    }
    if (stream) {
        cudaErrchk(cudaStreamDestroy(stream));
    }

    return time;
}


double solve_cusparse_ILU_CG(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution_h,
    int nnz,
    int matrice_size,
    double tolerance,
    bool flag_verbose){

    double time = -1.0;
    cudaStream_t stream = NULL;
    
    
    cusparseHandle_t cusparseHandle = 0;
    cusparseErrchk(cusparseCreate(&cusparseHandle));    

    cublasHandle_t cublasHandle = 0;
    cublasErrchk(cublasCreate(&cublasHandle));

    cudaErrchk(cudaStreamCreate(&stream));
    cusparseErrchk(cusparseSetStream(cusparseHandle, stream));
    cublasErrchk(cublasSetStream(cublasHandle, stream));


    double *data_d = NULL;
    double *col_indices_d = NULL;
    double *row_indptr_d = NULL;
    double *rhs_d = NULL;
    double *x_d = NULL;
    double *p_d = NULL;
    double *Ax_d = NULL;
    double dot;

    cusparseSpMatDescr_t matA = NULL;

    const double tol = 1.e-15;
    const int max_iter = 2000;
    double a, b, na;
    double alpha, beta, alpham1, r0, r1;
    size_t bufferSize = 0;
    void *buffer = NULL;

    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.0;


    //allocate memory on device
    cudaErrchk(cudaMalloc((void**)&data_d, nnz*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&col_indices_d, nnz*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&row_indptr_d, (matrice_size+1)*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&rhs_d, matrice_size*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&x_d, matrice_size*sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&p_d, matrice_size * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&Ax_d, matrice_size * sizeof(double)));

    /* Wrap raw data into cuSPARSE generic API objects */
    cusparseErrchk(cusparseCreateCsr(&matA, matrice_size, matrice_size,
                                        nnz, row_indptr_d, col_indices_d, data_d,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));



    cusparseDnVecDescr_t vecx = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecx, matrice_size, x_d, CUDA_R_64F));
    cusparseDnVecDescr_t vecp = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecp, matrice_size, p_d, CUDA_R_64F));
    cusparseDnVecDescr_t vecAx = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecAx, matrice_size, Ax_d, CUDA_R_64F));


    //copy data to device
    if(flag_verbose){
        std::printf("Copy data to device\n");
    }
    cudaErrchk(cudaMemcpy(rhs_d, rhs_h, matrice_size*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(col_indices_d, col_indices_h, nnz * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(row_indptr_d, row_indptr_h, (matrice_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(data_d, data_h, nnz * sizeof(double), cudaMemcpyHostToDevice));
    
    // setting starting guess to zero
    cudaErrchk(cudaMemset(x_d, 0.0, matrice_size*sizeof(double)))
    

    //figure out extra amount of memory needed
    if(flag_verbose){
        std::printf("Figure out extra amount of memory needed\n");
    }
    cusparseErrchk(cusparseSpMV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
        &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    cudaErrchk(cudaMalloc(&buffer, bufferSize));


    //begin CG
    time = -omp_get_wtime();
    cudaErrchk(cudaStreamSynchronize(stream));
    cudaErrchk(cudaDeviceSynchronize());
    if(flag_verbose){
        std::printf("CG starts\n");
    }

    // calc A*x
    cusparseErrchk(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecx, &beta, vecAx, CUDA_R_64F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    // r = b - A*x
    cublasErrchk(cublasDaxpy(cublasHandle, matrice_size, &alpham1, Ax_d, 1, rhs_d, 1));
    cublasErrchk(cublasDdot(cublasHandle, matrice_size, rhs_d, 1, rhs_d, 1, &r1));


    int k = 1;
    while (r1 > tol * tol && k <= max_iter) {
        if(k > 1){
            b = r1 / r0;
            cublasErrchk(cublasDscal(cublasHandle, matrice_size, &b, p_d, 1));
            cublasErrchk(cublasDaxpy(cublasHandle, matrice_size, &alpha, rhs_d, 1, p_d, 1));            
        }
        else {
            cublasErrchk(cublasDcopy(cublasHandle, matrice_size, rhs_d, 1, p_d, 1));
        }

        cusparseErrchk(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp,
            &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    
        cublasErrchk(cublasDdot(cublasHandle, matrice_size, p_d, 1, Ax_d, 1, &dot));
        a = r1 / dot;

        cublasErrchk(cublasDaxpy(cublasHandle, matrice_size, &a, p_d, 1, x_d, 1));
        na = -a;
        cublasErrchk(cublasDaxpy(cublasHandle, matrice_size, &na, Ax_d, 1, rhs_d, 1));

        r0 = r1;
        cublasErrchk(cublasDdot(cublasHandle, matrice_size, rhs_d, 1, rhs_d, 1, &r1));
        cudaErrchk(cudaStreamSynchronize(stream));
        if(flag_verbose){
            std::printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        }
        k++;
    }

    //end CG
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaStreamSynchronize(stream));
    time += omp_get_wtime();

    //copy solution to host
    if(flag_verbose){
        std::printf("Copy solution to host\n");
    }
    cudaErrchk(cudaMemcpy(rhs_h, x_d, matrice_size * sizeof(double), cudaMemcpyDeviceToHost));


    if(!assert_same_array<double>(rhs_h, reference_solution_h, tolerance, matrice_size)){
        std::printf("Error: CG solution is not the same as the reference solution\n");
        return -1.0;
    }
    else{
        std::printf("CG solution is the same as the reference solution\n");
    }


    if (buffer) {
        cudaErrchk(cudaFree(buffer));
    }
    if(cusparseHandle) {
        cusparseErrchk(cusparseDestroy(cusparseHandle));
    }
    if(cublasHandle) {
        cublasErrchk(cublasDestroy(cublasHandle));
    }
    if(stream) {
        cudaErrchk(cudaStreamDestroy(stream));
    }
    if(matA) {
        cusparseErrchk(cusparseDestroySpMat(matA));
    }
    if(vecx) {
        cusparseErrchk(cusparseDestroyDnVec(vecx));
    }
    if(vecAx) {
        cusparseErrchk(cusparseDestroyDnVec(vecAx));
    }
    if(vecp) {
        cusparseErrchk(cusparseDestroyDnVec(vecp));
    }

    if(data_d){
        cudaErrchk(cudaFree(data_d));
    }
    if(col_indices_d){
        cudaErrchk(cudaFree(col_indices_d));
    }
    if(row_indptr_d){
        cudaErrchk(cudaFree(row_indptr_d));
    }
    if(rhs_d){
        cudaErrchk(cudaFree(rhs_d));
    }
    if(x_d){
        cudaErrchk(cudaFree(x_d));
    }
    if(p_d){
        cudaErrchk(cudaFree(p_d));
    }
    if(Ax_d){
        cudaErrchk(cudaFree(Ax_d));
    }

    return time;
}

double solve_cusparse_CG(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution_h,
    int nnz,
    int matrice_size,
    double tolerance,
    bool flag_verbose){

    double time = -1.0;
    cudaStream_t stream = NULL;
    
    
    cusparseHandle_t cusparseHandle = 0;
    cusparseErrchk(cusparseCreate(&cusparseHandle));    

    cublasHandle_t cublasHandle = 0;
    cublasErrchk(cublasCreate(&cublasHandle));

    cudaErrchk(cudaStreamCreate(&stream));
    cusparseErrchk(cusparseSetStream(cusparseHandle, stream));
    cublasErrchk(cublasSetStream(cublasHandle, stream));


    double *data_d = NULL;
    double *col_indices_d = NULL;
    double *row_indptr_d = NULL;
    double *rhs_d = NULL;
    double *x_d = NULL;
    double *p_d = NULL;
    double *Ax_d = NULL;
    double dot;

    cusparseSpMatDescr_t matA = NULL;

    const double tol = 1.e-15;
    const int max_iter = 2000;
    double a, b, na;
    double alpha, beta, alpham1, r0, r1;
    size_t bufferSize = 0;
    void *buffer = NULL;

    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.0;


    //allocate memory on device
    cudaErrchk(cudaMalloc((void**)&data_d, nnz*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&col_indices_d, nnz*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&row_indptr_d, (matrice_size+1)*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&rhs_d, matrice_size*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&x_d, matrice_size*sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&p_d, matrice_size * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&Ax_d, matrice_size * sizeof(double)));

    /* Wrap raw data into cuSPARSE generic API objects */
    cusparseErrchk(cusparseCreateCsr(&matA, matrice_size, matrice_size,
                                        nnz, row_indptr_d, col_indices_d, data_d,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));



    cusparseDnVecDescr_t vecx = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecx, matrice_size, x_d, CUDA_R_64F));
    cusparseDnVecDescr_t vecp = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecp, matrice_size, p_d, CUDA_R_64F));
    cusparseDnVecDescr_t vecAx = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecAx, matrice_size, Ax_d, CUDA_R_64F));


    //copy data to device
    if(flag_verbose){
        std::printf("Copy data to device\n");
    }
    cudaErrchk(cudaMemcpy(rhs_d, rhs_h, matrice_size*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(col_indices_d, col_indices_h, nnz * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(row_indptr_d, row_indptr_h, (matrice_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(data_d, data_h, nnz * sizeof(double), cudaMemcpyHostToDevice));
    
    // setting starting guess to zero
    cudaErrchk(cudaMemset(x_d, 0.0, matrice_size*sizeof(double)))
    

    //figure out extra amount of memory needed
    if(flag_verbose){
        std::printf("Figure out extra amount of memory needed\n");
    }
    cusparseErrchk(cusparseSpMV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
        &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    cudaErrchk(cudaMalloc(&buffer, bufferSize));


    //begin CG
    time = -omp_get_wtime();
    cudaErrchk(cudaStreamSynchronize(stream));
    cudaErrchk(cudaDeviceSynchronize());
    if(flag_verbose){
        std::printf("CG starts\n");
    }

    // calc A*x
    cusparseErrchk(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecx, &beta, vecAx, CUDA_R_64F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    // r = b - A*x
    cublasErrchk(cublasDaxpy(cublasHandle, matrice_size, &alpham1, Ax_d, 1, rhs_d, 1));
    cublasErrchk(cublasDdot(cublasHandle, matrice_size, rhs_d, 1, rhs_d, 1, &r1));


    int k = 1;
    while (r1 > tol * tol && k <= max_iter) {
        if(k > 1){
            b = r1 / r0;
            cublasErrchk(cublasDscal(cublasHandle, matrice_size, &b, p_d, 1));
            cublasErrchk(cublasDaxpy(cublasHandle, matrice_size, &alpha, rhs_d, 1, p_d, 1));            
        }
        else {
            cublasErrchk(cublasDcopy(cublasHandle, matrice_size, rhs_d, 1, p_d, 1));
        }

        cusparseErrchk(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp,
            &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    
        cublasErrchk(cublasDdot(cublasHandle, matrice_size, p_d, 1, Ax_d, 1, &dot));
        a = r1 / dot;

        cublasErrchk(cublasDaxpy(cublasHandle, matrice_size, &a, p_d, 1, x_d, 1));
        na = -a;
        cublasErrchk(cublasDaxpy(cublasHandle, matrice_size, &na, Ax_d, 1, rhs_d, 1));

        r0 = r1;
        cublasErrchk(cublasDdot(cublasHandle, matrice_size, rhs_d, 1, rhs_d, 1, &r1));
        cudaErrchk(cudaStreamSynchronize(stream));
        if(flag_verbose){
            std::printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        }
        k++;
    }

    //end CG
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaStreamSynchronize(stream));
    time += omp_get_wtime();

    //copy solution to host
    if(flag_verbose){
        std::printf("Copy solution to host\n");
    }
    cudaErrchk(cudaMemcpy(rhs_h, x_d, matrice_size * sizeof(double), cudaMemcpyDeviceToHost));


    if(!assert_same_array<double>(rhs_h, reference_solution_h, tolerance, matrice_size)){
        std::printf("Error: CG solution is not the same as the reference solution\n");
        return -1.0;
    }
    else{
        std::printf("CG solution is the same as the reference solution\n");
    }


    if (buffer) {
        cudaErrchk(cudaFree(buffer));
    }
    if(cusparseHandle) {
        cusparseErrchk(cusparseDestroy(cusparseHandle));
    }
    if(cublasHandle) {
        cublasErrchk(cublasDestroy(cublasHandle));
    }
    if(stream) {
        cudaErrchk(cudaStreamDestroy(stream));
    }
    if(matA) {
        cusparseErrchk(cusparseDestroySpMat(matA));
    }
    if(vecx) {
        cusparseErrchk(cusparseDestroyDnVec(vecx));
    }
    if(vecAx) {
        cusparseErrchk(cusparseDestroyDnVec(vecAx));
    }
    if(vecp) {
        cusparseErrchk(cusparseDestroyDnVec(vecp));
    }

    if(data_d){
        cudaErrchk(cudaFree(data_d));
    }
    if(col_indices_d){
        cudaErrchk(cudaFree(col_indices_d));
    }
    if(row_indptr_d){
        cudaErrchk(cudaFree(row_indptr_d));
    }
    if(rhs_d){
        cudaErrchk(cudaFree(rhs_d));
    }
    if(x_d){
        cudaErrchk(cudaFree(x_d));
    }
    if(p_d){
        cudaErrchk(cudaFree(p_d));
    }
    if(Ax_d){
        cudaErrchk(cudaFree(Ax_d));
    }

    return time;
}


