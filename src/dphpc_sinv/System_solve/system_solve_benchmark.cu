#include <string> 
#include <omp.h>

#include "utils.h"

#include "mkl.h"
#include "cusolverDn.h"




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
    double epsilon)
{

    double time = 0.0;

    printf("Start MKL dgesv\n");

    int ipiv[matrice_size];
    int nrhs = 1;
    int info;
    time -= omp_get_wtime();
    dgesv(&matrice_size, &nrhs, matrix_dense, &matrice_size, ipiv, rhs, &matrice_size, &info);
    time += omp_get_wtime();

    if(info != 0){
        printf("Error in MKL dgesv\n");
        printf("info: %d\n", info);
        return -1.0;
    }

    printf("MKL dgesv done\n");

    if(!assert_same_array<double>(rhs, reference_solution, epsilon, matrice_size)){
        printf("Error: MKL dgesv solution is not the same as the reference solution\n");
        return -1.0;
    }
    else{
        printf("MKL dgesv solution is the same as the reference solution\n");
    }
    return time;
}

double solve_cusolver_LU(
    double *matrix_dense_h,
    double *rhs_h,
    double *reference_solution_h,
    int matrice_size,
    double epsilon
){

    double time = 0.0;
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
    printf("Copy data to device\n");
    cudaErrchk(cudaMemcpy(matrix_dense_d, matrix_dense_h, matrice_size*matrice_size*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemset(info_d, 0, sizeof(int)));
    cudaErrchk(cudaMemcpy(rhs_d, rhs_h, matrice_size*sizeof(double), cudaMemcpyHostToDevice));


    //figure out extra amount of memory needed
    cusolverErrchk(cusolverDnDgetrf_bufferSize(handle, matrice_size, matrice_size,
                                            (double *)matrix_dense_d,
                                              matrice_size, &bufferSize));
    cudaErrchk(cudaMalloc(&buffer, sizeof(double) * bufferSize));

    //LU factorization
    printf("LU factorization\n");
    time -= omp_get_wtime();
    cudaErrchk(cudaStreamSynchronize(stream));
    cusolverErrchk(cusolverDnDgetrf(handle, matrice_size, matrice_size,
                                matrix_dense_d, matrice_size, buffer, ipiv_d, info_d));
    cudaErrchk(cudaStreamSynchronize(stream));
    
    //copy info to host
    cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_h != 0) {
        fprintf(stderr, "Error: LU factorization failed\n");
    }
    else{
        printf("LU factorization done\n");
    }

    printf("Back substitution\n");
    cudaErrchk(cudaStreamSynchronize(stream));
    //back substitution
    cusolverErrchk(cusolverDnDgetrs(handle, CUBLAS_OP_N, matrice_size,
                                    1, matrix_dense_d, matrice_size, ipiv_d,
                                    rhs_d, matrice_size, info_d));
    cudaErrchk(cudaStreamSynchronize(stream));
    time += omp_get_wtime();


    cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_h != 0) {
        fprintf(stderr, "Error: Back substitution failed\n");
    }
    else{
        printf("Back substitution done\n");
    }

    //copy solution to host
    printf("Copy solution to host\n");
    cudaErrchk(cudaMemcpy(rhs_h, rhs_d, matrice_size*sizeof(double), cudaMemcpyDeviceToHost));

    if(!assert_same_array<double>(rhs_h, reference_solution_h, epsilon, matrice_size)){
        printf("Error: CuSolver solution is not the same as the reference solution\n");
        return -1.0;
    }
    else{
        printf("CuSolver solution is the same as the reference solution\n");
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


// todo I am not sure how to "corretly" compine .cu and .cpp
// what is the "correct" way
bool benchmark()
{
    // Get matrix parameters
    char path_data[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/matrix_sparse_data0.txt";
    char path_indices[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/matrix_sparse_indices0.txt";
    char path_indptr[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/matrix_sparse_indptr0.txt";
    char path_rhs[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/rhs_0.txt";
    char path_reference_solution[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/x_ref0.txt";
    int matrice_size = 7165;
    int number_of_nonzero = 182287;
    double epsilon = 1e-10;

    //print the matrix parameters
    printf("Matrix parameters:\n");
    printf("Matrix size: %d\n", matrice_size);
    printf("Number of nonzero: %d\n", number_of_nonzero);

    double *dense_matrix = (double*)malloc(matrice_size*matrice_size*sizeof(double));
    double *data = (double*)malloc(number_of_nonzero*sizeof(double));
    int *indices = (int*)malloc(number_of_nonzero*sizeof(int));
    int *indptr = (int*)malloc((matrice_size+1)*sizeof(int));
    double *rhs = (double*)malloc(matrice_size*sizeof(double));
    double *dense_matrix_copy = (double*)malloc(matrice_size*matrice_size*sizeof(double));
    double *rhs_copy = (double*)malloc(matrice_size*sizeof(double));
    double *reference_solution = (double*)malloc(matrice_size*sizeof(double));


    if(!load_text_array<double>(path_data, data, number_of_nonzero)){
        printf("Error loading data\n");
        return false;
    }
    if(!load_text_array<int>(path_indices, indices, number_of_nonzero)){
        printf("Error loading indices\n");
        return false;
    }
    if(!load_text_array<int>(path_indptr, indptr, matrice_size+1)){
        printf("Error loading indptr\n");
        return false;
    }
    if(!load_text_array<double>(path_rhs, rhs, matrice_size)){
        printf("Error loading rhs\n");
        return false;
    }
    if(!load_text_array<double>(path_reference_solution, reference_solution, matrice_size)){
        printf("Error loading reference solution\n");
        return false;
    }

    sparse_to_dense<double>(
        dense_matrix,
        data,
        indices,
        indptr,
        matrice_size);


    // char path_save[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test0.txt";
    // if(!save_text_array<double>(path_save, dense_matrix, matrice_size*matrice_size)){
    //     printf("Error saving dense matrix\n");
    //     return false;
    // }

    //copy dense matrix
    copy_array<double>(dense_matrix, dense_matrix_copy, matrice_size*matrice_size);
    copy_array<double>(rhs, rhs_copy, matrice_size);

    // double time_mkl_dense = solve_mkl_dgesv(
    //     dense_matrix_copy,
    //     rhs_copy,
    //     reference_solution,
    //     matrice_size,
    //     epsilon);
    // if(time_mkl_dense < 0.0){
    //     return false;
    // }
    // else{
    //     printf("Time MKL dgesv: %f\n s", time_mkl_dense);
    // }

    double time_cusolve_dense = solve_cusolver_LU(
        dense_matrix_copy,
        rhs_copy,
        reference_solution,
        matrice_size,
        epsilon);
    if(time_cusolve_dense < 0.0){
        return false;
    }
    else{
        printf("Time cusolver dense: %f\n s", time_cusolve_dense);
    }

    // char path_save[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/x0.txt";
    // if(!save_text_array<double>(path_save, rhs_copy, matrice_size)){
    //     printf("Error saving dense matrix\n");
    //     return false;
    // }


    free(dense_matrix_copy);
    free(rhs_copy);
    free(dense_matrix);
    free(data);
    free(indices);
    free(indptr);
    free(rhs);

    return true;
}