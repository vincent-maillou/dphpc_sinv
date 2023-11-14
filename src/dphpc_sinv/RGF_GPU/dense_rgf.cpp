// Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.

#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <iostream>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>
#include <cuda/std/complex>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include <Eigen/Dense>



#include "utils.h"


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

cusolverDnHandle_t CreateCusolverDnHandle(int device) {
    if (cudaSetDevice(device) != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device.");
    }
    cusolverDnHandle_t cusolver_handle;
    cusolverErrchk(cusolverDnCreate(&cusolver_handle));
    return cusolver_handle;
}


using complex_h = std::complex<double>;
using complex_d = cuDoubleComplex; //cuda::std::complex<double>;
int main() {
 
    if(sizeof(complex_h) != sizeof(complex_d)){
        printf("Error: complex_h and complex_d have different sizes\n");
        return 1;
    }
    else{
        printf("complex_h and complex_d have the same size\n");
    }
 
    // Get matrix parameters
    char f_matparam[] = "../../../tests/tests_cases/dense_blocks_matrix_0_parameters.txt";
    unsigned int matrix_size;
    unsigned int blocksize;

    load_matrix_parameters(f_matparam, &matrix_size, &blocksize);

    unsigned int n_blocks = matrix_size / blocksize;
    unsigned int off_diag_size = matrix_size - blocksize;

    // Print the matrix parameters
    printf("Matrix parameters:\n");
    printf("    Matrix size: %d\n", matrix_size);
    printf("    Block size: %d\n", blocksize);


    // Load matrix to invert
    std::complex<double>* matrix_diagblk = (std::complex<double>*) malloc(blocksize * matrix_size * sizeof(std::complex<double>));
    char f_mat_diagblk[] = "../../../tests/tests_cases/dense_blocks_matrix_0_diagblk.bin";
    load_binary_matrix(f_mat_diagblk, matrix_diagblk, blocksize, matrix_size);

    std::complex<double>* matrix_upperblk = (std::complex<double>*) malloc(blocksize * (off_diag_size) * sizeof(std::complex<double>));
    char f_mat_upperblk[] = "../../../tests/tests_cases/dense_blocks_matrix_0_upperblk.bin";
    load_binary_matrix(f_mat_upperblk, matrix_upperblk, blocksize, off_diag_size);

    std::complex<double>* matrix_lowerblk = (std::complex<double>*) malloc(blocksize * (off_diag_size) * sizeof(std::complex<double>));
    char f_mat_lowerblk[] = "../../../tests/tests_cases/dense_blocks_matrix_0_lowerblk.bin";
    load_binary_matrix(f_mat_lowerblk, matrix_lowerblk, blocksize, off_diag_size);

    /*
    Matrices are saved in the following way:

    matrix_diagblk = [A_0, A_1, ..., A_n]
    matrix_upperblk = [B_0, B_1, ..., B_n-1]
    matrix_lowerblk = [C_0, C_1, ..., C_n-1]

    where A_i, B_i, C_i are block matrices of size blocksize x blocksize

    The three above arrays are in Row-Major order which means the blocks are not contiguous in memory.

    Below they will be transformed to the following layout:

    matrix_diagblk_cont = [A_0;
                             A_1;
                             ...;
                             A_n]
    matrix_upperblk_cont = [B_0;
                              B_1;
                              ...;
                              B_n-1]
    matrix_lowerblk_cont = [C_0;
                              C_1;
                              ...;
                              C_n-1]

    where blocks are in column major order
    */


    complex_h* matrix_diagblk_cont = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
    complex_h* matrix_upperblk_cont = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
    complex_h* matrix_lowerblk_cont = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));


    for(unsigned int i = 0; i < blocksize * matrix_size; i++){
        // block index
        int k = i / (blocksize * blocksize);
        // index inside block
        int h = i % (blocksize * blocksize);
        // row inside block
        int m = h % blocksize;
        // col inside block
        int n = h / blocksize;
        matrix_diagblk_cont[i] = matrix_diagblk[m*matrix_size + k*blocksize + n];
    }
    for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
        // block index
        int k = i / (blocksize * blocksize);
        // index inside block
        int h = i % (blocksize * blocksize);
        // row inside block
        int m = h % blocksize;
        // col inside block
        int n = h / blocksize;
        matrix_upperblk_cont[i] = matrix_upperblk[m*off_diag_size + k*blocksize + n];
        matrix_lowerblk_cont[i] = matrix_lowerblk[m*off_diag_size + k*blocksize + n];
    }
    // for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
    //     std::cout << "matrix_upperblk_cont[" << i << "] = " << matrix_upperblk_cont[i] << std::endl;
    // }

    complex_d* matrix_diagblk_d = NULL;
    complex_d* matrix_upperblk_d = NULL;
    complex_d* matrix_lowerblk_d = NULL;
    
    cudaStream_t stream = NULL;
    cudaErrchk(cudaStreamCreate(&stream));
    cusolverDnHandle_t cusolver_handle = CreateCusolverDnHandle(0);
    cusolverErrchk(cusolverDnSetStream(cusolver_handle, stream));

    cublasHandle_t cublas_handle = 0;
    cublasErrchk(cublasCreate(&cublas_handle));
    cublasErrchk(cublasSetStream(cublas_handle, stream));


    cudaErrchk(cudaMalloc((void**)&matrix_diagblk_d, blocksize * matrix_size * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&matrix_upperblk_d, blocksize * (off_diag_size) * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&matrix_lowerblk_d, blocksize * (off_diag_size) * sizeof(complex_d)));

    cudaErrchk(cudaMemcpy(matrix_diagblk_d, reinterpret_cast<const complex_d*>(matrix_diagblk_cont),
                blocksize * matrix_size * sizeof(complex_d), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(matrix_upperblk_d, reinterpret_cast<const complex_d*>(matrix_upperblk_cont),
                blocksize * (off_diag_size) * sizeof(complex_d), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(matrix_lowerblk_d, reinterpret_cast<const complex_d*>(matrix_lowerblk_cont),
                blocksize * (off_diag_size) * sizeof(complex_d), cudaMemcpyHostToDevice));

    // Transform C style storage to vector of eigen matrices
    std::vector<Eigen::MatrixXcd> eig_diagblk;
    std::vector<Eigen::MatrixXcd> eig_upperblk;
    std::vector<Eigen::MatrixXcd> eig_lowerblk;

    for (unsigned int i = 0; i < n_blocks; ++i) {
        eig_diagblk.push_back(Eigen::Map<Eigen::MatrixXcd, 0, Eigen::Stride<1, 10>>
            (reinterpret_cast<std::complex<double>*>(matrix_diagblk + i*blocksize),
            blocksize, blocksize));

        if(i < n_blocks-1){
            eig_upperblk.push_back(Eigen::Map<Eigen::MatrixXcd, 0, Eigen::Stride<1, 8>>
                (reinterpret_cast<std::complex<double>*>(matrix_upperblk + i*blocksize), blocksize, blocksize));

            eig_lowerblk.push_back(Eigen::Map<Eigen::MatrixXcd, 0, Eigen::Stride<1, 8>>
                (reinterpret_cast<std::complex<double>*>(matrix_lowerblk + i*blocksize), blocksize, blocksize));
        }
    }

    // ----- END OF INIT SECTION -----


    // Pre-allocate memory for the inverted blocks
    std::vector<Eigen::MatrixXcd> eig_inv_diagblk(n_blocks);
    std::vector<Eigen::MatrixXcd> eig_inv_upperblk(n_blocks-1);
    std::vector<Eigen::MatrixXcd> eig_inv_lowerblk(n_blocks-1);

    // 0. Inverse of the first block
    eig_inv_diagblk[0] = eig_diagblk[0].inverse();

    // 1. Forward substitution (performed left to right)
    for (unsigned int i = 1; i < n_blocks; ++i) {
        eig_inv_diagblk[i] = (eig_diagblk[i] - eig_lowerblk[i-1] * eig_inv_diagblk[i-1] * eig_upperblk[i-1]).inverse();
    }

    // 2. Backward substitution (performed right to left)
    for(int i = n_blocks-2; i >= 0; --i){
        Eigen::MatrixXcd tmp_lowerfactor = eig_inv_diagblk[i+1] * eig_lowerblk[i] * eig_inv_diagblk[i];

        eig_inv_lowerblk[i] = -tmp_lowerfactor;
        eig_inv_upperblk[i] = -eig_inv_diagblk[i] * eig_upperblk[i] * eig_inv_diagblk[i+1];

        eig_inv_diagblk[i] += eig_inv_diagblk[i] * eig_upperblk[i] * tmp_lowerfactor;
    }


    complex_h* inv_diagblk_h = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
    complex_h* inv_upperblk_h = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
    complex_h* inv_lowerblk_h = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));


    complex_d* inv_diagblk_d = NULL;
    complex_d* inv_upperblk_d = NULL;
    complex_d* inv_lowerblk_d = NULL;

    cudaErrchk(cudaMalloc((void**)&inv_diagblk_d, blocksize * matrix_size * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&inv_upperblk_d, blocksize * (off_diag_size) * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&inv_lowerblk_d, blocksize * (off_diag_size) * sizeof(complex_d)));

    // create right hand side identity matrix
    complex_h* identity_h = (complex_h*) malloc(blocksize * blocksize * sizeof(complex_h));

    for(unsigned int i = 0; i < blocksize * blocksize; i++){
        identity_h[i] = 0.0;
        if(i / blocksize == i % blocksize){
            identity_h[i] = 1.0;
        }
    }

    complex_d* identity_d = NULL;
    complex_d* identity_cpy_d = NULL;
    cudaErrchk(cudaMalloc((void**)&identity_d, blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&identity_cpy_d, blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMemcpy(identity_d, reinterpret_cast<const complex_d*>(identity_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(identity_cpy_d, identity_d,
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice));


    int info_h = 0;
    int *ipiv_d = NULL;
    int *info_d = NULL;
    cudaErrchk(cudaMalloc((void**)&info_d, sizeof(int)))
    cudaErrchk(cudaMalloc((void**)&ipiv_d, matrix_size*sizeof(int)));

    //figure out extra amount of memory needed
    complex_d *buffer = NULL;
    int bufferSize = 0;
    cusolverErrchk(cusolverDnZgetrf_bufferSize(cusolver_handle, blocksize, blocksize,
                                            (complex_d *)matrix_diagblk_d,
                                              blocksize, &bufferSize));
    cudaErrchk(cudaMalloc((void**)&buffer, sizeof(complex_d) * bufferSize));


    cudaErrchk(cudaStreamSynchronize(stream));
    cusolverErrchk(cusolverDnZgetrf(cusolver_handle, blocksize, blocksize,
                                matrix_diagblk_d, blocksize, buffer, ipiv_d, info_d));
    
    //copy info to host
    cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_h != 0) {
        fprintf(stderr, "Error: LU factorization failed\n");
    }
    else{
        std::printf("LU factorization done\n");
    }

    //back substitution
    cusolverErrchk(cusolverDnZgetrs(cusolver_handle, CUBLAS_OP_N, blocksize,
                                    blocksize, matrix_diagblk_d, blocksize, ipiv_d,
                                    identity_cpy_d, blocksize, info_d));
    cudaErrchk(cudaStreamSynchronize(stream));
    //copy info to host
    cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_h != 0) {
        fprintf(stderr, "Error: Backsub failed\n");
    }
    else{
        std::printf("Backsub done\n");
    }


    // // 0. Inverse of the first block
    cudaErrchk(cudaMemcpy(inv_diagblk_d, identity_cpy_d,
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice));



    complex_d alpha;
    complex_d beta;
    // // 1. Forward substitution (performed left to right)
    for (unsigned int i = 1; i < n_blocks; ++i) {

        // MatMul tmp = eig_lowerblk[i-1] * eig_inv_diagblk[i-1]
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            matrix_lowerblk_d + (i-1)*blocksize*blocksize, blocksize,
            inv_diagblk_d + (i-1)*blocksize*blocksize, blocksize,
            &beta,
            inv_lowerblk_d, blocksize));
        //MatMul eig_diagblk[i] - tmp * eig_upperblk[i-1]
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            inv_lowerblk_d, blocksize,
            matrix_upperblk_d + (i-1)*blocksize*blocksize, blocksize,
            &beta,
            matrix_diagblk_d + i*blocksize*blocksize, blocksize));

        // invert schur complement
        cudaErrchk(cudaMemcpy(identity_cpy_d, identity_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice));

        cudaErrchk(cudaStreamSynchronize(stream));
        cusolverErrchk(cusolverDnZgetrf(cusolver_handle, blocksize, blocksize,
                                    matrix_diagblk_d + i*blocksize*blocksize,
                                    blocksize, buffer, ipiv_d, info_d));
        
        //copy info to host
        cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));

        if (info_h != 0) {
            fprintf(stderr, "Error: LU factorization failed\n");
        }
        else{
            std::printf("LU factorization done\n");
        }

        //back substitution
        cusolverErrchk(cusolverDnZgetrs(cusolver_handle, CUBLAS_OP_N, blocksize,
                                        blocksize,
                                        matrix_diagblk_d + i*blocksize*blocksize,
                                        blocksize, ipiv_d,
                                        identity_cpy_d, blocksize, info_d));
        cudaErrchk(cudaStreamSynchronize(stream));
        //copy info to host
        cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));

        if (info_h != 0) {
            fprintf(stderr, "Error: Backsub failed\n");
        }
        else{
            std::printf("Backsub done\n");
        }

        cudaErrchk(cudaMemcpy(inv_diagblk_d + i*blocksize*blocksize,
                    identity_cpy_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice));


    }


    // 2. Backward substitution (performed right to left)
    for(int i = n_blocks-2; i >= 0; --i){
        //tmp = eig_inv_diagblk[i+1] * eig_lowerblk[i]
        // use identity_cpy_d as tmp
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            inv_diagblk_d + (i+1)*blocksize*blocksize, blocksize,
            matrix_lowerblk_d + i*blocksize*blocksize, blocksize,
            &beta,
            identity_cpy_d, blocksize));

        // eig_inv_lowerblk[i] = -tmp*eig_inv_diagblk[i];
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            identity_cpy_d, blocksize,
            inv_diagblk_d + i*blocksize*blocksize, blocksize,
            &beta,
            inv_lowerblk_d + i*blocksize*blocksize, blocksize));

        // tmp = eig_inv_diagblk[i] * eig_upperblk[i]
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            inv_diagblk_d + i*blocksize*blocksize, blocksize,
            matrix_upperblk_d + i*blocksize*blocksize, blocksize,
            &beta,
            identity_cpy_d, blocksize));

        //eig_inv_upperblk[i] = -tmp * eig_inv_diagblk[i+1];
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            identity_cpy_d, blocksize,
            inv_diagblk_d + (i+1)*blocksize*blocksize, blocksize,
            &beta,
            inv_upperblk_d + i*blocksize*blocksize, blocksize));

        //eig_inv_diagblk[i] -= tmp * eig_inv_lowerblk[i];
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            identity_cpy_d, blocksize,
            inv_lowerblk_d + i*blocksize*blocksize, blocksize,
            &beta,
            inv_diagblk_d + i*blocksize*blocksize, blocksize));
    }


    cudaErrchk(cudaMemcpy(inv_diagblk_h, reinterpret_cast<const complex_h*>(inv_diagblk_d),
                blocksize * matrix_size * sizeof(complex_h), cudaMemcpyDeviceToHost));
    cudaErrchk(cudaMemcpy(inv_upperblk_h, reinterpret_cast<const complex_h*>(inv_upperblk_d),
                blocksize * off_diag_size * sizeof(complex_h), cudaMemcpyDeviceToHost));
    cudaErrchk(cudaMemcpy(inv_lowerblk_h, reinterpret_cast<const complex_h*>(inv_lowerblk_d),
                blocksize * off_diag_size * sizeof(complex_h), cudaMemcpyDeviceToHost));

    // ----- RESULT CHECKING SECTION -----

    // Load reference solution of the matrix inverse
    std::complex<double>* matrix_inv_diagblk = (std::complex<double>*) malloc(blocksize * matrix_size * sizeof(std::complex<double>));
    char f_mat_inv_diagblk[] = "../../../tests/tests_cases/dense_blocks_matrix_0_inverse_diagblk.bin";
    load_binary_matrix(f_mat_inv_diagblk, matrix_inv_diagblk, blocksize, matrix_size);

    std::complex<double>* matrix_inv_upperblk = (std::complex<double>*) malloc(blocksize * (off_diag_size) * sizeof(std::complex<double>));
    char f_mat_inv_upperblk[] = "../../../tests/tests_cases/dense_blocks_matrix_0_inverse_upperblk.bin";
    load_binary_matrix(f_mat_inv_upperblk, matrix_inv_upperblk, blocksize, off_diag_size);
    
    std::complex<double>* matrix_inv_lowerblk = (std::complex<double>*) malloc(blocksize * (off_diag_size) * sizeof(std::complex<double>));
    char f_mat_inv_lowerblk[] = "../../../tests/tests_cases/dense_blocks_matrix_0_inverse_lowerblk.bin";
    load_binary_matrix(f_mat_inv_lowerblk, matrix_inv_lowerblk, blocksize, off_diag_size);


    // Convert back std::vector<Eigen::Matrix> to std::vector<std::complex<double>>
    std::complex<double>* eigen_inverted_diagblk = (std::complex<double>*) malloc(blocksize * matrix_size * sizeof(std::complex<double>));
    std::complex<double>* eigen_inverted_upperblk = (std::complex<double>*) malloc(blocksize * off_diag_size * sizeof(std::complex<double>));
    std::complex<double>* eigen_inverted_lowerblk = (std::complex<double>*) malloc(blocksize * off_diag_size * sizeof(std::complex<double>));

    for(unsigned int i = 0; i < blocksize; i++){
        for(unsigned int j = 0; j < blocksize; j++){
            for(unsigned int k = 0; k < n_blocks; ++k){
                //std::cout << "vectorized eig_inv_diagblk[" << i << "][" << j+k*blocksize << "] = " << eig_inv_diagblk[k].data()[i+j*blocksize] << std::endl;
                unsigned int row = i*matrix_size;
                unsigned int col = j+k*blocksize;
                eigen_inverted_diagblk[row+col] = eig_inv_diagblk[k].data()[i+j*blocksize];
            }

            for(unsigned int k = 0; k < n_blocks-1; ++k){
                unsigned int row = i*off_diag_size;
                unsigned int col = j+k*blocksize;
                eigen_inverted_upperblk[row+col] = eig_inv_upperblk[k].data()[i+j*blocksize];
                eigen_inverted_lowerblk[row+col] = eig_inv_lowerblk[k].data()[i+j*blocksize];
            }
        }
    }

    complex_h* inv_diagblk_ref = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
    complex_h* inv_upperblk_ref = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
    complex_h* inv_lowerblk_ref = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));


    for(unsigned int i = 0; i < blocksize * matrix_size; i++){
        // block index
        int k = i / (blocksize * blocksize);
        // index inside block
        int h = i % (blocksize * blocksize);
        // row inside block
        int m = h % blocksize;
        // col inside block
        int n = h / blocksize;
        inv_diagblk_ref[i] = matrix_inv_diagblk[m*matrix_size + k*blocksize + n];
    }
    for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
        // block index
        int k = i / (blocksize * blocksize);
        // index inside block
        int h = i % (blocksize * blocksize);
        // row inside block
        int m = h % blocksize;
        // col inside block
        int n = h / blocksize;
        inv_upperblk_ref[i] = matrix_inv_upperblk[m*off_diag_size + k*blocksize + n];
        inv_lowerblk_ref[i] = matrix_inv_lowerblk[m*off_diag_size + k*blocksize + n];
    }

    // // print last block of inverted matrix
    // for(unsigned int i = blocksize *(matrix_size-blocksize); i < blocksize * matrix_size; i++){
    //     std::cout << "inv_diagblk_ref[" << i << "] = " << inv_diagblk_ref[i] << std::endl;
    //     std::cout << "inv_diagblk_h[" << i << "] = " << inv_diagblk_h[i] << std::endl;
    // }


    double norm_diagblk = 0.0;
    double norm_upperblk = 0.0;
    double norm_lowerblk = 0.0;
    double diff_diagblk = 0.0;
    double diff_upperblk = 0.0;
    double diff_lowerblk = 0.0;
    for(unsigned int i = 0; i < blocksize * matrix_size; i++){
        norm_diagblk += std::abs(inv_diagblk_h[i]);
        diff_diagblk += std::abs(inv_diagblk_h[i] - inv_diagblk_ref[i]);
    }
    for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
        norm_upperblk += std::abs(inv_upperblk_h[i]);
        norm_lowerblk += std::abs(inv_lowerblk_h[i]);
        diff_upperblk += std::abs(inv_upperblk_h[i] - inv_upperblk_ref[i]);
        diff_lowerblk += std::abs(inv_lowerblk_h[i] - inv_lowerblk_ref[i]);
    }
    double eps = 1e-12;
    if(diff_diagblk/norm_diagblk > eps){
        printf("Error: inv_diagblk_h and inv_diagblk_ref are not equal\n");
    }
    else{
        printf("inv_diagblk_h and inv_diagblk_ref are equal\n");
    }
    if(diff_upperblk/norm_upperblk > eps){
        printf("Error: inv_upperblk_h and inv_upperblk_ref are not equal\n");
    }
    else{
        printf("inv_upperblk_h and inv_upperblk_ref are equal\n");
    }
    if(diff_lowerblk/norm_lowerblk > eps){
        printf("Error: inv_lowerblk_h and inv_lowerblk_ref are not equal\n");
    }
    else{
        printf("inv_lowerblk_h and inv_lowerblk_ref are equal\n");
    }


    if(!are_equals(eigen_inverted_diagblk, matrix_inv_diagblk, matrix_size, blocksize)){
        printf("Error: eigen_inverted_diagblk and matrix_inv_diagblk are not equal\n");

        print_matrix(eigen_inverted_diagblk, blocksize, blocksize);
        print_matrix(matrix_inv_diagblk, blocksize, blocksize);
    }
    else{
        printf("eigen_inverted_diagblk and matrix_inv_diagblk are equal\n");
    }

    if(!are_equals(eigen_inverted_upperblk, matrix_inv_upperblk, off_diag_size, blocksize)){
        printf("Error: eigen_inverted_upperblk and matrix_inv_upperblk are not equal\n");
    }
    else{
        printf("eigen_inverted_upperblk and matrix_inv_upperblk are equal\n");
    }

    if(!are_equals(eigen_inverted_lowerblk, matrix_inv_lowerblk, off_diag_size, blocksize)){
        printf("Error: eigen_inverted_lowerblk and matrix_inv_lowerblk are not equal\n");
    }
    else{
        printf("eigen_inverted_lowerblk and matrix_inv_lowerblk are equal\n");
    }

    if (stream) {
        cudaErrchk(cudaStreamDestroy(stream));
    }
    if(matrix_diagblk_d) {
        cudaErrchk(cudaFree(matrix_diagblk_d));
    }
    if(matrix_upperblk_d) {
        cudaErrchk(cudaFree(matrix_upperblk_d));
    }
    if(matrix_lowerblk_d) {
        cudaErrchk(cudaFree(matrix_lowerblk_d));
    }
    if(inv_diagblk_d) {
        cudaErrchk(cudaFree(inv_diagblk_d));
    }
    if(inv_upperblk_d) {
        cudaErrchk(cudaFree(inv_upperblk_d));
    }
    if(inv_lowerblk_d) {
        cudaErrchk(cudaFree(inv_lowerblk_d));
    }
    if(inv_diagblk_h){
        free(inv_diagblk_h);
    }
    if(inv_upperblk_h){
        free(inv_upperblk_h);
    }
    if(inv_lowerblk_h){
        free(inv_lowerblk_h);
    }
    if(inv_diagblk_ref){
        free(inv_diagblk_ref);
    }
    if(inv_upperblk_ref){
        free(inv_upperblk_ref);
    }
    if(inv_lowerblk_ref){
        free(inv_lowerblk_ref);
    }
    if(buffer){
        cudaErrchk(cudaFree(buffer));
    }
    if(ipiv_d){
        cudaErrchk(cudaFree(ipiv_d));
    }
    if(info_d){
        cudaErrchk(cudaFree(info_d));
    }

    return 0;
}








