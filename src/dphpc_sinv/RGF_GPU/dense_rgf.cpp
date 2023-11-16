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


#include "utils.h"


#define cudaErrchk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      std::printf("CUDAassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define cusolverErrchk(ans) { cusolverAssert((ans), __FILE__, __LINE__); }
inline void cusolverAssert(cusolverStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUSOLVER_STATUS_SUCCESS) 
   {
        //Did not find a counter part to cudaGetErrorString in cusolver
        std::printf("CUSOLVERassert: %s %d\n", file, line);
        if (abort) exit(code);
   }
}


#define cublasErrchk(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUBLAS_STATUS_SUCCESS) 
   {
        //Did not find a counter part to cudaGetErrorString in cublas
        std::printf("CUBLASassert: %s %d\n", file, line);
        if (abort) exit(code);
   }
}

// both should be equivalent, thus reinterpret_cast should be fine
using complex_h = std::complex<double>;
using complex_d = cuDoubleComplex; //cuda::std::complex<double>;

bool rgf_matrix_fits_gpu_memory(
    unsigned int blocksize,
    unsigned int matrix_size,
    complex_h *matrix_diagblk_h,
    complex_h *matrix_upperblk_h,
    complex_h *matrix_lowerblk_h,
    complex_h *inv_diagblk_h,
    complex_h *inv_upperblk_h,
    complex_h *inv_lowerblk_h)
{
    if(matrix_size % blocksize != 0){
        printf("Error: matrix_size is not a multiple of blocksize\n");
        return false;
    }
    unsigned int n_blocks = matrix_size / blocksize;
    unsigned int off_diag_size = matrix_size - blocksize;
    bool success = true;

    // Init cuda stuff
    cudaStream_t stream = NULL;
    cudaErrchk(cudaStreamCreate(&stream));

    cusolverDnHandle_t cusolver_handle;
    cusolverErrchk(cusolverDnCreate(&cusolver_handle));
    cusolverErrchk(cusolverDnSetStream(cusolver_handle, stream));

    cublasHandle_t cublas_handle = 0;
    cublasErrchk(cublasCreate(&cublas_handle));
    cublasErrchk(cublasSetStream(cublas_handle, stream));

    complex_d* matrix_diagblk_d = NULL;
    complex_d* matrix_upperblk_d = NULL;
    complex_d* matrix_lowerblk_d = NULL;

    // load the whole matrix to device
    cudaErrchk(cudaMalloc((void**)&matrix_diagblk_d, blocksize * matrix_size * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&matrix_upperblk_d, blocksize * (off_diag_size) * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&matrix_lowerblk_d, blocksize * (off_diag_size) * sizeof(complex_d)));

    cudaErrchk(cudaMemcpy(matrix_diagblk_d, reinterpret_cast<const complex_d*>(matrix_diagblk_h),
                blocksize * matrix_size * sizeof(complex_d), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(matrix_upperblk_d, reinterpret_cast<const complex_d*>(matrix_upperblk_h),
                blocksize * (off_diag_size) * sizeof(complex_d), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(matrix_lowerblk_d, reinterpret_cast<const complex_d*>(matrix_lowerblk_h),
                blocksize * (off_diag_size) * sizeof(complex_d), cudaMemcpyHostToDevice));

    // allocate memory for the inverse
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

    // init right hand side identity matrix on device for backsub
    complex_d* identity_d = NULL;
    complex_d* identity_cpy_d = NULL;
    cudaErrchk(cudaMalloc((void**)&identity_d, blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&identity_cpy_d, blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMemcpy(identity_d, reinterpret_cast<const complex_d*>(identity_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(identity_cpy_d, identity_d,
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice));
    // ----- END OF INIT SECTION -----

    int info_h = 0;
    int *ipiv_d = NULL;
    int *info_d = NULL;
    cudaErrchk(cudaMalloc((void**)&info_d, sizeof(int)))
    cudaErrchk(cudaMalloc((void**)&ipiv_d, blocksize*sizeof(int)));

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
        std::printf("Error: LU factorization failed\n");
        success = false;
    }


    //back substitution
    cusolverErrchk(cusolverDnZgetrs(cusolver_handle, CUBLAS_OP_N, blocksize,
                                    blocksize, matrix_diagblk_d, blocksize, ipiv_d,
                                    identity_cpy_d, blocksize, info_d));
    cudaErrchk(cudaStreamSynchronize(stream));
    //copy info to host
    cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_h != 0) {
        std::printf("Error: Backsub failed\n");
        success = false;
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
        //MatMul schur complement = eig_diagblk[i] - tmp * eig_upperblk[i-1]
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
            std::printf("Error: LU factorization failed\n");
            std::printf("info_h = %d\n", info_h);
            success = false;
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
            std::printf("Error: Backsub failed\n");
            std::printf("info_h = %d\n", info_h);
            success = false;
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

    // load selected inverse to host
    cudaErrchk(cudaMemcpy(inv_diagblk_h, reinterpret_cast<const complex_h*>(inv_diagblk_d),
                blocksize * matrix_size * sizeof(complex_h), cudaMemcpyDeviceToHost));
    cudaErrchk(cudaMemcpy(inv_upperblk_h, reinterpret_cast<const complex_h*>(inv_upperblk_d),
                blocksize * off_diag_size * sizeof(complex_h), cudaMemcpyDeviceToHost));
    cudaErrchk(cudaMemcpy(inv_lowerblk_h, reinterpret_cast<const complex_h*>(inv_lowerblk_d),
                blocksize * off_diag_size * sizeof(complex_h), cudaMemcpyDeviceToHost));


    // deallocate device memory
    if (stream) {
        cudaErrchk(cudaStreamDestroy(stream));
    }
    if(cublas_handle) {
        cublasErrchk(cublasDestroy(cublas_handle));
    }
    if(cusolver_handle) {
        cusolverErrchk(cusolverDnDestroy(cusolver_handle));
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
    if(identity_d){
        cudaErrchk(cudaFree(identity_d));
    }
    if(identity_cpy_d){
        cudaErrchk(cudaFree(identity_cpy_d));
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
    return success;
}


bool rgf_dense_matrix_does_not_fit_gpu_memory(
    unsigned int blocksize,
    unsigned int matrix_size,
    complex_h *matrix_diagblk_h,
    complex_h *matrix_upperblk_h,
    complex_h *matrix_lowerblk_h,
    complex_h *inv_diagblk_h,
    complex_h *inv_upperblk_h,
    complex_h *inv_lowerblk_h)
{
    if(matrix_size % blocksize != 0){
        printf("Error: matrix_size is not a multiple of blocksize\n");
        return false;
    }
    unsigned int n_blocks = matrix_size / blocksize;
    bool success = true;

    // Init cuda stuff
    cudaStream_t stream = NULL;
    cudaErrchk(cudaStreamCreate(&stream));

    cusolverDnHandle_t cusolver_handle;
    cusolverErrchk(cusolverDnCreate(&cusolver_handle));
    cusolverErrchk(cusolverDnSetStream(cusolver_handle, stream));

    cublasHandle_t cublas_handle = 0;
    cublasErrchk(cublasCreate(&cublas_handle));
    cublasErrchk(cublasSetStream(cublas_handle, stream));

    // not allowed to load full matrix to device
    // allocate memory for the blocks
    
    complex_d* matrix_diagblk_d = NULL;
    complex_d* matrix_upperblk_d = NULL;
    complex_d* matrix_lowerblk_d = NULL;

    // allocate single blocks of the matrix
    cudaErrchk(cudaMalloc((void**)&matrix_diagblk_d, blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&matrix_upperblk_d, blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&matrix_lowerblk_d, blocksize * blocksize * sizeof(complex_d)));


    // allocate memory for the inverse
    complex_d* inv_diagblk_d = NULL;
    complex_d* inv_diagblk_small_d = NULL;
    complex_d* inv_upperblk_d = NULL;
    complex_d* inv_lowerblk_d = NULL;

    cudaErrchk(cudaMalloc((void**)&inv_diagblk_d, blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&inv_diagblk_small_d, blocksize * blocksize * sizeof(complex_d)));

    cudaErrchk(cudaMalloc((void**)&inv_upperblk_d, blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&inv_lowerblk_d, blocksize * blocksize * sizeof(complex_d)));

    // create right hand side identity matrix
    complex_h* identity_h = (complex_h*) malloc(blocksize * blocksize * sizeof(complex_h));

    for(unsigned int i = 0; i < blocksize * blocksize; i++){
        identity_h[i] = 0.0;
        if(i / blocksize == i % blocksize){
            identity_h[i] = 1.0;
        }
    }

    // init right hand side identity matrix on device for backsub
    complex_d* identity_d = NULL;
    complex_d* identity_cpy_d = NULL;
    cudaErrchk(cudaMalloc((void**)&identity_d, blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&identity_cpy_d, blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMemcpy(identity_d, reinterpret_cast<const complex_d*>(identity_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(identity_cpy_d, identity_d,
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice));
    // ----- END OF INIT SECTION -----


    cudaErrchk(cudaMemcpy(matrix_diagblk_d, reinterpret_cast<const complex_d*>(matrix_diagblk_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice));


    int info_h = 0;
    int *ipiv_d = NULL;
    int *info_d = NULL;
    cudaErrchk(cudaMalloc((void**)&info_d, sizeof(int)))
    cudaErrchk(cudaMalloc((void**)&ipiv_d, blocksize*sizeof(int)));

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
        std::printf("Error: LU factorization failed\n");
        success = false;
    }


    //back substitution
    cusolverErrchk(cusolverDnZgetrs(cusolver_handle, CUBLAS_OP_N, blocksize,
                                    blocksize, matrix_diagblk_d, blocksize, ipiv_d,
                                    identity_cpy_d, blocksize, info_d));
    cudaErrchk(cudaStreamSynchronize(stream));
    //copy info to host
    cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_h != 0) {
        std::printf("Error: Backsub failed\n");
        success = false;
    }


    // // 0. Inverse of the first block
    cudaErrchk(cudaMemcpy(inv_diagblk_d, identity_cpy_d,
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice));
    cudaErrchk(cudaMemcpy(inv_diagblk_h, inv_diagblk_d,
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost));



    complex_d alpha;
    complex_d beta;
    // // 1. Forward substitution (performed left to right)
    for (unsigned int i = 1; i < n_blocks; ++i) {


        cudaErrchk(cudaMemcpy(matrix_diagblk_d,
                    reinterpret_cast<const complex_d*>(matrix_diagblk_h + i*blocksize*blocksize),
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(matrix_upperblk_d,
                    reinterpret_cast<const complex_d*>(matrix_upperblk_h + (i-1)*blocksize*blocksize),
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(matrix_lowerblk_d,
                    reinterpret_cast<const complex_d*>(matrix_lowerblk_h  + (i-1)*blocksize*blocksize),
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice));


        // MatMul tmp = eig_lowerblk[i-1] * eig_inv_diagblk[i-1]
        // use the inv_diagblk_d from last iteration
        // use inv_lowerblk_d as tmp
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            matrix_lowerblk_d, blocksize,
            inv_diagblk_d, blocksize,
            &beta,
            inv_lowerblk_d, blocksize));
        //MatMul schur complement = eig_diagblk[i] - tmp * eig_upperblk[i-1]
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            inv_lowerblk_d, blocksize,
            matrix_upperblk_d, blocksize,
            &beta,
            matrix_diagblk_d, blocksize));

        //copy identity
        cudaErrchk(cudaMemcpy(identity_cpy_d, identity_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice));
        // invert schur complement
        cudaErrchk(cudaStreamSynchronize(stream));
        cusolverErrchk(cusolverDnZgetrf(cusolver_handle, blocksize, blocksize,
                                    matrix_diagblk_d,
                                    blocksize, buffer, ipiv_d, info_d));
        
        //copy info to host
        cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));

        if (info_h != 0) {
            std::printf("Error: LU factorization failed\n");
            std::printf("info_h = %d\n", info_h);
            success = false;
        }


        //back substitution
        cusolverErrchk(cusolverDnZgetrs(cusolver_handle, CUBLAS_OP_N, blocksize,
                                        blocksize,
                                        matrix_diagblk_d,
                                        blocksize, ipiv_d,
                                        identity_cpy_d, blocksize, info_d));
        cudaErrchk(cudaStreamSynchronize(stream));
        //copy info to host
        cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));

        if (info_h != 0) {
            std::printf("Error: Backsub failed\n");
            std::printf("info_h = %d\n", info_h);
            success = false;
        }

        cudaErrchk(cudaMemcpy(inv_diagblk_d,
                    identity_cpy_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice));
        cudaErrchk(cudaMemcpy(inv_diagblk_h + i*blocksize*blocksize,
                    inv_diagblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost));
    }


    // 2. Backward substitution (performed right to left)
    for(int i = n_blocks-2; i >= 0; --i){
        cudaErrchk(cudaMemcpy(matrix_upperblk_d,
                    reinterpret_cast<const complex_d*>(matrix_upperblk_h + i*blocksize*blocksize),
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(matrix_lowerblk_d,
                    reinterpret_cast<const complex_d*>(matrix_lowerblk_h + i*blocksize*blocksize),
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(inv_diagblk_small_d,
                    reinterpret_cast<const complex_d*>(inv_diagblk_h  + i*blocksize*blocksize),
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice));        

        //tmp = eig_inv_diagblk[i+1] * eig_lowerblk[i]
        // use identity_cpy_d as tmp
        // reuse inv_diagblk_d from last iteration
        // which is the last true inverse block
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            inv_diagblk_d, blocksize,
            matrix_lowerblk_d, blocksize,
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
            inv_diagblk_small_d, blocksize,
            &beta,
            inv_lowerblk_d, blocksize));

        cudaErrchk(cudaMemcpy(inv_lowerblk_h + i*blocksize*blocksize,
                    inv_lowerblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost));

        // tmp = eig_inv_diagblk[i] * eig_upperblk[i]
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            inv_diagblk_small_d, blocksize,
            matrix_upperblk_d, blocksize,
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
            inv_diagblk_d, blocksize,
            &beta,
            inv_upperblk_d, blocksize));

        cudaErrchk(cudaMemcpy(inv_upperblk_h + i*blocksize*blocksize,
                    inv_upperblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost));

        //eig_inv_diagblk[i] -= tmp * eig_inv_lowerblk[i];
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            identity_cpy_d, blocksize,
            inv_lowerblk_d, blocksize,
            &beta,
            inv_diagblk_small_d, blocksize));


        cudaErrchk(cudaMemcpy(inv_diagblk_d,
                    inv_diagblk_small_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice));
        cudaErrchk(cudaMemcpy(inv_diagblk_h + i*blocksize*blocksize,
                    inv_diagblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost));
    }


    // deallocate device memory
    if (stream) {
        cudaErrchk(cudaStreamDestroy(stream));
    }
    if(cublas_handle) {
        cublasErrchk(cublasDestroy(cublas_handle));
    }
    if(cusolver_handle) {
        cusolverErrchk(cusolverDnDestroy(cusolver_handle));
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
    if(identity_d){
        cudaErrchk(cudaFree(identity_d));
    }
    if(identity_cpy_d){
        cudaErrchk(cudaFree(identity_cpy_d));
    }
    if(inv_diagblk_small_d){
        cudaErrchk(cudaFree(inv_diagblk_small_d));
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
    return success;
}


bool rgf_dense_matrix_does_not_fit_gpu_memory_with_copy_compute_overlap(
    unsigned int blocksize,
    unsigned int matrix_size,
    complex_h *matrix_diagblk_h,
    complex_h *matrix_upperblk_h,
    complex_h *matrix_lowerblk_h,
    complex_h *inv_diagblk_h,
    complex_h *inv_upperblk_h,
    complex_h *inv_lowerblk_h)
{
    if(matrix_size % blocksize != 0){
        printf("Error: matrix_size is not a multiple of blocksize\n");
        return false;
    }
    unsigned int n_blocks = matrix_size / blocksize;
    bool success = true;

    // Init cuda stuff

    // need multiple streams for overlap
    int number_streams = 3;
    cudaStream_t stream[number_streams];
    for(int i = 0; i < number_streams; i++){
        cudaErrchk(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));
    }

    cusolverDnHandle_t cusolver_handle[number_streams];
    for(int i = 0; i < number_streams; i++){
        cusolverErrchk(cusolverDnCreate(&cusolver_handle[i]));
        cusolverErrchk(cusolverDnSetStream(cusolver_handle[i], stream[i]));
    }

    cublasHandle_t cublas_handle[number_streams];
    for(int i = 0; i < number_streams; i++){
        cublasErrchk(cublasCreate(&cublas_handle[i]));
        cublasErrchk(cublasSetStream(cublas_handle[i], stream[i]));
    }

    cudaEvent_t schur_inverted[n_blocks];
    for(unsigned int i = 0; i < n_blocks; i++){
        cudaErrchk(cudaEventCreate(&schur_inverted[i]))
    }
    cudaEvent_t unload[n_blocks];
    for(unsigned int i = 0; i < n_blocks; i++){
        cudaErrchk(cudaEventCreate(&unload[i]))
    }
    complex_d alpha;
    complex_d beta;
    int stream_memload = 1;
    int stream_compute = 0;
    int stream_memunload = 2;

    // not allowed to load full matrix to device
    // allocate memory for the blocks
    
    complex_d* matrix_diagblk_d[2];
    complex_d* matrix_upperblk_d[2];
    complex_d* matrix_lowerblk_d[2];

    // allocate single blocks of the matrix
    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&matrix_diagblk_d[i], blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&matrix_upperblk_d[i], blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&matrix_lowerblk_d[i], blocksize * blocksize * sizeof(complex_d)));
    }



    // allocate memory for the inverse
    complex_d* inv_diagblk_d = NULL;
    complex_d* inv_diagblk_small_d[2];
    complex_d* inv_upperblk_d = NULL;
    complex_d* inv_lowerblk_d = NULL;

    cudaErrchk(cudaMalloc((void**)&inv_diagblk_d, blocksize * blocksize * sizeof(complex_d)));
    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&inv_diagblk_small_d[i], blocksize * blocksize * sizeof(complex_d)));
    }
    
    cudaErrchk(cudaMalloc((void**)&inv_upperblk_d, blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&inv_lowerblk_d, blocksize * blocksize * sizeof(complex_d)));

    //memory for pivoting
    int *ipiv_d = NULL;
    int *info_d = NULL;
    cudaErrchk(cudaMalloc((void**)&info_d, sizeof(int)))
    cudaErrchk(cudaMalloc((void**)&ipiv_d, blocksize*sizeof(int)));


    // create right hand side identity matrix
    complex_h* identity_h = (complex_h*) malloc(blocksize * blocksize * sizeof(complex_h));
    complex_d* identity_d = NULL;
    complex_d* identity_cpy_d = NULL;
    cudaErrchk(cudaMalloc((void**)&identity_d, blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&identity_cpy_d, blocksize * blocksize * sizeof(complex_d)));


    for(unsigned int i = 0; i < blocksize * blocksize; i++){
        identity_h[i] = 0.0;
        if(i / blocksize == i % blocksize){
            identity_h[i] = 1.0;
        }
    }


    //figure out extra amount of memory needed
    complex_d *buffer = NULL;
    int bufferSize = 0;
    cusolverErrchk(cusolverDnZgetrf_bufferSize(cusolver_handle[stream_compute], blocksize, blocksize,
                                            (complex_d *)matrix_diagblk_d[stream_compute],
                                              blocksize, &bufferSize));
    cudaErrchk(cudaMalloc((void**)&buffer, sizeof(complex_d) * bufferSize));

    // ----- END OF INIT SECTION -----

    // init right hand side identity matrix on device for backsub
    cudaErrchk(cudaMemcpyAsync(identity_d, reinterpret_cast<const complex_d*>(identity_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_compute]));
    cudaErrchk(cudaMemcpyAsync(inv_diagblk_d, identity_d,
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
    

    cudaErrchk(cudaMemcpyAsync(matrix_diagblk_d[stream_compute], reinterpret_cast<const complex_d*>(matrix_diagblk_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_compute]));


    cusolverErrchk(cusolverDnZgetrf(cusolver_handle[stream_compute], blocksize, blocksize,
                                matrix_diagblk_d[stream_compute], blocksize, buffer, ipiv_d, info_d));
    

    //back substitution
    cusolverErrchk(cusolverDnZgetrs(cusolver_handle[stream_compute], CUBLAS_OP_N, blocksize,
                                    blocksize, matrix_diagblk_d[stream_compute], blocksize, ipiv_d,
                                    inv_diagblk_d, blocksize, info_d));

    // record finishing the inverse of the first block
    cudaErrchk(cudaEventRecord(schur_inverted[0], stream[stream_compute]));


    //wait for the inverse of the first block
    cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[0]));
    // 0. Inverse of the first block
    cudaErrchk(cudaMemcpyAsync(inv_diagblk_h, inv_diagblk_d,
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
    // unloading finished
    cudaErrchk(cudaEventRecord(unload[0], stream[stream_memunload]));


    // first memcpy happens before loop
    cudaErrchk(cudaMemcpyAsync(matrix_diagblk_d[1],
                reinterpret_cast<const complex_d*>(matrix_diagblk_h + blocksize*blocksize),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(matrix_upperblk_d[1],
                reinterpret_cast<const complex_d*>(matrix_upperblk_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(matrix_lowerblk_d[1],
                reinterpret_cast<const complex_d*>(matrix_lowerblk_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));




    // // 1. Forward substitution (performed left to right)
    for (unsigned int i = 1; i < n_blocks; ++i) {


        int stream_memload = (i+1) % 2;
        int stream_compute = i % 2;
        int stream_memunload = 2;


        if(i < n_blocks+1){
            // load the blocks for the next iteration
            cudaErrchk(cudaMemcpyAsync(matrix_diagblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(matrix_diagblk_h + (i+1)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(matrix_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(matrix_upperblk_h + (i)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(matrix_lowerblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(matrix_lowerblk_h  + (i)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
        }

        //wait for the schur inverse from the previous iteration
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], schur_inverted[i-1]));

        // MatMul tmp = eig_lowerblk[i-1] * eig_inv_diagblk[i-1]
        // use the inv_diagblk_d from last iteration
        // use inv_lowerblk_d as tmp
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            matrix_lowerblk_d[stream_compute], blocksize,
            inv_diagblk_d, blocksize,
            &beta,
            inv_lowerblk_d, blocksize));
        //MatMul schur complement = eig_diagblk[i] - tmp * eig_upperblk[i-1]
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);

        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            inv_lowerblk_d, blocksize,
            matrix_upperblk_d[stream_compute], blocksize,
            &beta,
            matrix_diagblk_d[stream_compute], blocksize));

        // wait to not overwrite block to unload
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], unload[i-1]));
        //copy identity
        cudaErrchk(cudaMemcpyAsync(inv_diagblk_d, identity_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        // inverse schur complement
        cusolverErrchk(cusolverDnZgetrf(cusolver_handle[stream_compute], blocksize, blocksize,
                                    matrix_diagblk_d[stream_compute],
                                    blocksize, buffer, ipiv_d, info_d));
        

        //back substitution
        cusolverErrchk(cusolverDnZgetrs(cusolver_handle[stream_compute], CUBLAS_OP_N, blocksize,
                                        blocksize,
                                        matrix_diagblk_d[stream_compute],
                                        blocksize, ipiv_d,
                                        inv_diagblk_d, blocksize, info_d));
        // record finishing of computation in step i
        cudaErrchk(cudaEventRecord(schur_inverted[i], stream[stream_compute]));

        // wait to unload for the finish of computations
        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[i]));
        cudaErrchk(cudaMemcpyAsync(inv_diagblk_h + i*blocksize*blocksize,
                    inv_diagblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        // unloading finished
        cudaErrchk(cudaEventRecord(unload[i], stream[stream_memunload]));

    }

    int stream_memload_before = (n_blocks) % 2;
    int stream_compute_before = (n_blocks-1) % 2;

    cudaErrchk(cudaMemcpyAsync(matrix_upperblk_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(matrix_upperblk_h + (n_blocks-2)*blocksize*blocksize),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    cudaErrchk(cudaMemcpyAsync(matrix_lowerblk_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(matrix_lowerblk_h + (n_blocks-2)*blocksize*blocksize),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    // possible race condition with unloading of previous loop
    // not sure
    cudaErrchk(cudaMemcpyAsync(inv_diagblk_small_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(inv_diagblk_h  + (n_blocks-2)*blocksize*blocksize),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
  

    // 2. Backward substitution (performed right to left)
    for(int i = n_blocks-2; i >= 0; --i){

        // fix stream compute to be the stream which loaded
        // blocks before the loop
        stream_memload = (stream_compute_before + (i - n_blocks + 2) ) % 2;
        stream_compute = (stream_memload_before + (i - n_blocks + 2) ) % 2;
        stream_memunload = 2;

        if(i > 0){
            cudaErrchk(cudaMemcpyAsync(matrix_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(matrix_upperblk_h + (i-1)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(matrix_lowerblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(matrix_lowerblk_h + (i-1)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(inv_diagblk_small_d[stream_memload],
                        reinterpret_cast<const complex_d*>(inv_diagblk_h  + (i-1)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));    
        }
    

        // wait for the block of the last iteration
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], schur_inverted[i+1]));

        //tmp = eig_inv_diagblk[i+1] * eig_lowerblk[i]
        // use identity_cpy_d as tmp
        // reuse inv_diagblk_d from last iteration
        // which is the last true inverse block
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            inv_diagblk_d, blocksize,
            matrix_lowerblk_d[stream_compute], blocksize,
            &beta,
            identity_cpy_d, blocksize));

        // eig_inv_lowerblk[i] = -tmp*eig_inv_diagblk[i];
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        
        // wait to not overwrite blocks to unload
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], unload[i+1]));   
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            identity_cpy_d, blocksize,
            inv_diagblk_small_d[stream_compute], blocksize,
            &beta,
            inv_lowerblk_d, blocksize));

        // tmp = eig_inv_diagblk[i] * eig_upperblk[i]
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            inv_diagblk_small_d[stream_compute], blocksize,
            matrix_upperblk_d[stream_compute], blocksize,
            &beta,
            identity_cpy_d, blocksize));

        //eig_inv_upperblk[i] = -tmp * eig_inv_diagblk[i+1];
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            identity_cpy_d, blocksize,
            inv_diagblk_d, blocksize,
            &beta,
            inv_upperblk_d, blocksize));


        //eig_inv_diagblk[i] -= tmp * eig_inv_lowerblk[i];
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);

        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            identity_cpy_d, blocksize,
            inv_lowerblk_d, blocksize,
            &beta,
            inv_diagblk_small_d[stream_compute], blocksize));
 
        cudaErrchk(cudaMemcpyAsync(inv_diagblk_d,
                    inv_diagblk_small_d[stream_compute],
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        cudaErrchk(cudaEventRecord(schur_inverted[i], stream[stream_compute]));

        // wait to unload for the finish of computations
        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[i]));
        cudaErrchk(cudaMemcpyAsync(inv_diagblk_h + i*blocksize*blocksize,
                    inv_diagblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        cudaErrchk(cudaMemcpyAsync(inv_upperblk_h + i*blocksize*blocksize,
                    inv_upperblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        cudaErrchk(cudaMemcpyAsync(inv_lowerblk_h + i*blocksize*blocksize,
                    inv_lowerblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        // unloading finished
        cudaErrchk(cudaEventRecord(unload[i], stream[stream_memunload]));
    }

    // synchronize all the streams
    for(int j = 0; j < number_streams; j++){
        cudaErrchk(cudaStreamSynchronize(stream[j]));
    }
    // deallocate device memory
    for(int i = 0; i < number_streams; i++){
        if (stream[i]) {
            cudaErrchk(cudaStreamDestroy(stream[i]));
        }
        if(cublas_handle[i]) {
            cublasErrchk(cublasDestroy(cublas_handle[i]));
        }
        if(cusolver_handle[i]) {
            cusolverErrchk(cusolverDnDestroy(cusolver_handle[i]));
        }
    }
    for(int i = 0; i < 2; i++){
        if(matrix_diagblk_d[i]) {
            cudaErrchk(cudaFree(matrix_diagblk_d[i]));
        }
        if(matrix_upperblk_d[i]) {
            cudaErrchk(cudaFree(matrix_upperblk_d[i]));
        }
        if(matrix_lowerblk_d[i]) {
            cudaErrchk(cudaFree(matrix_lowerblk_d[i]));
        }
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
    if(identity_d){
        cudaErrchk(cudaFree(identity_d));
    }
    if(identity_cpy_d){
        cudaErrchk(cudaFree(identity_cpy_d));
    }

    for(int i = 0; i < 2; i++){
        if(inv_diagblk_small_d[i]){
            cudaErrchk(cudaFree(inv_diagblk_small_d[i]));
        }        
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
    for(unsigned int i = 0; i < n_blocks; i++){
        if(schur_inverted[i]){
            cudaErrchk(cudaEventDestroy(schur_inverted[i]));
        }        
    }
    for(unsigned int i = 0; i < n_blocks; i++){
        if(unload[i]){
            cudaErrchk(cudaEventDestroy(unload[i]));
        }
    }

    return success;
}


int main() {

    if (cudaSetDevice(0) != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device.");
    }

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
    printf("    Number of blocks: %d\n", n_blocks);


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

    matrix_diagblk_h = [A_0;
                           A_1;
                           ...;
                           A_n]
    matrix_upperblk_h = [B_0;
                            B_1;
                            ...;
                            B_n-1]
    matrix_lowerblk_h = [C_0;
                            C_1;
                            ...;
                            C_n-1]

    where blocks are in column major order
    */


    complex_h* matrix_diagblk_h = NULL;
    complex_h* matrix_upperblk_h = NULL;
    complex_h* matrix_lowerblk_h = NULL;
    cudaMallocHost((void**)&matrix_diagblk_h, blocksize * matrix_size * sizeof(complex_h));
    cudaMallocHost((void**)&matrix_upperblk_h, blocksize * off_diag_size * sizeof(complex_h));
    cudaMallocHost((void**)&matrix_lowerblk_h, blocksize * off_diag_size * sizeof(complex_h));

    for(unsigned int i = 0; i < blocksize * matrix_size; i++){
        // block index
        int k = i / (blocksize * blocksize);
        // index inside block
        int h = i % (blocksize * blocksize);
        // row inside block
        int m = h % blocksize;
        // col inside block
        int n = h / blocksize;
        matrix_diagblk_h[i] = matrix_diagblk[m*matrix_size + k*blocksize + n];
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
        matrix_upperblk_h[i] = matrix_upperblk[m*off_diag_size + k*blocksize + n];
        matrix_lowerblk_h[i] = matrix_lowerblk[m*off_diag_size + k*blocksize + n];
    }
    // for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
    //     std::cout << "matrix_upperblk_h[" << i << "] = " << matrix_upperblk_h[i] << std::endl;
    // }

    // allocate memory for the inverse
    complex_h* inv_diagblk_h = NULL;
    complex_h* inv_upperblk_h = NULL;
    complex_h* inv_lowerblk_h = NULL;

    cudaMallocHost((void**)&inv_diagblk_h, blocksize * matrix_size * sizeof(complex_h));
    cudaMallocHost((void**)&inv_upperblk_h, blocksize * off_diag_size * sizeof(complex_h));
    cudaMallocHost((void**)&inv_lowerblk_h, blocksize * off_diag_size * sizeof(complex_h));

    // if(!(rgf_matrix_fits_gpu_memory(blocksize, matrix_size,
    //                                 matrix_diagblk_h,
    //                                 matrix_upperblk_h,
    //                                 matrix_lowerblk_h,
    //                                 inv_diagblk_h,
    //                                 inv_upperblk_h,
    //                                 inv_lowerblk_h))){
    //     printf("Error: rgf_matrix_fits_gpu_memory failed\n");
    // }
    // else{
    //     printf("rgf_matrix_fits_gpu_memory succeeded\n");
    // }

    // if(!rgf_dense_matrix_does_not_fit_gpu_memory(blocksize, matrix_size,
    //                                 matrix_diagblk_h,
    //                                 matrix_upperblk_h,
    //                                 matrix_lowerblk_h,
    //                                 inv_diagblk_h,
    //                                 inv_upperblk_h,
    //                                 inv_lowerblk_h)){
    //     printf("Error: rgf_dense_matrix_does_not_fit_gpu_memory failed\n");
    // }
    // else{
    //     printf("rgf_dense_matrix_does_not_fit_gpu_memory succeeded\n");
    // }
    if(!rgf_dense_matrix_does_not_fit_gpu_memory_with_copy_compute_overlap(blocksize, matrix_size,
                                    matrix_diagblk_h,
                                    matrix_upperblk_h,
                                    matrix_lowerblk_h,
                                    inv_diagblk_h,
                                    inv_upperblk_h,
                                    inv_lowerblk_h)){
        printf("Error: rgf_dense_matrix_does_not_fit_gpu_memory_with_copy_compute_overlap failed\n");
    }
    else{
        printf("rgf_dense_matrix_does_not_fit_gpu_memory_with_copy_compute_overlap succeeded\n");
    }

    // ----- RESULT CHECKING SECTION -----

    // Load reference solution of the matrix inverse
    std::complex<double>* matrix_inv_diagblk_ref = (std::complex<double>*) malloc(blocksize * matrix_size * sizeof(std::complex<double>));
    char f_mat_inv_diagblk[] = "../../../tests/tests_cases/dense_blocks_matrix_0_inverse_diagblk.bin";
    load_binary_matrix(f_mat_inv_diagblk, matrix_inv_diagblk_ref, blocksize, matrix_size);

    std::complex<double>* matrix_inv_upperblk_ref = (std::complex<double>*) malloc(blocksize * (off_diag_size) * sizeof(std::complex<double>));
    char f_mat_inv_upperblk[] = "../../../tests/tests_cases/dense_blocks_matrix_0_inverse_upperblk.bin";
    load_binary_matrix(f_mat_inv_upperblk, matrix_inv_upperblk_ref, blocksize, off_diag_size);
    
    std::complex<double>* matrix_inv_lowerblk_ref = (std::complex<double>*) malloc(blocksize * (off_diag_size) * sizeof(std::complex<double>));
    char f_mat_inv_lowerblk[] = "../../../tests/tests_cases/dense_blocks_matrix_0_inverse_lowerblk.bin";
    load_binary_matrix(f_mat_inv_lowerblk, matrix_inv_lowerblk_ref, blocksize, off_diag_size);


    // Transform the reference solution to contiguous blocks where the blocks have column-major order
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
        inv_diagblk_ref[i] = matrix_inv_diagblk_ref[m*matrix_size + k*blocksize + n];
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
        inv_upperblk_ref[i] = matrix_inv_upperblk_ref[m*off_diag_size + k*blocksize + n];
        inv_lowerblk_ref[i] = matrix_inv_lowerblk_ref[m*off_diag_size + k*blocksize + n];
    }

    // // print last block of inverted matrix
    // for(unsigned int i = blocksize *(matrix_size-blocksize); i < blocksize * matrix_size; i++){
    //     std::cout << "inv_diagblk_h[" << i << "] = " << inv_diagblk_h[i] << std::endl;
    //     std::cout << "inv_diagblk_ref[" << i << "] = " << inv_diagblk_ref[i] << std::endl;
    // }

    double norm_diagblk = 0.0;
    double norm_upperblk = 0.0;
    double norm_lowerblk = 0.0;
    double diff_diagblk = 0.0;
    double diff_upperblk = 0.0;
    double diff_lowerblk = 0.0;
    for(unsigned int i = 0; i < blocksize * matrix_size; i++){
        norm_diagblk += std::abs(inv_diagblk_ref[i]);
        diff_diagblk += std::abs(inv_diagblk_h[i] - inv_diagblk_ref[i]);
    }
    for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
        norm_upperblk += std::abs(inv_upperblk_ref[i]);
        norm_lowerblk += std::abs(inv_lowerblk_ref[i]);
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


    if(inv_diagblk_h){
        cudaFreeHost(inv_diagblk_h);
    }
    if(inv_upperblk_h){
        cudaFreeHost(inv_upperblk_h);
    }
    if(inv_lowerblk_h){
        cudaFreeHost(inv_lowerblk_h);
    }
    if(matrix_diagblk_h){
        cudaFreeHost(matrix_diagblk_h);
    }
    if(matrix_upperblk_h){
        cudaFreeHost(matrix_upperblk_h);
    }
    if(matrix_lowerblk_h){
        cudaFreeHost(matrix_lowerblk_h);
    }
    if(matrix_diagblk){
        free(matrix_diagblk);
    }
    if(matrix_upperblk){
        free(matrix_upperblk);
    }
    if(matrix_lowerblk){
        free(matrix_lowerblk);
    }
    if(matrix_inv_diagblk_ref){
        free(matrix_inv_diagblk_ref);
    }
    if(matrix_inv_upperblk_ref){
        free(matrix_inv_upperblk_ref);
    }
    if(matrix_inv_lowerblk_ref){
        free(matrix_inv_lowerblk_ref);
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



    return 0;
}








