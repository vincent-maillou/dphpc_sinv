// Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.

#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <iostream>
#include <string>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>
#include <cuda/std/complex>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include "dense_rgf.h"
#include <cuda_runtime.h>
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
        std::printf("CUSOLVERassert: %s %s %d\n", cudaGetErrorString((cudaError_t)code), file, line);
        if (abort) exit(code);
   }
}


#define cublasErrchk(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUBLAS_STATUS_SUCCESS) 
   {
        //Did not find a counter part to cudaGetErrorString in cublas
        std::printf("CUBLASassert: %s %s %d\n", cudaGetErrorString((cudaError_t)code), file, line);
        if (abort) exit(code);
   }
}

// both should be equivalent, thus reinterpret_cast should be fine
using complex_h = std::complex<double>;
using complex_d = cuDoubleComplex; //cuda::complex_h;



void rgf_multiple_energy_points_for_loop(
    unsigned int blocksize,
    unsigned int matrix_size,
    unsigned int batch_size,
    complex_h **batch_diagblk_h,
    complex_h **batch_upperblk_h,
    complex_h **batch_lowerblk_h,
    complex_h **batch_inv_diagblk_h,
    complex_h **batch_inv_upperblk_h,
    complex_h **batch_inv_lowerblk_h)
{

    
    if(matrix_size % blocksize != 0){
        printf("Error: matrix_size is not a multiple of blocksize\n");
    }
    unsigned int n_blocks = matrix_size / blocksize;

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
    complex_h* identity_h;
    cudaErrchk(cudaMallocHost((void**)&identity_h, blocksize * blocksize * sizeof(complex_h)));
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
    cudaErrchk(cudaMemcpyAsync(identity_d, reinterpret_cast<const complex_d*>(identity_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_compute]));

    //figure out extra amount of memory needed
    complex_d *buffer = NULL;
    int bufferSize = 0;
    cusolverErrchk(cusolverDnZgetrf_bufferSize(cusolver_handle[stream_compute], blocksize, blocksize,
                                            (complex_d *)matrix_diagblk_d[stream_compute],
                                              blocksize, &bufferSize));
    cudaErrchk(cudaMalloc((void**)&buffer, sizeof(complex_d) * bufferSize));



    // ----- END OF INIT SECTION -----

    for(unsigned int batch = 0; batch < batch_size; batch++){

        // init right hand side identity matrix on device for backsubstitution
        cudaErrchk(cudaMemcpyAsync(inv_diagblk_d, identity_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        

        cudaErrchk(cudaMemcpyAsync(matrix_diagblk_d[stream_compute],
                    reinterpret_cast<const complex_d*>(batch_diagblk_h[0] + batch*blocksize*blocksize),
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
        cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_h[0] + batch*blocksize*blocksize, inv_diagblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        // unloading finished
        cudaErrchk(cudaEventRecord(unload[0], stream[stream_memunload]));


        // first memcpy happens before loop
        cudaErrchk(cudaMemcpyAsync(matrix_diagblk_d[1],
                    reinterpret_cast<const complex_d*>(batch_diagblk_h[1] + batch*blocksize*blocksize),
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
        cudaErrchk(cudaMemcpyAsync(matrix_upperblk_d[1],
                    reinterpret_cast<const complex_d*>(batch_upperblk_h[0] + batch*blocksize*blocksize),
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
        cudaErrchk(cudaMemcpyAsync(matrix_lowerblk_d[1],
                    reinterpret_cast<const complex_d*>(batch_lowerblk_h[0] + batch*blocksize*blocksize),
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));



        // // 1. Forward substitution (performed left to right)
        for (unsigned int i = 1; i < n_blocks; ++i) {


            int stream_memload = (i+1) % 2;
            int stream_compute = i % 2;
            int stream_memunload = 2;


            if(i < n_blocks-1){
                // load the blocks for the next iteration
                cudaErrchk(cudaMemcpyAsync(matrix_diagblk_d[stream_memload],
                            reinterpret_cast<const complex_d*>(batch_diagblk_h[i+1] + batch*blocksize*blocksize),
                            blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
                cudaErrchk(cudaMemcpyAsync(matrix_upperblk_d[stream_memload],
                            reinterpret_cast<const complex_d*>(batch_upperblk_h[i] + batch*blocksize*blocksize),
                            blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
                cudaErrchk(cudaMemcpyAsync(matrix_lowerblk_d[stream_memload],
                            reinterpret_cast<const complex_d*>(batch_lowerblk_h[i] + batch*blocksize*blocksize),
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
            cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_h[i] + batch*blocksize*blocksize,
                        inv_diagblk_d,
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
            // unloading finished
            cudaErrchk(cudaEventRecord(unload[i], stream[stream_memunload]));

        }
        int stream_memload_before = (n_blocks) % 2;
        int stream_compute_before = (n_blocks-1) % 2;

        cudaErrchk(cudaMemcpyAsync(matrix_upperblk_d[stream_memload_before],
                    reinterpret_cast<const complex_d*>(batch_upperblk_h[n_blocks-2] + batch*blocksize*blocksize),
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
        cudaErrchk(cudaMemcpyAsync(matrix_lowerblk_d[stream_memload_before],
                    reinterpret_cast<const complex_d*>(batch_lowerblk_h[n_blocks-2] + batch*blocksize*blocksize),
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
        // possible race condition with unloading of previous loop
        // not sure
        cudaErrchk(cudaMemcpyAsync(inv_diagblk_small_d[stream_memload_before],
                    reinterpret_cast<const complex_d*>(batch_inv_diagblk_h[n_blocks-2] + batch*blocksize*blocksize),
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    

        // TODO possible to save memory by allocating and freeing
        // memory which is not needed anymore (to reduce max memory consumption at one point)


        // 2. Backward substitution (performed right to left)
        for(int i = n_blocks-2; i >= 0; --i){

            // fix stream compute to be the stream which loaded
            // blocks before the loop
            stream_memload = (stream_compute_before + (i - n_blocks + 2) ) % 2;
            stream_compute = (stream_memload_before + (i - n_blocks + 2) ) % 2;
            stream_memunload = 2;

            if(i > 0){
                cudaErrchk(cudaMemcpyAsync(matrix_upperblk_d[stream_memload],
                            reinterpret_cast<const complex_d*>(batch_upperblk_h[i-1] + batch*blocksize*blocksize),
                            blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
                cudaErrchk(cudaMemcpyAsync(matrix_lowerblk_d[stream_memload],
                            reinterpret_cast<const complex_d*>(batch_lowerblk_h[i-1] + batch*blocksize*blocksize),
                            blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
                cudaErrchk(cudaMemcpyAsync(inv_diagblk_small_d[stream_memload],
                            reinterpret_cast<const complex_d*>(batch_inv_diagblk_h[i-1] + batch*blocksize*blocksize),
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
            

            // use temporary buffer for inv_lowerblk_d
            cublasErrchk(cublasZgemm(
                cublas_handle[stream_compute],
                CUBLAS_OP_N, CUBLAS_OP_N,
                blocksize, blocksize, blocksize,
                &alpha,
                identity_cpy_d, blocksize,
                inv_diagblk_small_d[stream_compute], blocksize,
                &beta,
                matrix_diagblk_d[1], blocksize));

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

            // use temporary buffer for inv_upperblk_d
            cublasErrchk(cublasZgemm(
                cublas_handle[stream_compute],
                CUBLAS_OP_N, CUBLAS_OP_N,
                blocksize, blocksize, blocksize,
                &alpha,
                identity_cpy_d, blocksize,
                inv_diagblk_d, blocksize,
                &beta,
                matrix_diagblk_d[0], blocksize));


            //eig_inv_diagblk[i] -= tmp * eig_inv_lowerblk[i];
            alpha = make_cuDoubleComplex(-1.0, 0.0);
            beta = make_cuDoubleComplex(1.0, 0.0);

            cublasErrchk(cublasZgemm(
                cublas_handle[stream_compute],
                CUBLAS_OP_N, CUBLAS_OP_N,
                blocksize, blocksize, blocksize,
                &alpha,
                identity_cpy_d, blocksize,
                matrix_diagblk_d[1], blocksize,
                &beta,
                inv_diagblk_small_d[stream_compute], blocksize));
    

            // wait to not overwrite blocks to unload
            cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], unload[i+1]));

            // use allocated buffers for inv
            // since host2host is cheap
            // matrix_diagblk_d is only used in forward pass
            cudaErrchk(cudaMemcpyAsync(inv_diagblk_d,
                        inv_diagblk_small_d[stream_compute],
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
            cudaErrchk(cudaMemcpyAsync(inv_upperblk_d,
                        matrix_diagblk_d[0],
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
            cudaErrchk(cudaMemcpyAsync(inv_lowerblk_d,
                        matrix_diagblk_d[1],
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
            cudaErrchk(cudaEventRecord(schur_inverted[i], stream[stream_compute]));

            // wait to unload for the finish of computations
            cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[i]));

            cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_h[i] + batch*blocksize*blocksize,
                        inv_diagblk_d,
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
            cudaErrchk(cudaMemcpyAsync(batch_inv_upperblk_h[i] + batch*blocksize*blocksize,
                        inv_upperblk_d,
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
            cudaErrchk(cudaMemcpyAsync(batch_inv_lowerblk_h[i] + batch*blocksize*blocksize,
                        inv_lowerblk_d,
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
            // unloading finished
            cudaErrchk(cudaEventRecord(unload[i], stream[stream_memunload]));
        }
        // synchronize all the streams
        for(int j = 0; j < number_streams; j++){
            cudaErrchk(cudaStreamSynchronize(stream[j]));
        }
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

}

void rgf_batched(
    unsigned int blocksize,
    unsigned int matrix_size,
    unsigned int batch_size,
    complex_h **batch_diagblk_h,
    complex_h **batch_upperblk_h,
    complex_h **batch_lowerblk_h,
    complex_h **batch_inv_diagblk_h,
    complex_h **batch_inv_upperblk_h,
    complex_h **batch_inv_lowerblk_h)
{


    if(matrix_size % blocksize != 0){
        printf("Error: matrix_size is not a multiple of blocksize\n");
        return;
    }
    unsigned int n_blocks = matrix_size / blocksize;

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
    complex_d* batch_diagblk_d[2];
    complex_d* batch_upperblk_d[2];
    complex_d* batch_lowerblk_d[2];

    complex_d* batch_diagblk_ptr_h[2][batch_size];
    complex_d* batch_upperblk_ptr_h[2][batch_size];
    complex_d* batch_lowerblk_ptr_h[2][batch_size];
    complex_d** batch_diagblk_ptr_d[2];
    complex_d** batch_upperblk_ptr_d[2];
    complex_d** batch_lowerblk_ptr_d[2];


    // allocate single blocks of the matrix
    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&batch_diagblk_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&batch_upperblk_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&batch_lowerblk_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
        for(unsigned int j = 0; j < batch_size; j++){
            batch_diagblk_ptr_h[i][j] = batch_diagblk_d[i] + j * blocksize * blocksize;
            batch_upperblk_ptr_h[i][j] = batch_upperblk_d[i] + j * blocksize * blocksize;
            batch_lowerblk_ptr_h[i][j] = batch_lowerblk_d[i] + j * blocksize * blocksize;
        }
    
    }
    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&batch_diagblk_ptr_d[i], batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMalloc((void**)&batch_upperblk_ptr_d[i], batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMalloc((void**)&batch_lowerblk_ptr_d[i], batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMemcpy(batch_diagblk_ptr_d[i], batch_diagblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(batch_upperblk_ptr_d[i], batch_upperblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(batch_lowerblk_ptr_d[i], batch_lowerblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    }



    // allocate memory for the inverse
    complex_d* batch_inv_diagblk_d = NULL;
    complex_d* batch_inv_upperblk_d = NULL;
    complex_d* batch_inv_lowerblk_d = NULL;
    complex_d* intermediate_d = NULL;

    complex_d* batch_inv_diagblk_ptr_h[batch_size];
    complex_d* batch_inv_lowerblk_ptr_h[batch_size];
    complex_d* intermediate_ptr_h[batch_size];

    complex_d** batch_inv_diagblk_ptr_d;
    complex_d** batch_inv_lowerblk_ptr_d;
    complex_d** intermediate_ptr_d;

    // used to for the small g in the forward pass
    complex_d* batch_inv_diagblk_small_d[2];
    complex_d* batch_inv_diagblk_small_ptr_h[2][batch_size];
    complex_d** batch_inv_diagblk_small_ptr_d[2];


    cudaErrchk(cudaMalloc((void**)&batch_inv_diagblk_d, batch_size * blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&batch_inv_upperblk_d, batch_size * blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&batch_inv_lowerblk_d, batch_size * blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&intermediate_d, batch_size * blocksize * blocksize * sizeof(complex_d)));

    for(unsigned int i = 0; i < batch_size; i++){
        batch_inv_diagblk_ptr_h[i] = batch_inv_diagblk_d + i * blocksize * blocksize;
        batch_inv_lowerblk_ptr_h[i] = batch_inv_lowerblk_d + i * blocksize * blocksize;
        intermediate_ptr_h[i] = intermediate_d + i * blocksize * blocksize;
    }
    cudaErrchk(cudaMalloc((void**)&batch_inv_diagblk_ptr_d, batch_size * sizeof(complex_d*)));
    cudaErrchk(cudaMalloc((void**)&batch_inv_lowerblk_ptr_d, batch_size * sizeof(complex_d*)));
    cudaErrchk(cudaMalloc((void**)&intermediate_ptr_d, batch_size * sizeof(complex_d*)));
    cudaErrchk(cudaMemcpy(batch_inv_diagblk_ptr_d, batch_inv_diagblk_ptr_h, batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(batch_inv_lowerblk_ptr_d, batch_inv_lowerblk_ptr_h, batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(intermediate_ptr_d, intermediate_ptr_h, batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));



    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&batch_inv_diagblk_small_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
        for(unsigned int j = 0; j < batch_size; j++){
            batch_inv_diagblk_small_ptr_h[i][j] = batch_inv_diagblk_small_d[i] + j * blocksize * blocksize;
        }
    }
    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&batch_inv_diagblk_small_ptr_d[i], batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMemcpy(batch_inv_diagblk_small_ptr_d[i], batch_inv_diagblk_small_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    }

    //memory for pivoting

    int *info_d = NULL;
    cudaErrchk(cudaMalloc((void**)&info_d, batch_size * sizeof(int)))

    int *ipiv_d = NULL;
    cudaErrchk(cudaMalloc((void**)&ipiv_d, batch_size * blocksize * sizeof(int)));

    // ----- END OF INIT SECTION -----


    cudaErrchk(cudaMemcpyAsync(batch_diagblk_d[stream_compute], reinterpret_cast<const complex_d*>(batch_diagblk_h[0]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_compute]));


    cublasErrchk(cublasZgetrfBatched(
            cublas_handle[stream_compute],
            blocksize,
            batch_diagblk_ptr_d[stream_compute], blocksize, ipiv_d,
            info_d, batch_size));


    // inversion
    cublasErrchk(cublasZgetriBatched(
                                cublas_handle[stream_compute],
                                blocksize,
                                batch_diagblk_ptr_d[stream_compute],
                                blocksize,
                                ipiv_d,
                                batch_inv_diagblk_ptr_d,
                                blocksize,
                                info_d,
                                batch_size));


    // record finishing the inverse of the first block
    cudaErrchk(cudaEventRecord(schur_inverted[0], stream[stream_compute]));


    //wait for the inverse of the first block
    cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[0]));
    // 0. Inverse of the first block
    cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_h[0], batch_inv_diagblk_d,
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
    // unloading finished
    cudaErrchk(cudaEventRecord(unload[0], stream[stream_memunload]));


    // first memcpy happens before loop
    cudaErrchk(cudaMemcpyAsync(batch_diagblk_d[1],
                reinterpret_cast<const complex_d*>(batch_diagblk_h[1]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(batch_upperblk_d[1],
                reinterpret_cast<const complex_d*>(batch_upperblk_h[0]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(batch_lowerblk_d[1],
                reinterpret_cast<const complex_d*>(batch_lowerblk_h[0]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));



    // // 1. Forward substitution (performed left to right)
    for (unsigned int i = 1; i < n_blocks; ++i) {


        int stream_memload = (i+1) % 2;
        int stream_compute = i % 2;
        int stream_memunload = 2;


        if(i < n_blocks-1){
            // load the blocks for the next iteration
            cudaErrchk(cudaMemcpyAsync(batch_diagblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(batch_diagblk_h[i+1]),
                        batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(batch_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(batch_upperblk_h[i]),
                        batch_size *  blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(batch_lowerblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(batch_lowerblk_h[i]),
                        batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
        }

        //wait for the schur inverse from the previous iteration
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], schur_inverted[i-1]));

        // MatMul tmp = eig_lowerblk[i-1] * eig_inv_diagblk[i-1]
        // use the batch_inv_diagblk_d from last iteration
        // use batch_inv_lowerblk_d as tmp
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            batch_lowerblk_ptr_d[stream_compute], blocksize,
            batch_inv_diagblk_ptr_d, blocksize,
            &beta,
            batch_inv_lowerblk_ptr_d, blocksize, batch_size));
        //MatMul schur complement = eig_diagblk[i] - tmp * eig_upperblk[i-1]
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);

        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            batch_inv_lowerblk_ptr_d, blocksize,
            batch_upperblk_ptr_d[stream_compute], blocksize,
            &beta,
            batch_diagblk_ptr_d[stream_compute], blocksize, batch_size));



        cublasErrchk(cublasZgetrfBatched(
                cublas_handle[stream_compute],
                blocksize,
                batch_diagblk_ptr_d[stream_compute], blocksize, ipiv_d,
                info_d, batch_size));


        // inversion
        cublasErrchk(cublasZgetriBatched(
                                    cublas_handle[stream_compute],
                                    blocksize,
                                    batch_diagblk_ptr_d[stream_compute],
                                    blocksize,
                                    ipiv_d,
                                    batch_inv_diagblk_ptr_d,
                                    blocksize,
                                    info_d,
                                    batch_size));



        // record finishing of computation in step i
        cudaErrchk(cudaEventRecord(schur_inverted[i], stream[stream_compute]));

        // wait to unload for the finish of computations
        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[i]));
        cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_h[i],
                    batch_inv_diagblk_d,
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        // unloading finished
        cudaErrchk(cudaEventRecord(unload[i], stream[stream_memunload]));

    }


    int stream_memload_before = (n_blocks) % 2;
    int stream_compute_before = (n_blocks-1) % 2;

    cudaErrchk(cudaMemcpyAsync(batch_upperblk_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(batch_upperblk_h[n_blocks-2]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    cudaErrchk(cudaMemcpyAsync(batch_lowerblk_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(batch_lowerblk_h[n_blocks-2]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    // possible race condition with unloading of previous loop
    // not sure
    cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_small_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(batch_inv_diagblk_h[n_blocks-2]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
  

    // TODO possible to save memory by allocating and freeing
    // memory which is not needed anymore (to reduce max memory consumption at one point)


    // 2. Backward substitution (performed right to left)
    for(int i = n_blocks-2; i >= 0; --i){

        // fix stream compute to be the stream which loaded
        // blocks before the loop
        stream_memload = (stream_compute_before + (i - n_blocks + 2) ) % 2;
        stream_compute = (stream_memload_before + (i - n_blocks + 2) ) % 2;
        stream_memunload = 2;

        if(i > 0){
            cudaErrchk(cudaMemcpyAsync(batch_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(batch_upperblk_h[i-1]),
                        batch_size *blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(batch_lowerblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(batch_lowerblk_h[i-1]),
                        batch_size *blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_small_d[stream_memload],
                        reinterpret_cast<const complex_d*>(batch_inv_diagblk_h[i-1]),
                        batch_size *blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));    
        }
    

        // wait for the block of the last iteration
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], schur_inverted[i+1]));

        //tmp = eig_inv_diagblk[i+1] * eig_lowerblk[i]
        // use intermediate_ptr_d as tmp
        // reuse batch_inv_diagblk_d from last iteration
        // which is the last true inverse block
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            batch_inv_diagblk_ptr_d, blocksize,
            batch_lowerblk_ptr_d[stream_compute], blocksize,
            &beta,
            intermediate_ptr_d, blocksize, batch_size));

        // eig_inv_lowerblk[i] = -tmp*eig_inv_diagblk[i];
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        

        // use temporary buffer for batch_inv_lowerblk_d
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            intermediate_ptr_d, blocksize,
            batch_inv_diagblk_small_ptr_d[stream_compute], blocksize,
            &beta,
            batch_diagblk_ptr_d[1], blocksize, batch_size));

        // tmp = eig_inv_diagblk[i] * eig_upperblk[i]
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            batch_inv_diagblk_small_ptr_d[stream_compute], blocksize,
            batch_upperblk_ptr_d[stream_compute], blocksize,
            &beta,
            intermediate_ptr_d, blocksize, batch_size));

        //eig_inv_upperblk[i] = -tmp * eig_inv_diagblk[i+1];
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        // use temporary buffer for batch_inv_upperblk_d
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            intermediate_ptr_d, blocksize,
            batch_inv_diagblk_ptr_d, blocksize,
            &beta,
            batch_diagblk_ptr_d[0], blocksize, batch_size));


        //eig_inv_diagblk[i] -= tmp * eig_inv_lowerblk[i];
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);

        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            intermediate_ptr_d, blocksize,
            batch_diagblk_ptr_d[1], blocksize,
            &beta,
            batch_inv_diagblk_small_ptr_d[stream_compute], blocksize, batch_size));
 

        // wait to not overwrite blocks to unload
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], unload[i+1]));

        // use allocated buffers for inv
        // since host2host is cheap
        // batch_diagblk_d is only used in forward pass
        cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_d,
                    batch_inv_diagblk_small_d[stream_compute],
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        cudaErrchk(cudaMemcpyAsync(batch_inv_upperblk_d,
                    batch_diagblk_d[0],
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        cudaErrchk(cudaMemcpyAsync(batch_inv_lowerblk_d,
                    batch_diagblk_d[1],
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        cudaErrchk(cudaEventRecord(schur_inverted[i], stream[stream_compute]));

        // wait to unload for the finish of computations
        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[i]));

        cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_h[i],
                    batch_inv_diagblk_d,
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        cudaErrchk(cudaMemcpyAsync(batch_inv_upperblk_h[i],
                    batch_inv_upperblk_d,
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        cudaErrchk(cudaMemcpyAsync(batch_inv_lowerblk_h[i],
                    batch_inv_lowerblk_d,
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
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
        if(batch_diagblk_d[i]) {
            cudaErrchk(cudaFree(batch_diagblk_d[i]));
        }
        if(batch_upperblk_d[i]) {
            cudaErrchk(cudaFree(batch_upperblk_d[i]));
        }
        if(batch_lowerblk_d[i]) {
            cudaErrchk(cudaFree(batch_lowerblk_d[i]));
        }
        if(batch_diagblk_ptr_d[i]) {
            cudaErrchk(cudaFree(batch_diagblk_ptr_d[i]));
        }
        if(batch_upperblk_ptr_d[i]) {
            cudaErrchk(cudaFree(batch_upperblk_ptr_d[i]));
        }
        if(batch_lowerblk_ptr_d[i]) {
            cudaErrchk(cudaFree(batch_lowerblk_ptr_d[i]));
        }

    }
    if(batch_inv_diagblk_d) {
        cudaErrchk(cudaFree(batch_inv_diagblk_d));
    }
    if(batch_inv_upperblk_d) {
        cudaErrchk(cudaFree(batch_inv_upperblk_d));
    }
    if(batch_inv_lowerblk_d) {
        cudaErrchk(cudaFree(batch_inv_lowerblk_d));
    }
    if(intermediate_d){
        cudaErrchk(cudaFree(intermediate_d));
    }


    if(batch_inv_diagblk_ptr_d){
        cudaErrchk(cudaFree(batch_inv_diagblk_ptr_d));
    }
    if(batch_inv_lowerblk_ptr_d){
        cudaErrchk(cudaFree(batch_inv_lowerblk_ptr_d));
    }
    if(intermediate_ptr_d){
        cudaErrchk(cudaFree(intermediate_ptr_d));
    }

    for(int i = 0; i < 2; i++){
        if(batch_inv_diagblk_small_d[i]){
            cudaErrchk(cudaFree(batch_inv_diagblk_small_d[i]));
        }    
        if(batch_inv_diagblk_small_ptr_d[i]){
            cudaErrchk(cudaFree(batch_inv_diagblk_small_ptr_d[i]));
        }

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
 
    // std::string base_path = "../../../tests/tests_cases/";
    std::string base_path = "/usr/scratch/mont-fort17/almaeder/rgf_test/";

    // Get matrix parameters
    std::string parameter_path = base_path + "batched_matrix_parameters.txt";
    unsigned int matrix_size;
    unsigned int blocksize;
    unsigned int batch_size;

    load_matrix_parameters_batched(parameter_path.c_str(), &matrix_size, &blocksize, &batch_size);

    unsigned int n_blocks = matrix_size / blocksize;
    unsigned int off_diag_size = matrix_size - blocksize;

    // Print the matrix parameters
    printf("Matrix parameters:\n");
    printf("    Matrix size: %d\n", matrix_size);
    printf("    Block size: %d\n", blocksize);
    printf("    Number of blocks: %d\n", n_blocks);
    printf("    Batch size: %d\n", batch_size);

    complex_h *matrices_diagblk_h[batch_size];
    complex_h *matrices_upperblk_h[batch_size];
    complex_h *matrices_lowerblk_h[batch_size];

    complex_h *inv_matrices_diagblk_ref[batch_size];
    complex_h *inv_matrices_upperblk_ref[batch_size];
    complex_h *inv_matrices_lowerblk_ref[batch_size];

    for(unsigned int batch = 0; batch < batch_size; batch++){

        // Load matrix to invert
        complex_h* matrix_diagblk = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
        std::string diagblk_path = base_path + "dense_blocks_matrix_"+ std::to_string(batch) +"_diagblk.bin";
        load_binary_matrix(diagblk_path.c_str(), matrix_diagblk, blocksize, matrix_size);

        complex_h* matrix_upperblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
        std::string upperblk_path = base_path + "dense_blocks_matrix_"+ std::to_string(batch) +"_upperblk.bin";
        load_binary_matrix(upperblk_path.c_str(), matrix_upperblk, blocksize, off_diag_size);

        complex_h* matrix_lowerblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
        std::string lowerblk_path = base_path + "dense_blocks_matrix_"+ std::to_string(batch) +"_lowerblk.bin";
        load_binary_matrix(lowerblk_path.c_str(), matrix_lowerblk, blocksize, off_diag_size);

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
        matrices_diagblk_h[batch] = matrix_diagblk_h;
        matrices_upperblk_h[batch] = matrix_upperblk_h;
        matrices_lowerblk_h[batch] = matrix_lowerblk_h;


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

        // if(!rgf_dense_matrix_does_not_fit_gpu_memory_with_copy_compute_overlap(blocksize, matrix_size,
        //                                 matrix_diagblk_h,
        //                                 matrix_upperblk_h,
        //                                 matrix_lowerblk_h,
        //                                 inv_diagblk_h,
        //                                 inv_upperblk_h,
        //                                 inv_lowerblk_h)){
        //     printf("Error: rgf_dense_matrix_does_not_fit_gpu_memory_with_copy_compute_overlap failed\n");
        // }
        // else{
        //     printf("rgf_dense_matrix_does_not_fit_gpu_memory_with_copy_compute_overlap succeeded\n");
        // }

        std::cout << "batch " << batch << std::endl;

        // ----- RESULT CHECKING SECTION -----

        // Load reference solution of the matrix inverse
        complex_h* matrix_inv_diagblk_ref = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
        std::string inv_diagblk_path = base_path + "dense_blocks_matrix_" + std::to_string(batch) + "_inverse_diagblk.bin";
        load_binary_matrix(inv_diagblk_path.c_str(), matrix_inv_diagblk_ref, blocksize, matrix_size);

        complex_h* matrix_inv_upperblk_ref = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
        std::string inv_upperblk_path = base_path + "dense_blocks_matrix_" + std::to_string(batch) + "_inverse_upperblk.bin";
        load_binary_matrix(inv_upperblk_path.c_str(), matrix_inv_upperblk_ref, blocksize, off_diag_size);
        
        complex_h* matrix_inv_lowerblk_ref = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
        std::string inv_lowerblk_path = base_path + "dense_blocks_matrix_" + std::to_string(batch) + "_inverse_lowerblk.bin";
        load_binary_matrix(inv_lowerblk_path.c_str(), matrix_inv_lowerblk_ref, blocksize, off_diag_size);


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
        inv_matrices_diagblk_ref[batch] = inv_diagblk_ref;
        inv_matrices_upperblk_ref[batch] = inv_upperblk_ref;
        inv_matrices_lowerblk_ref[batch] = inv_lowerblk_ref;

        // // print last block of inverted matrix
        // for(unsigned int i = blocksize *(matrix_size-blocksize); i < blocksize * matrix_size; i++){
        //     std::cout << "batch_inv_diagblk_h[" << i << "] = " << batch_inv_diagblk_h[i] << std::endl;
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
        // double eps = 1e-9;
        // if(diff_diagblk/norm_diagblk > eps){
        //     printf("Error: batch_inv_diagblk_h and inv_diagblk_ref are not equal\n");
        // }
        // else{
        //     printf("batch_inv_diagblk_h and inv_diagblk_ref are equal\n");
        // }
        // std::cout << diff_diagblk/norm_diagblk << std::endl;

        // if(diff_upperblk/norm_upperblk > eps){
        //     printf("Error: batch_inv_upperblk_h and inv_upperblk_ref are not equal\n");
        // }
        // else{
        //     printf("batch_inv_upperblk_h and inv_upperblk_ref are equal\n");
        // }
        // std::cout << diff_upperblk/norm_upperblk << std::endl;

        // if(diff_lowerblk/norm_lowerblk > eps){
        //     printf("Error: batch_inv_lowerblk_h and inv_lowerblk_ref are not equal\n");
        // }
        // else{
        //     printf("batch_inv_lowerblk_h and inv_lowerblk_ref are equal\n");
        // }
        // std::cout << diff_lowerblk/norm_lowerblk << std::endl;


        if(inv_diagblk_h){
            cudaFreeHost(inv_diagblk_h);
        }
        if(inv_upperblk_h){
            cudaFreeHost(inv_upperblk_h);
        }
        if(inv_lowerblk_h){
            cudaFreeHost(inv_lowerblk_h);
        }
        // free non contiguous memory
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

    }


    // transform to batched blocks
    complex_h *batch_diagblk_h[n_blocks];
    complex_h *batch_upperblk_h[n_blocks-1];
    complex_h *batch_lowerblk_h[n_blocks-1];
    complex_h *batched_inv_matrices_diagblk_ref[n_blocks];
    complex_h *batched_inv_matrices_upperblk_ref[n_blocks-1];
    complex_h *batched_inv_matrices_lowerblk_ref[n_blocks-1];
    complex_h *batch_inv_diagblk_h[n_blocks];
    complex_h *batch_inv_upperblk_h[n_blocks-1];
    complex_h *batch_inv_lowerblk_h[n_blocks-1];

    for(unsigned int i = 0; i < n_blocks; i++){
        cudaErrchk(cudaMallocHost((void**)&batch_diagblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
        cudaErrchk(cudaMallocHost((void**)&batched_inv_matrices_diagblk_ref[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
        cudaErrchk(cudaMallocHost((void**)&batch_inv_diagblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));    
        for(unsigned int batch = 0; batch < batch_size; batch++){
            for(unsigned int j = 0; j < blocksize * blocksize; j++){
                batch_diagblk_h[i][batch * blocksize * blocksize + j] =
                    matrices_diagblk_h[batch][i * blocksize * blocksize + j];
                batched_inv_matrices_diagblk_ref[i][batch * blocksize * blocksize + j] =
                    inv_matrices_diagblk_ref[batch][i * blocksize * blocksize + j];
            }
        }
    


    }
    for(unsigned int i = 0; i < n_blocks - 1; i++){
        cudaErrchk(cudaMallocHost((void**)&batch_upperblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
        cudaErrchk(cudaMallocHost((void**)&batch_lowerblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
        cudaErrchk(cudaMallocHost((void**)&batched_inv_matrices_upperblk_ref[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
        cudaErrchk(cudaMallocHost((void**)&batched_inv_matrices_lowerblk_ref[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
        cudaErrchk(cudaMallocHost((void**)&batch_inv_upperblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
        cudaErrchk(cudaMallocHost((void**)&batch_inv_lowerblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
        for(unsigned int batch = 0; batch < batch_size; batch++){
            for(unsigned int j = 0; j < blocksize * blocksize; j++){
                batch_upperblk_h[i][batch * blocksize * blocksize + j] =
                    matrices_upperblk_h[batch][i * blocksize * blocksize + j];
                batch_lowerblk_h[i][batch * blocksize * blocksize + j] =
                    matrices_lowerblk_h[batch][i * blocksize * blocksize + j];
                batched_inv_matrices_upperblk_ref[i][batch * blocksize * blocksize + j] =
                    inv_matrices_upperblk_ref[batch][i * blocksize * blocksize + j];
                batched_inv_matrices_lowerblk_ref[i][batch * blocksize * blocksize + j] =
                    inv_matrices_lowerblk_ref[batch][i * blocksize * blocksize + j];
            }
        }
    
    }

    rgf_batched(
        blocksize, matrix_size, batch_size,
        batch_diagblk_h, batch_upperblk_h, batch_lowerblk_h,
        batch_inv_diagblk_h, batch_inv_upperblk_h, batch_inv_lowerblk_h
    );

    double norm_diagblk = 0.0;
    double norm_upperblk = 0.0;
    double norm_lowerblk = 0.0;
    double diff_diagblk = 0.0;
    double diff_upperblk = 0.0;
    double diff_lowerblk = 0.0;
    for(unsigned int i = 0; i < n_blocks; i++){
        for(unsigned j = 0; j < batch_size * blocksize * blocksize; j++){
            norm_diagblk += std::abs(batch_inv_diagblk_h[i][j]);
            diff_diagblk += std::abs(batch_inv_diagblk_h[i][j] - batched_inv_matrices_diagblk_ref[i][j]);
        }

    }
    for(unsigned int i = 0; i < n_blocks - 1; i++){
        for(unsigned int j = 0; j < batch_size * blocksize * blocksize; j++){
            norm_upperblk += std::abs(batch_inv_upperblk_h[i][j]);
            norm_lowerblk += std::abs(batch_inv_lowerblk_h[i][j]);
            diff_upperblk += std::abs(batch_inv_upperblk_h[i][j] - batched_inv_matrices_upperblk_ref[i][j]);
            diff_lowerblk += std::abs(batch_inv_lowerblk_h[i][j] - batched_inv_matrices_lowerblk_ref[i][j]);
        }
    }


    double eps = 1e-9;
    if(diff_diagblk/norm_diagblk > eps){
        printf("Error: batch_inv_diagblk_h and batched_inv_matrices_diagblk_ref are not equal\n");
    }
    else{
        printf("batch_inv_diagblk_h and batched_inv_matrices_diagblk_ref are equal\n");
    }
    std::cout << diff_diagblk/norm_diagblk << std::endl;
    if(diff_upperblk/norm_upperblk > eps){
        printf("Error: batch_inv_upperblk_h and batched_inv_matrices_upperblk_ref are not equal\n");
    }
    else{
        printf("batch_inv_upperblk_h and batched_inv_matrices_upperblk_ref are equal\n");
    }
    std::cout << diff_upperblk/norm_upperblk << std::endl;
    if(diff_lowerblk/norm_lowerblk > eps){
        printf("Error: batch_inv_lowerblk_h and batched_inv_matrices_lowerblk_ref are not equal\n");
    }
    else{
        printf("batch_inv_lowerblk_h and batched_inv_matrices_lowerblk_ref are equal\n");
    }
    std::cout << diff_lowerblk/norm_lowerblk << std::endl;


    // set inverse to zero
    for(unsigned int i = 0; i < n_blocks; i++){
        for(unsigned j = 0; j < batch_size * blocksize * blocksize; j++){
            batch_inv_diagblk_h[i][j] = 0.0;
        }

    }
    for(unsigned int i = 0; i < n_blocks - 1; i++){
        for(unsigned int j = 0; j < batch_size * blocksize * blocksize; j++){
            batch_inv_upperblk_h[i][j] = 0.0;
            batch_inv_lowerblk_h[i][j] = 0.0;
        }
    }


    rgf_multiple_energy_points_for_loop(
        blocksize, matrix_size, batch_size,
        batch_diagblk_h, batch_upperblk_h, batch_lowerblk_h,
        batch_inv_diagblk_h, batch_inv_upperblk_h, batch_inv_lowerblk_h
    );

    norm_diagblk = 0.0;
    norm_upperblk = 0.0;
    norm_lowerblk = 0.0;
    diff_diagblk = 0.0;
    diff_upperblk = 0.0;
    diff_lowerblk = 0.0;

    for(unsigned int i = 0; i < n_blocks; i++){
        for(unsigned j = 0; j < batch_size * blocksize * blocksize; j++){
            norm_diagblk += std::abs(batch_inv_diagblk_h[i][j]);
            diff_diagblk += std::abs(batch_inv_diagblk_h[i][j] - batched_inv_matrices_diagblk_ref[i][j]);
        }

    }
    for(unsigned int i = 0; i < n_blocks - 1; i++){
        for(unsigned int j = 0; j < batch_size * blocksize * blocksize; j++){
            norm_upperblk += std::abs(batch_inv_upperblk_h[i][j]);
            norm_lowerblk += std::abs(batch_inv_lowerblk_h[i][j]);
            diff_upperblk += std::abs(batch_inv_upperblk_h[i][j] - batched_inv_matrices_upperblk_ref[i][j]);
            diff_lowerblk += std::abs(batch_inv_lowerblk_h[i][j] - batched_inv_matrices_lowerblk_ref[i][j]);
        }
    }


    if(diff_diagblk/norm_diagblk > eps){
        printf("Error: batch_inv_diagblk_h and batched_inv_matrices_diagblk_ref are not equal\n");
    }
    else{
        printf("batch_inv_diagblk_h and batched_inv_matrices_diagblk_ref are equal\n");
    }
    std::cout << diff_diagblk/norm_diagblk << std::endl;
    if(diff_upperblk/norm_upperblk > eps){
        printf("Error: batch_inv_upperblk_h and batched_inv_matrices_upperblk_ref are not equal\n");
    }
    else{
        printf("batch_inv_upperblk_h and batched_inv_matrices_upperblk_ref are equal\n");
    }
    std::cout << diff_upperblk/norm_upperblk << std::endl;
    if(diff_lowerblk/norm_lowerblk > eps){
        printf("Error: batch_inv_lowerblk_h and batched_inv_matrices_lowerblk_ref are not equal\n");
    }
    else{
        printf("batch_inv_lowerblk_h and batched_inv_matrices_lowerblk_ref are equal\n");
    }
    std::cout << diff_lowerblk/norm_lowerblk << std::endl;



    // free contiguous memory
    for(unsigned int batch = 0; batch < batch_size; batch++){
        if(matrices_diagblk_h[batch]){
            cudaFreeHost(matrices_diagblk_h[batch]);
        }
        if(matrices_upperblk_h[batch]){
            cudaFreeHost(matrices_upperblk_h[batch]);
        }
        if(matrices_lowerblk_h[batch]){
            cudaFreeHost(matrices_lowerblk_h[batch]);
        }
        if(inv_matrices_diagblk_ref[batch]){
            free(inv_matrices_diagblk_ref[batch]);
        }
        if(inv_matrices_upperblk_ref[batch]){
            free(inv_matrices_upperblk_ref[batch]);
        }
        if(inv_matrices_lowerblk_ref[batch]){
            free(inv_matrices_lowerblk_ref[batch]);
        }
    }
    for(unsigned int i = 0; i < n_blocks; i++){
        if(batch_diagblk_h[i]){
            cudaFreeHost(batch_diagblk_h[i]);
        }
        if(batched_inv_matrices_diagblk_ref[i]){
            cudaFreeHost(batched_inv_matrices_diagblk_ref[i]);
        }
        if(batch_inv_diagblk_h[i]){
            cudaFreeHost(batch_inv_diagblk_h[i]);
        }
    }
    for(unsigned int i = 0; i < n_blocks - 1; i++){
        if(batch_upperblk_h[i]){
            cudaFreeHost(batch_upperblk_h[i]);
        }
        if(batch_lowerblk_h[i]){
            cudaFreeHost(batch_lowerblk_h[i]);
        }
        if(batched_inv_matrices_upperblk_ref[i]){
            cudaFreeHost(batched_inv_matrices_upperblk_ref[i]);
        }
        if(batched_inv_matrices_lowerblk_ref[i]){
            cudaFreeHost(batched_inv_matrices_lowerblk_ref[i]);
        }
        if(batch_inv_upperblk_h[i]){
            cudaFreeHost(batch_inv_upperblk_h[i]);
        }
        if(batch_inv_lowerblk_h[i]){
            cudaFreeHost(batch_inv_lowerblk_h[i]);
        }
    }

    return 0;
}








