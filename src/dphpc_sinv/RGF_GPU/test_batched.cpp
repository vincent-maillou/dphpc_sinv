#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <iostream>
#include <string>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include "magma_v2.h"


#define cudaErrchk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
        std::cout << code << std::endl;
        std::printf("CUDAassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
   }
}

#define cublasErrchk(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true)
{
   
   if (code != CUBLAS_STATUS_SUCCESS) 
   {
        std::cout << code << std::endl;
        std::cout << CUBLAS_STATUS_ARCH_MISMATCH << std::endl;
        std::cout << CUBLAS_STATUS_EXECUTION_FAILED << std::endl;
        std::cout << CUBLAS_STATUS_INTERNAL_ERROR << std::endl;
        std::cout << CUBLAS_STATUS_INVALID_VALUE << std::endl;
        std::cout << CUBLAS_STATUS_MAPPING_ERROR << std::endl;
        std::cout << CUBLAS_STATUS_NOT_INITIALIZED << std::endl;
        std::cout << CUBLAS_STATUS_NOT_SUPPORTED << std::endl;
        std::cout << CUBLAS_STATUS_INVALID_VALUE << std::endl;
        //Did not find a counter part to cudaGetErrorString in cublas
        std::printf("CUBLASassert: %s %s %d\n", cudaGetErrorString((cudaError_t)code), file, line);
        if (abort) exit(code);
   }
}

// both should be equivalent, thus reinterpret_cast should be fine
using complex_h = std::complex<double>;
using complex_d = cuDoubleComplex; //cuda::complex_h;


int main() {
    cudaSetDevice(0);
    magma_int_t init = magma_init();
    magma_print_environment();
    if(init != MAGMA_SUCCESS){
        printf("Error: magma_init() failed\n");
        return 1;
    }
    



    if(sizeof(magma_int_t) != sizeof(int)){
        printf("Error: sizeof(magma_int_t) != sizeof(int)\n");
        return 1;
    }


    int blocksize = 2;
    int batch_size = 3;
    cublasHandle_t cublas_handle;
    cublasErrchk(cublasCreate(&cublas_handle));
    cudaStream_t stream;
    cudaErrchk(cudaStreamCreate(&stream));
    cublasErrchk(cublasSetStream(cublas_handle, stream));

    magma_queue_t magma_queue;
    magma_queue_create(0, &magma_queue);

    complex_d* tmp1_ptr_h[batch_size];
    complex_d* tmp2_ptr_h[batch_size];
    complex_d* tmp3_ptr_h[batch_size];
    complex_d** tmp1_ptr_d;
    complex_d** tmp2_ptr_d;
    complex_d** tmp3_ptr_d;
    complex_d* tmp1_d;
    complex_d* tmp2_d;
    complex_d* tmp3_d;
    complex_h tmp1_h[batch_size * blocksize * blocksize];
    complex_h tmp2_h[batch_size * blocksize * blocksize];
    complex_h tmp3_h[batch_size * blocksize * blocksize];

    // fill tmp1_h with unit matrix for each batch
    for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < blocksize * blocksize; j++){
            tmp1_h[i * blocksize * blocksize + j] = 1.0;
        }
        
        for(int j = 0; j < blocksize; j++){
            tmp1_h[i * blocksize * blocksize + j * blocksize + j] = 10.0;
        }
        tmp1_h[i * blocksize * blocksize + 0] = 9.0;
    }

    for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < blocksize * blocksize; j++){
            tmp2_h[i * blocksize * blocksize + j] = 0.0;
        }
        for(int j = 0; j < blocksize; j++){
            tmp2_h[i * blocksize * blocksize + j * blocksize + j] = 10.0;
        }
    }

    std::cout << "tmp1_h before getrf" << std::endl;
    for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < blocksize * blocksize; j++){
            std::cout << tmp1_h[i * blocksize * blocksize + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "tmp2_h before matmul" << std::endl;
    for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < blocksize * blocksize; j++){
            std::cout << tmp2_h[i * blocksize * blocksize + j] << " ";
        }
        std::cout << std::endl;
    }


    cudaErrchk(cudaMalloc((void**)&tmp1_d, batch_size * blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&tmp2_d, batch_size * blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&tmp3_d, batch_size * blocksize * blocksize * sizeof(complex_d)));


    cudaErrchk(cudaMalloc((void**)&tmp1_ptr_d, batch_size * sizeof(complex_d*)));
    cudaErrchk(cudaMalloc((void**)&tmp2_ptr_d, batch_size * sizeof(complex_d*)));
    cudaErrchk(cudaMalloc((void**)&tmp3_ptr_d, batch_size * sizeof(complex_d*)));



    for(int j = 0; j < batch_size; j++){
        tmp1_ptr_h[j] = tmp1_d + j * blocksize * blocksize;
        tmp2_ptr_h[j] = tmp2_d + j * blocksize * blocksize;
        tmp3_ptr_h[j] = tmp3_d + j * blocksize * blocksize;
    }


    cudaErrchk(cudaMemcpy(tmp1_ptr_d, tmp1_ptr_h, batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(tmp2_ptr_d, tmp2_ptr_h, batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(tmp3_ptr_d, tmp3_ptr_h, batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));

    cudaErrchk(cudaMemset(tmp3_d, 1.0, batch_size * blocksize * blocksize * sizeof(complex_d)));
    
    cudaErrchk(cudaMemcpy(tmp1_d, tmp1_h, batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(tmp2_d, tmp2_h, batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice));

    int *info_d = NULL;
    int info_h[batch_size];
    cudaErrchk(cudaMalloc((void**)&info_d, batch_size * sizeof(int)))

    int *ipiv_d = NULL;
    int ipiv_h[batch_size * blocksize];
    cudaErrchk(cudaMalloc((void**)&ipiv_d, batch_size * blocksize * sizeof(int)));

    int *ipiv_ptr_h[batch_size];
    int **ipiv_ptr_d;
    for(int i = 0; i < batch_size; i++){
        ipiv_ptr_h[i] = ipiv_d + i * blocksize;
    }
    cudaErrchk(cudaMalloc((void**)&ipiv_ptr_d, batch_size * sizeof(int*)));
    cudaErrchk(cudaMemcpy(ipiv_ptr_d, ipiv_ptr_h, batch_size * sizeof(int*), cudaMemcpyHostToDevice));



    int ifk1 = -1;
    magma_queue_sync(magma_queue);


    int* m_d = NULL;
    cudaErrchk(cudaMalloc((void**)&m_d, batch_size*sizeof(int)));
    cudaErrchk(cudaMemset(m_d, blocksize, batch_size*sizeof(int)));
    int *n_d = NULL;
    cudaErrchk(cudaMalloc((void**)&n_d, batch_size*sizeof(int)));
    cudaErrchk(cudaMemset(n_d, blocksize, batch_size*sizeof(int)));

    int *ldda = NULL;
    cudaErrchk(cudaMalloc((void**)&ldda, batch_size*sizeof(int)));
    cudaErrchk(cudaMemset(ldda, blocksize, batch_size*sizeof(int)));


    cublasZgetrfBatched(
                    cublas_handle,
                    blocksize,
                    tmp1_ptr_d, 
                    blocksize,
                    ipiv_d,
                    info_d,
                    batch_size);


    magma_queue_sync(magma_queue);
    cudaErrchk(cudaDeviceSynchronize());

    cudaMemcpy(ipiv_h, ipiv_d, batch_size * blocksize * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "ipiv_h after getrf" << std::endl;
    for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < blocksize; j++){
            std::cout << ipiv_h[i * blocksize + j] << " ";
        }
        std::cout << std::endl;
    }

    //TODO check info_d for errors
    cudaErrchk(cudaMemcpy(info_h, info_d, batch_size * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "info_h after getrf" << std::endl;
    for(int i = 0; i < batch_size; i++){
        std::cout << "info_h[" << i << "] = " << info_h[i] << std::endl;
    }
    std::cout << ifk1 << std::endl;

    cudaMemcpy(tmp1_h, tmp1_d, batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost);
    std::cout << "tmp1_h after getrf" << std::endl;
    for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < blocksize * blocksize; j++){
            std::cout << tmp1_h[i * blocksize * blocksize + j] << " ";
        }
        std::cout << std::endl;
    }

    complex_d alpha = make_cuDoubleComplex(1.0, 0.0);
    complex_d beta = make_cuDoubleComplex(0.0, 0.0);
    cublasErrchk(cublasZgemmBatched(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        blocksize, blocksize, blocksize,
        &alpha,
        tmp1_ptr_d, blocksize,
        tmp2_ptr_d, blocksize,
        &beta,
        tmp3_ptr_d, blocksize,
        batch_size));

    cudaMemcpy(tmp3_h, tmp3_d, batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost);
    std::cout << "tmp3_h after matmul" << std::endl;
    for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < blocksize * blocksize; j++){
            std::cout << tmp3_h[i * blocksize * blocksize + j] << " ";
        }
        std::cout << std::endl;
    }


    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaFree(tmp1_d));
    cudaErrchk(cudaFree(tmp2_d));
    cudaErrchk(cudaFree(tmp3_d));
    cudaErrchk(cudaFree(tmp1_ptr_d));
    cudaErrchk(cudaFree(tmp2_ptr_d));
    cudaErrchk(cudaFree(tmp3_ptr_d));
    cudaErrchk(cudaFree(info_d));
    cudaErrchk(cudaFree(ipiv_d));
    cudaErrchk(cudaFree(ipiv_ptr_d));
    cudaErrchk(cudaStreamDestroy(stream));
    cublasErrchk(cublasDestroy(cublas_handle));
    cudaErrchk(cudaFree(m_d));
    cudaErrchk(cudaFree(n_d));

    return 0;

}