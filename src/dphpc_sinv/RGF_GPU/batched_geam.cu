#include "batched_geam.h"

__global__ void quatrexblasZgeamBatched_C_C_kernel(
    int m, int n,
    cuDoubleComplex alpha,
    cuDoubleComplex *const * __restrict__ Aarray, int lda,
    cuDoubleComplex beta,
    cuDoubleComplex *const * __restrict__ Barray, int ldb,
    cuDoubleComplex ** __restrict__ Carray, int ldc,
    int batchSize
){
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;

    for(int id = tid; id < m*n*batchSize; id += stride){
        int batch = id/(m*n);
        int ij = id % (m*n);
        int row = ij % m;
        int col = ij / m;

        cuDoubleComplex a = cuConj(Aarray[batch][col + row*lda]);
        cuDoubleComplex b = cuConj(Barray[batch][col + row*ldb]);

        Carray[batch][row + col*ldc] = cuCadd(cuCmul(alpha, a), cuCmul(beta, b));
    }
}

__global__ void quatrexblasZgeamBatched_N_C_kernel(
    int m, int n,
    cuDoubleComplex alpha,
    cuDoubleComplex *const *Aarray, int lda,
    cuDoubleComplex beta,
    cuDoubleComplex *const * __restrict__ Barray, int ldb,
    cuDoubleComplex **Carray, int ldc,
    int batchSize
){
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;

    for(int id = tid; id < m*n*batchSize; id += stride){
        int batch = id/(m*n);
        int ij = id % (m*n);
        int row = ij % m;
        int col = ij / m;

        cuDoubleComplex a = Aarray[batch][row + col*ldc];
        cuDoubleComplex b = cuConj(Barray[batch][col + row*ldb]);

        Carray[batch][row + col*ldc] = cuCadd(cuCmul(alpha, a), cuCmul(beta, b));
    }
}

void quatrexblasZgeamBatched_C_C(
    cublasHandle_t handle,
    int m, int n,
    cuDoubleComplex *alpha,
    cuDoubleComplex *const Aarray[], int lda,
    cuDoubleComplex *beta,
    cuDoubleComplex *const Barray[], int ldb,
    cuDoubleComplex *Carray[], int ldc,
    int batchSize
){
    cudaStream_t stream;
    cublasErrchk(cublasGetStream(handle, &stream));

    int threads = 1024;
    int blocks = (m*n*batchSize + threads - 1)/threads;
    quatrexblasZgeamBatched_C_C_kernel<<<blocks, threads, 0, stream>>>(
        m, n,
        alpha[0],
        Aarray, lda,
        beta[0],
        Barray, ldb,
        Carray, ldc,
        batchSize
    );
}

void quatrexblasZgeamBatched_N_C(
    cublasHandle_t handle,
    int m, int n,
    cuDoubleComplex *alpha,
    cuDoubleComplex *const Aarray[], int lda,
    cuDoubleComplex *beta,
    cuDoubleComplex *const Barray[], int ldb,
    cuDoubleComplex *Carray[], int ldc,
    int batchSize
){
    cudaStream_t stream;
    cublasErrchk(cublasGetStream(handle, &stream));

    int threads = 1024;
    int blocks = (m*n*batchSize + threads - 1)/threads;
    quatrexblasZgeamBatched_N_C_kernel<<<blocks, threads, 0, stream>>>(
        m, n,
        alpha[0],
        Aarray, lda,
        beta[0],
        Barray, ldb,
        Carray, ldc,
        batchSize
    );
}

cublasStatus_t quatrexblasZgeamBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n,
    cuDoubleComplex *alpha,
    cuDoubleComplex *const Aarray[], int lda,
    cuDoubleComplex *beta,
    cuDoubleComplex *const Barray[], int ldb,
    cuDoubleComplex *Carray[], int ldc,
    int batchSize
){
    // assumes arrays are column major
    //TODO: Add error checking
    // inputs are not checked for validity

    if(transa == CUBLAS_OP_C && transb == CUBLAS_OP_C){
        quatrexblasZgeamBatched_C_C(
            handle,
            m, n,
            alpha,
            Aarray, lda,
            beta,
            Barray, ldb,
            Carray, ldc,
            batchSize
        );
    }
    else if(transa == CUBLAS_OP_N && transb == CUBLAS_OP_C){
        quatrexblasZgeamBatched_N_C(
            handle,
            m, n,
            alpha,
            Aarray, lda,
            beta,
            Barray, ldb,
            Carray, ldc,
            batchSize
        );
    }
    else{
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    return CUBLAS_STATUS_SUCCESS;

}