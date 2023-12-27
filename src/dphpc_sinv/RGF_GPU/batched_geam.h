#pragma once
#include <cuda.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cuComplex.h>
#include "cudaerrchk.h"

cublasStatus_t quatrexblasZgeamBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n,
    cuDoubleComplex *alpha,
    cuDoubleComplex *const Aarray[], int lda,
    cuDoubleComplex *beta,
    cuDoubleComplex *const Barray[], int ldb,
    cuDoubleComplex *Carray[], int ldc,
    int batchCount
);