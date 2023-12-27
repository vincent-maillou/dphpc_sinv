// Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
#pragma once
#include <stdio.h>
#include <complex>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>
#include <cuda/std/complex>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include "cudaerrchk.h"

bool rgf_sparse_matrix_does_not_fit_gpu_memory_with_copy_compute_overlap(
    unsigned int blocksize,
    unsigned int matrix_size,
    int* diag_nnz,
    int* upper_nnz,
    int* lower_nnz,
    complex_h **diagblk_data_h,
    int **diagblk_indices_h,
    int **diagblk_indptr_h,
    complex_h **upperblk_data_h,
    int **upperblk_indices_h,
    int **upperblk_indptr_h,
    complex_h **lowerblk_data_h,
    int **lowerblk_indices_h,
    int **lowerblk_indptr_h,
    complex_h *inv_diagblk_h,
    complex_h *inv_upperblk_h,
    complex_h *inv_lowerblk_h);