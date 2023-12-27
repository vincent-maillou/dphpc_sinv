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
#include <cuda_runtime.h>
#include "utils.h"
#include "cudaerrchk.h"

// both should be equivalent, thus reinterpret_cast should be fine
using complex_h = std::complex<double>;
using complex_d = cuDoubleComplex; //cuda::complex_h;

void rgf_retarded_for(
    unsigned int blocksize,
    unsigned int matrix_size,
    unsigned int batch_size,
    complex_h **batch_diagblk_h,
    complex_h **batch_upperblk_h,
    complex_h **batch_lowerblk_h,
    complex_h **batch_inv_diagblk_h,
    complex_h **batch_inv_upperblk_h,
    complex_h **batch_inv_lowerblk_h);

void rgf_retarded_batched(
    unsigned int blocksize,
    unsigned int matrix_size,
    unsigned int batch_size,
    complex_h **batch_diagblk_h,
    complex_h **batch_upperblk_h,
    complex_h **batch_lowerblk_h,
    complex_h **batch_inv_diagblk_h,
    complex_h **batch_inv_upperblk_h,
    complex_h **batch_inv_lowerblk_h);

void rgf_retarded_batched_strided(
    unsigned int blocksize,
    unsigned int matrix_size,
    unsigned int batch_size,
    complex_h **batch_diagblk_h,
    complex_h **batch_upperblk_h,
    complex_h **batch_lowerblk_h,
    complex_h **batch_inv_diagblk_h,
    complex_h **batch_inv_upperblk_h,
    complex_h **batch_inv_lowerblk_h);