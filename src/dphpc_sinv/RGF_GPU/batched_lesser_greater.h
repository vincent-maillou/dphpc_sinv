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
#include "cudaerrchk.h"

using complex_h = std::complex<double>;
using complex_d = cuDoubleComplex;

void rgf_lesser_greater_batched(
    unsigned int blocksize,
    unsigned int matrix_size,
    unsigned int batch_size,
    complex_h **system_matrix_diagblk_h,
    complex_h **system_matrix_upperblk_h,
    complex_h **system_matrix_lowerblk_h,
    complex_h **self_energy_lesser_diagblk_h,
    complex_h **self_energy_lesser_upperblk_h,
    complex_h **self_energy_greater_diagblk_h,
    complex_h **self_energy_greater_upperblk_h,
    complex_h **lesser_inv_diagblk_h,
    complex_h **lesser_inv_upperblk_h,
    complex_h **greater_inv_diagblk_h,
    complex_h **greater_inv_upperblk_h);

void rgf_lesser_greater_for(
    unsigned int blocksize,
    unsigned int matrix_size,
    unsigned int batch_size,
    complex_h **system_matrix_diagblk_h,
    complex_h **system_matrix_upperblk_h,
    complex_h **system_matrix_lowerblk_h,
    complex_h **self_energy_lesser_diagblk_h,
    complex_h **self_energy_lesser_upperblk_h,
    complex_h **self_energy_greater_diagblk_h,
    complex_h **self_energy_greater_upperblk_h,
    complex_h **lesser_inv_diagblk_h,
    complex_h **lesser_inv_upperblk_h,
    complex_h **greater_inv_diagblk_h,
    complex_h **greater_inv_upperblk_h);


void rgf_lesser_greater_batched_without_hostmalloc(
    unsigned int blocksize,
    unsigned int matrix_size,
    unsigned int batch_size,
    complex_h **system_matrix_diagblk_h,
    complex_h **system_matrix_upperblk_h,
    complex_h **system_matrix_lowerblk_h,
    complex_h **self_energy_lesser_diagblk_h,
    complex_h **self_energy_lesser_upperblk_h,
    complex_h **self_energy_greater_diagblk_h,
    complex_h **self_energy_greater_upperblk_h,
    complex_h **lesser_inv_diagblk_h,
    complex_h **lesser_inv_upperblk_h,
    complex_h **greater_inv_diagblk_h,
    complex_h **greater_inv_upperblk_h);