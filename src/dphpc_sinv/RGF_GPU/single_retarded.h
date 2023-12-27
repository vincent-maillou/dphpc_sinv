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
#include "cudaerrchk.h"

using complex_h = std::complex<double>;
using complex_d = cuDoubleComplex;


void rgf_retarded_fits_gpu_memory(
    unsigned int blocksize,
    unsigned int matrix_size,
    complex_h *matrix_diagblk_h,
    complex_h *matrix_upperblk_h,
    complex_h *matrix_lowerblk_h,
    complex_h *inv_diagblk_h,
    complex_h *inv_upperblk_h,
    complex_h *inv_lowerblk_h);

void rgf_retarded_fits_gpu_memory_with_copy_compute_overlap(
    unsigned int blocksize,
    unsigned int matrix_size,
    complex_h *matrix_diagblk_h,
    complex_h *matrix_upperblk_h,
    complex_h *matrix_lowerblk_h,
    complex_h *inv_diagblk_h,
    complex_h *inv_upperblk_h,
    complex_h *inv_lowerblk_h);

void rgf_retarded_does_not_fit_gpu_memory(
    unsigned int blocksize,
    unsigned int matrix_size,
    complex_h *matrix_diagblk_h,
    complex_h *matrix_upperblk_h,
    complex_h *matrix_lowerblk_h,
    complex_h *inv_diagblk_h,
    complex_h *inv_upperblk_h,
    complex_h *inv_lowerblk_h);

void rgf_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap(
    unsigned int blocksize,
    unsigned int matrix_size,
    complex_h *matrix_diagblk_h,
    complex_h *matrix_upperblk_h,
    complex_h *matrix_lowerblk_h,
    complex_h *inv_diagblk_h,
    complex_h *inv_upperblk_h,
    complex_h *inv_lowerblk_h);

