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
#include <cuda_runtime.h>
#include "utils.h"

using complex_h = std::complex<double>;
using complex_d = cuDoubleComplex;

void rgf_lesser_greater_for(
    unsigned int blocksize,
    unsigned int matrix_size,
    unsigned int batch_size,
    complex_h **batch_diagblk_h,
    complex_h **batch_upperblk_h,
    complex_h **batch_lowerblk_h,
    complex_h **batch_inv_diagblk_h,
    complex_h **batch_inv_upperblk_h,
    complex_h **batch_inv_lowerblk_h);

void rgf_lesser_greater_batched(
    unsigned int blocksize,
    unsigned int matrix_size,
    unsigned int batch_size,
    complex_h **batch_diagblk_h,
    complex_h **batch_upperblk_h,
    complex_h **batch_lowerblk_h,
    complex_h **batch_inv_diagblk_h,
    complex_h **batch_inv_upperblk_h,
    complex_h **batch_inv_lowerblk_h);