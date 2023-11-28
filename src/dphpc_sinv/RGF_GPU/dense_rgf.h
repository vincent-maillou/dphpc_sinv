#include <complex>

using complex_h = std::complex<double>;

bool rgf_dense_matrix_does_not_fit_gpu_memory_with_copy_compute_overlap(
    unsigned int blocksize,
    unsigned int matrix_size,
    complex_h *matrix_diagblk_h,
    complex_h *matrix_upperblk_h,
    complex_h *matrix_lowerblk_h,
    complex_h *inv_diagblk_h,
    complex_h *inv_upperblk_h,
    complex_h *inv_lowerblk_h);
