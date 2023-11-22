#include <cstdlib>
#include <complex>
#include <cstdio>


template<typename T>
bool load_binary_array(
    std::string filename, 
    T *array, 
    int size);

template<typename T>
bool load_text_array(
    std::string filename, 
    T *array,
    int size);

template <typename T>
bool save_text_array(
    std::string filename,
    const T* array,
    int size);

template<typename T>
void sparse_to_dense(
    T *dense_matrix,
    T *data,
    int *indices,
    int *indptr,
    int matrice_size);

template<typename T>
void copy_array(
    T *array,
    T *copy,
    int size);

template<typename T>
bool assert_array_elementwise(
    T *array_test,
    T *array_ref,
    double abstol,
    double reltol,
    int size);

template<typename T>
bool assert_array_magnitude(
    T *array_test,
    T *array_ref,
    double abstol,
    double reltol,
    int size);

bool are_equals(
    std::complex<double> *A,
    std::complex<double> *B,
    unsigned int matrice_size, 
    unsigned int blocksize);

template<typename T>
void calc_bandwidth(
    T * matrix,
    int matrix_size,
    int * ku,
    int * kl);

template<typename T>
void dense_to_band_for_LU(
    T *dense_matrix,
    T *matrix_band,
    int matrix_size,
    int ku,
    int kl);

template<typename T>
void dense_to_band_for_U_CHOL(
    T *dense_matrix,
    T *matrix_band,
    int matrix_size,
    int kd);

template<typename T>
bool assert_symmetric(
    T *dense_matrix,
    int matrix_size,
    double abstol,
    double reltol);
