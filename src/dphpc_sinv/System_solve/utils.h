#include <cstdlib>
#include <complex>
#include <cstdio>


bool load_binary_matrix(
    char *filename, 
    std::complex<double> *matrix, 
    int rows, 
    int cols);


void free_matrix(
    std::complex<double> *matrix);

void print_matrix(
    std::complex<double> *matrix, 
    int rows, 
    int cols);

bool load_matrix_parameters(
    char *filename, 
    unsigned int *matrice_size, 
    unsigned int *blocksize);

template<typename T>
bool load_text_array(
    char *filename, 
    T *array,
    int size);

template <typename T>
bool save_text_array(
    char *filename,
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
bool assert_same_array(
    T *array1,
    T *array2,
    double tolerance,
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
