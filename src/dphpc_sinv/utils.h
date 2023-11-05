#include <cstdio>
#include <cstdlib>
#include <complex>

bool load_binary_matrix(
    char *filename, 
    std::complex<double> **matrix, 
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
bool load_text_vector(
    char *filename, 
    T *matrix,
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
    double epsilon,
    int size);