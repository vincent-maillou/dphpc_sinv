#include <cstdio>
#include <cstdlib>
#include <complex>

bool load_binary_matrix(
    char *filename, 
    std::complex<double> **matrix, 
    int rows, 
    int cols);

template<typename T>
bool load_text_vector(
    char *filename, 
    T **matrix,
    int number_of_elements);

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
void sparse_to_dense(
    T **dense_matrix,
    T *data,
    int *indices,
    int *indptr,
    int matrice_size);