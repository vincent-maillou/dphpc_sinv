// Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.

#include <cstdlib>
#include <complex>
#include <cstdio>


bool load_binary_matrix(
    char *filename, 
    std::complex<double> *matrix, 
    int rows, 
    int cols);

template<typename T>
bool load_binary_array(
    std::string filename, 
    T *array,
    int size);

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
    const char *filename, 
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
    double epsilon,
    int size);

bool are_equals(
    std::complex<double> *A,
    std::complex<double> *B,
    unsigned int matrice_size, 
    unsigned int blocksize);
