// Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <complex>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>


bool load_binary_matrix(
    const char *filename, 
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
    const char *filename, 
    unsigned int *matrice_size, 
    unsigned int *blocksize);

bool load_matrix_parameters_batched(
    const char *filename, 
    unsigned int *matrice_size, 
    unsigned int *blocksize,
    unsigned int *batchsize);

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
bool assert_array_magnitude(
    T *array_test,
    T *array_ref,
    double abstol,
    double reltol,
    int size);

template<typename T>
void transform_diagblk(
    T *matrix_diagblk,
    T *matrix_diagblk_h,
    unsigned int blocksize,
    unsigned int matrix_size);

template<typename T>
void transform_offblk(
    T *matrix_offblk,
    T *matrix_offblk_h,
    unsigned int blocksize,
    unsigned int off_diag_size);
