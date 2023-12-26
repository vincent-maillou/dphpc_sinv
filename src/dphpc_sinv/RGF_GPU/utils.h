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
