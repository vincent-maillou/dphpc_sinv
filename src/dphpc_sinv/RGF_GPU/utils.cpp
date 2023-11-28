// Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.

#include <cstdio>
#include <cstdlib>
#include <complex>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>

#include "utils.h"

bool load_binary_matrix(
    const char *filename, 
    std::complex<double> *matrix, 
    int rows, 
    int cols)
{
    std::FILE *fp;

    fp = std::fopen(filename, "rb");
    if (fp == nullptr) {
        std::printf("Error opening file\n");
        return false;
    }

    std::fread(matrix, sizeof(std::complex<double>), rows * cols, fp);

    std::fclose(fp);
    return true;
}

template<typename T>
bool load_binary_array(
    std::string filename, 
    T *array,
    int size)
{
    std::FILE *fp;

    fp = std::fopen(filename.c_str(), "rb");
    if (fp == nullptr) {
        std::printf("Error opening file\n");
        return false;
    }

    std::fread(array, sizeof(T), size, fp);

    std::fclose(fp);
    return true;
}
template bool load_binary_array<double>(std::string filename, double* array, int size);
template bool load_binary_array<int>(std::string filename, int* array, int size);
template bool load_binary_array<std::complex<double>>(std::string filename, std::complex<double>* array, int size);


void free_matrix(
    std::complex<double> *matrix)
{
    free(matrix);
}


void print_matrix(
    std::complex<double> *matrix, 
    int rows, 
    int cols)
{
    // Access matrix elements
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            std::printf("%f + %fi ", std::real(matrix[i * cols + j]), std::imag(matrix[i * cols + j]));
        std::printf("\n");
    }
}

bool load_matrix_parameters(
    const char *filename, 
    unsigned int *matrice_size, 
    unsigned int *blocksize)
{
    FILE *fp;

    fp = std::fopen(filename, "r");
    if (fp == NULL) {
        std::printf("Error opening file\n");
        return true;
    }

    std::fscanf(fp, "%u %u", matrice_size, blocksize);

    std::fclose(fp);

    return false;
}

bool load_matrix_parameters_batched(
    const char *filename, 
    unsigned int *matrice_size, 
    unsigned int *blocksize,
    unsigned int *batchsize)
{
    FILE *fp;

    fp = std::fopen(filename, "r");
    if (fp == NULL) {
        std::printf("Error opening file\n");
        return true;
    }

    std::fscanf(fp, "%u %u %u", matrice_size, blocksize, batchsize);

    std::fclose(fp);

    return false;
}


template<typename T>
bool load_text_array(
    const char *filename, 
    T *array,
    int size)
{

    std::ifstream ifile(filename, std::ios::in);
    if (!ifile.is_open()) {
        std::printf("Error opening file\n");
        return false;
    }

    double num = T(0.0);
    //keep storing values from the text file so long as data exists:
    for (int i = 0; i < size; i++) {
        ifile >> num;
        array[i] = (T)num;
    }

    ifile.close();

    return true;
}
// Explicit instantiation of the template
// else not found in compilation
// other option would be to put the implementation in the header file
template bool load_text_array<double>(const char* filename, double* array, int size);
template bool load_text_array<int>(const char* filename, int* array, int size);

template <typename T>
bool save_text_array(
    char *filename,
    const T* array,
    int size)
{

    std::ofstream file(filename);
    if (file.is_open()) {
        for(int i = 0; i < size; i++){
            file << array[i] << " "; 
        }
        file.close();
        std::printf("Array data written to file.\n");
        return true;
    } else {
        std::printf("Unable to open the file for writing.\n");
        return false;
    }
}
template bool save_text_array<double>(char* filename, const double* array, int size);

template<typename T>
void sparse_to_dense(
    T *dense_matrix,
    T *data,
    int *indices,
    int *indptr,
    int matrice_size)
{

    for (int i = 0; i < matrice_size; i++) {
        for (int j = 0; j < matrice_size; j++) {
            // could not work for complex data type
            dense_matrix[i*matrice_size + j] = T(0);
        }
    }

    for(int i = 0; i < matrice_size; i++){
        for(int j = indptr[i]; j < indptr[i+1]; j++){
            dense_matrix[i*matrice_size + indices[j]] = data[j];
        }
    }
}

template void sparse_to_dense<double>(double *dense_matrix,
    double *data,
    int *indices,
    int *indptr,
    int matrice_size);

template<typename T>
void copy_array(
    T *array,
    T *copy,
    int size)
{
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        copy[i] = array[i];
    }
}
template void copy_array<double>(double *array, double *copy, int size);
template void copy_array<int>(int *array, int *copy, int size);
template void copy_array<std::complex<double>>(std::complex<double> *array, std::complex<double> *copy, int size);

template<typename T>
bool assert_same_array(
    T *array1,
    T *array2,
    double epsilon,
    int size)
{
    for (int i = 0; i < size; i++) {
        if ( std::abs(array1[i] - array2[i]) > epsilon) {
            std::printf("Arrays are not the same at index %d\n", i);
            return false;
        }
    }
    return true;
}
template bool assert_same_array<double>(double *array1, double *array2, double epsilon, int size);
template bool assert_same_array<int>(int *array1, int *array2, double epsilon, int size);
template bool assert_same_array<std::complex<double>>(std::complex<double> *array1, std::complex<double> *array2, double epsilon, int size);


bool are_equals(
    std::complex<double> *A,
    std::complex<double> *B,
    unsigned int matrice_size, 
    unsigned int blocksize)
{
    for(unsigned int i = 0; i < blocksize; i++){
        for(unsigned int j = 0; j < matrice_size; j++){
            if (std::abs(A[i*matrice_size+j] - B[i*matrice_size+j]) > 1e-10) {
                std::cout << "A[" << i << "][" << j << "] = " << A[i * matrice_size + j] << std::endl;
                return false;
            }
        }
    }
    return true;
}