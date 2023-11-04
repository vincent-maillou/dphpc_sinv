/*
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-09

Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
*/

#include <cstdio>
#include <cstdlib>
#include <complex>
#include <fstream>
#include <vector>

#include "utils.h"

bool load_binary_matrix(
    char *filename, 
    std::complex<double> **matrix, 
    int rows, 
    int cols)
{
    std::FILE *fp;

    fp = std::fopen(filename, "rb");
    if (fp == nullptr) {
        std::printf("Error opening file\n");
        return false;
    }

    *matrix = (std::complex<double>*) malloc(rows * cols * sizeof(std::complex<double>));

    std::fread(*matrix, sizeof(std::complex<double>), rows * cols, fp);

    std::fclose(fp);
    return true;
}

template<typename T>
bool load_text_vector(
    char *filename, 
    T **matrix,
    int number_of_elements)
{

    std::ifstream ifile(filename, std::ios::in);
    if (!ifile.is_open()) {
        std::printf("Error opening file\n");
        return false;
    }

    *matrix = (T*) malloc(number_of_elements* sizeof(T));

    T num = 0.0;
    //keep storing values from the text file so long as data exists:
    for (int i = 0; i < number_of_elements; i++) {
        ifile >> num;
        (*matrix)[i] = num;
    }

    ifile.close();

    return true;
}
// Explicit instantiation of the template
// else not found in compilation
// other option would be to put the implementation in the header file
template bool load_text_vector<double>(char* filename, double** matrix, int number_of_elements);
template bool load_text_vector<int>(char* filename, int** matrix, int number_of_elements);


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
    char *filename, 
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

template<typename T>
void sparse_to_dense(
    T **dense_matrix,
    T *data,
    int *indices,
    int *indptr,
    int matrice_size)
{

    *dense_matrix = (T*) malloc(matrice_size * matrice_size * sizeof(T));

    for (int i = 0; i < matrice_size; i++) {
        for (int j = 0; j < matrice_size; j++) {
            // this seems kinda illegal
            (*dense_matrix)[i*matrice_size + j] = (T)0;
        }
    }

    for(int i = 0; i < matrice_size; i++){
        for(int j = indptr[i]; j < indptr[i+1]; j++){
            (*dense_matrix)[i*matrice_size + indices[j]] = data[j];
        }
    }
}

template void sparse_to_dense<double>(double **dense_matrix,
    double *data,
    int *indices,
    int *indptr,
    int matrice_size);