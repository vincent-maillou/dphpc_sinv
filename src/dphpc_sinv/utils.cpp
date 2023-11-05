/*
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-09

Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
*/

#include <cstdio>
#include <cstdlib>
#include <complex>

int load_matrix(
    char *filename, 
    std::complex<double> **matrix, 
    int rows, 
    int cols)
{
    std::FILE *fp;

    fp = std::fopen(filename, "rb");
    if (fp == NULL) {
        std::printf("Error opening file\n");
        return 1;
    }

    *matrix = (std::complex<double>*) malloc(rows * cols * sizeof(std::complex<double>));

    std::fread(*matrix, sizeof(std::complex<double>), rows * cols, fp);

    std::fclose(fp);
    return 0;
}


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


int load_matrix_parameters(
    char *filename, 
    unsigned int *matrice_size, 
    unsigned int *blocksize)
{
    FILE *fp;

    fp = std::fopen(filename, "r");
    if (fp == NULL) {
        std::printf("Error opening file\n");
        return 1;
    }

    std::fscanf(fp, "%u %u", matrice_size, blocksize);

    std::fclose(fp);

    return 0;
}

bool are_equals(
    std::complex<double> *A,
    std::complex<double> *B,
    unsigned int matrice_size, 
    unsigned int blocksize)
{
    // Check that the two parsed matrices are equals
    for (unsigned int i = 0; i < matrice_size; i++) {
        for (unsigned int j = 0; j < blocksize; j++) {
            if (std::abs(A[i * matrice_size + j] - B[i * matrice_size + j]) > 1e-10) {
                return false;
            }
        }
    }
    return true;
}

