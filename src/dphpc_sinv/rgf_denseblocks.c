/*
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-09

Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
*/

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

#include "/home/vmaillou/Documents/eigen/Eigen/Dense"

//#include "utils.h"


int load_matrix(
    char *filename, 
    double complex **matrix, 
    int rows, 
    int cols)
{
    FILE *fp;

    fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("Error opening file\n");
        return 1;
    }

    *matrix = malloc(rows * cols * sizeof(double complex));

    fread(*matrix, sizeof(double complex), rows * cols, fp);

    fclose(fp);
}


void free_matrix(
    double complex *matrix)
{
    free(matrix);
}


void print_matrix(
    double complex *matrix, 
    int rows, 
    int cols)
{
    // Access matrix elements
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("%f + %fi ", creal(matrix[i * cols + j]), cimag(matrix[i * cols + j]));
        printf("\n");
    }
}


int load_matrix_parameters(
    char *filename, 
    unsigned int *matrice_size, 
    unsigned int *blocksize)
{
    FILE *fp;

    fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Error opening file\n");
        return 1;
    }

    fscanf(fp, "%u %u", matrice_size, blocksize);

    fclose(fp);

    return 0;
}







int main() {
    // Get matrix parameters
    char f_matparam[] = "../../tests/tests_cases/mat_parameters_0.txt";
    unsigned int matrice_size;
    unsigned int blocksize;

    load_matrix_parameters(f_matparam, &matrice_size, &blocksize);
    

    // Load matrix to invert
    double complex *matrix_diagblk;
    char f_mat_diagblk[] = "../../tests/tests_cases/matrix_0_diagblk.bin";
    load_matrix(f_mat_diagblk, &matrix_diagblk, blocksize, matrice_size);

    double complex *matrix_upperblk;
    char f_mat_upperblk[] = "../../tests/tests_cases/matrix_0_upperblk.bin";
    load_matrix(f_mat_upperblk, &matrix_upperblk, blocksize, matrice_size-blocksize);

    double complex *matrix_lowerblk;
    char f_mat_lowerblk[] = "../../tests/tests_cases/matrix_0_lowerblk.bin";
    load_matrix(f_mat_lowerblk, &matrix_lowerblk, blocksize, matrice_size-blocksize);


    // Load reference solution of the matrix inverse
    double complex *matrix_inv_diagblk;
    char f_mat_inv_diagblk[] = "../../tests/tests_cases/matrix_0_inverse_diagblk.bin";
    load_matrix(f_mat_inv_diagblk, &matrix_inv_diagblk, blocksize, matrice_size);

    double complex *matrix_inv_upperblk;
    char f_mat_inv_upperblk[] = "../../tests/tests_cases/matrix_0_inverse_upperblk.bin";
    load_matrix(f_mat_inv_upperblk, &matrix_inv_upperblk, blocksize, matrice_size-blocksize);
    
    double complex *matrix_inv_lowerblk;
    char f_mat_inv_lowerblk[] = "../../tests/tests_cases/matrix_0_inverse_lowerblk.bin";
    load_matrix(f_mat_inv_lowerblk, &matrix_inv_lowerblk, blocksize, matrice_size-blocksize);

    // ----- END OF INIT SECTION -----

    Eigen::MatrixXcd eig_matrix_diagblk  = Eigen::Map<Eigen::MatrixXcd>(reinterpret_cast<std::complex<double>*>(matrix_diagblk), blocksize, matrice_size);
    Eigen::MatrixXcd eig_matrix_upperblk = Eigen::Map<Eigen::MatrixXcd>(reinterpret_cast<std::complex<double>*>(matrix_upperblk), blocksize, matrice_size-blocksize);
    Eigen::MatrixXcd eig_matrix_lowerblk = Eigen::Map<Eigen::MatrixXcd>(reinterpret_cast<std::complex<double>*>(matrix_lowerblk), blocksize, matrice_size-blocksize);

    







    // ----- START OF CLEANING SECTION -----

    free_matrix(matrix_diagblk);
    free_matrix(matrix_upperblk);
    free_matrix(matrix_lowerblk);
    free_matrix(matrix_inv_diagblk);
    free_matrix(matrix_inv_upperblk);
    free_matrix(matrix_inv_lowerblk);

    return 0;
}








