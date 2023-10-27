/*
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-09

Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
*/

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

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

