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


int main() {
    double complex *matrix;
    char filename[] = "../../tests/tests_cases/matrix_0.bin";

    unsigned int mat_rows = 10;
    unsigned int mat_cols = 10;

    load_matrix(filename, &matrix, mat_rows, mat_cols);

    print_matrix(matrix, mat_rows, mat_cols);

    free_matrix(matrix);

    return 0;
}






