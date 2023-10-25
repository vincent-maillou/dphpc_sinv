#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

#define ROWS 10
#define COLS 10



int load_matrix(
    char *filename, 
    double complex *matrix, 
    int rows, 
    int cols)
{
    FILE *fp;

    fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("Error opening file\n");
        return 1;
    }

    matrix = (double complex *)malloc(rows * cols * sizeof(double complex));

    fread(matrix, sizeof(double complex), ROWS * COLS, fp);

    fclose(fp);
}


void print_matrix(
    double complex *matrix, 
    int rows, 
    int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            printf("%f + %fi ", creal(matrix[i * cols + j]), cimag(matrix[i * cols + j]));
        }
        printf("\n");
    }
}



int main() {
    double complex *matrix;
    char filename[] = "../../tests/tests_cases/matrix_0.bin";

    load_matrix(filename, matrix, ROWS, COLS);

    print_matrix((double complex *)matrix, ROWS, COLS);

    return 0;
}



/* int main() {
    FILE *fp;
    char filename[] = "../../tests/tests_cases/matrix_0.bin";
    double complex matrix[ROWS][COLS];

    fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("Error opening file\n");
        return 1;
    }

    fread(matrix, sizeof(double complex), ROWS * COLS, fp);

    fclose(fp);

    // Access matrix elements
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%f + %fi ", creal(matrix[i][j]), cimag(matrix[i][j]));
        }
        printf("\n");
    }

    return 0;
} */