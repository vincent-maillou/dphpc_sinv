#include <cstdio>
#include <cstdlib>
#include <complex>

int load_matrix(
    char *filename, 
    std::complex<double> **matrix, 
    int rows, 
    int cols);

void free_matrix(
    std::complex<double> *matrix);

void print_matrix(
    std::complex<double> *matrix, 
    int rows, 
    int cols);

int load_matrix_parameters(
    char *filename, 
    unsigned int *matrice_size, 
    unsigned int *blocksize);
