#include <string> 

#include "utils.h"
#include "mkl.h"


// todo I am not sure how to "corretly" compine .cu and .cpp
// what is the "correct" way
bool benchmark_manasa()
{
    // Get matrix parameters
    char path_data[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/matrix_sparse_data0.txt";
    char path_indices[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/matrix_sparse_indices0.txt";
    char path_indptr[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/matrix_sparse_indptr0.txt";
    char path_rhs[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/rhs_0.txt";
    int matrice_size = 7165;
    int number_of_nonzero = 182287;

    //print the matrix parameters
    printf("Matrix parameters:\n");
    printf("Matrix size: %d\n", matrice_size);
    printf("Number of nonzero: %d\n", number_of_nonzero);

    double *dense_matrix = (double*)malloc(matrice_size*matrice_size*sizeof(double));
    double *data = (double*)malloc(number_of_nonzero*sizeof(double));
    int *indices = (int*)malloc(number_of_nonzero*sizeof(int));
    int *indptr = (int*)malloc((matrice_size+1)*sizeof(int));
    double *rhs = (double*)malloc(matrice_size*sizeof(double));
    double *dense_matrix_copy = (double*)malloc(matrice_size*matrice_size*sizeof(double));
    double *rhs_copy = (double*)malloc(matrice_size*sizeof(double));



    if(!load_text_vector<double>(path_data, data, number_of_nonzero)){
        printf("Error loading data\n");
        return false;
    }
    if(!load_text_vector<int>(path_indices, indices, number_of_nonzero)){
        printf("Error loading indices\n");
        return false;
    }
    if(!load_text_vector<int>(path_indptr, indptr, matrice_size+1)){
        printf("Error loading indptr\n");
        return false;
    }
    if(!load_text_vector<double>(path_rhs, rhs, matrice_size)){
        printf("Error loading rhs\n");
        return false;
    }

    sparse_to_dense<double>(
        dense_matrix,
        data,
        indices,
        indptr,
        matrice_size);

    //copy dense matrix
    copy_array<double>(dense_matrix, dense_matrix_copy, matrice_size*matrice_size);
    copy_array<double>(rhs, rhs_copy, matrice_size);

    int ipiv[matrice_size];
    int nrhs = 1;
    int info;
    dgesv(&matrice_size, &nrhs, dense_matrix, &matrice_size, ipiv, rhs_copy, &matrice_size, &info);

    if(info != 0){
        printf("Error in MKL dgesv\n");
        printf("info: %d\n", info);
        return false;
    }

    printf("MKL dgesv done\n");

    free(dense_matrix_copy);
    free(rhs_copy);
    free(dense_matrix);
    free(data);
    free(indices);
    free(indptr);
    free(rhs);

    return true;
}