#include <string> 

#include "utils.h"
#include "mkl_lapack.h"


// todo I am not sure how to "corretly" compine .cu and .cpp
// what is the "correct" way
bool benchmark_manasa()
{
    // Get matrix parameters
    char path_data[] = "/scratch/sem23f28/manasa_kmc/test_matrices/matrix_sparse_data0.txt";
    char path_indices[] = "/scratch/sem23f28/manasa_kmc/test_matrices/matrix_sparse_indices0.txt";
    char path_indptr[] = "/scratch/sem23f28/manasa_kmc/test_matrices/matrix_sparse_indptr0.txt";
    char path_rhs[] = "/scratch/sem23f28/manasa_kmc/test_matrices/rhs_0.txt";
    int matrice_size = 7165;
    int number_of_nonzero = 182287;

    //print the matrix parameters
    printf("Matrix parameters:\n");
    printf("Matrix size: %d\n", matrice_size);
    printf("Number of nonzero: %d\n", number_of_nonzero);

    double *dense_matrix;
    double *data;
    int *indices;
    int *indptr;
    double *rhs;

    if(!load_text_vector<double>(path_data, &data, number_of_nonzero)){
        printf("Error loading data\n");
        return false;
    }
    if(!load_text_vector<int>(path_indices, &indices, number_of_nonzero)){
        printf("Error loading indices\n");
        return false;
    }
    if(!load_text_vector<int>(path_indptr, &indptr, matrice_size+1)){
        printf("Error loading indptr\n");
        return false;
    }
    if(!load_text_vector<double>(path_rhs, &rhs, matrice_size)){
        printf("Error loading rhs\n");
        return false;
    }

    sparse_to_dense<double>(
        &dense_matrix,
        data,
        indices,
        indptr,
        matrice_size);


    free(dense_matrix);
    free(data);
    free(indices);
    free(indptr);
    free(rhs);
    return true;
}