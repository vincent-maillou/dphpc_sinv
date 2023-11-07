/*
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-09

Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
*/

#include <stdio.h>
#include <stdlib.h>
#include <complex>

#include <Eigen/Dense>

#include "utils.h"
#include "system_solve_benchmark.h"

int main() {
    // Get matrix parameters
    char path_data[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/matrix_sparse_data0.txt";
    char path_indices[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/matrix_sparse_indices0.txt";
    char path_indptr[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/matrix_sparse_indptr0.txt";
    char path_rhs[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/rhs_0.txt";
    char path_reference_solution[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/x_ref0.txt";
    int matrice_size = 7165;
    int number_of_nonzero = 182287;
    double tolerance = 1e-10;
    bool flag_verbose = true;
    bool flag_failed = false;



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
    double *reference_solution = (double*)malloc(matrice_size*sizeof(double));


    if(!load_text_array<double>(path_data, data, number_of_nonzero)){
        if(flag_verbose){
            printf("Error loading data\n");
        }
        flag_failed = true;
    }
    if(!load_text_array<int>(path_indices, indices, number_of_nonzero)){
        if(flag_verbose){
            printf("Error loading indices\n");
        }
        flag_failed = true;
    }
    if(!load_text_array<int>(path_indptr, indptr, matrice_size+1)){
        if(flag_verbose){
            printf("Error loading indptr\n");
        }
        flag_failed = true;
    }
    if(!load_text_array<double>(path_rhs, rhs, matrice_size)){
        if(flag_verbose){
            printf("Error loading rhs\n");
        }
        flag_failed = true;
    }
    if(!load_text_array<double>(path_reference_solution, reference_solution, matrice_size)){
        if(flag_verbose){
            printf("Error loading reference solution\n");
        }
        flag_failed = true;
    }

    if(!flag_failed){
        sparse_to_dense<double>(
            dense_matrix,
            data,
            indices,
            indptr,
            matrice_size);


        // char path_save[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test0.txt";
        // if(!save_text_array<double>(path_save, dense_matrix, matrice_size*matrice_size)){
        //     printf("Error saving dense matrix\n");
        //     return false;
        // }

        //copy dense matrix
        // copy_array<double>(dense_matrix, dense_matrix_copy, matrice_size*matrice_size);
        // copy_array<double>(rhs, rhs_copy, matrice_size);

        // double time_mkl_dense = solve_mkl_dgesv(
        //     dense_matrix_copy,
        //     rhs_copy,
        //     reference_solution,
        //     matrice_size,
        //     tolerance,
        //     flag_verbose);
        // if(time_mkl_dense < 0.0){
        //     return false;
        // }
        // else{
        //     printf("Time MKL dgesv: %f\n s", time_mkl_dense);
        // }


        copy_array<double>(rhs, rhs_copy, matrice_size);

        double time_cusparse_CG = solve_cusparse_CG(
            data,
            indices,
            indptr,
            rhs_copy,
            reference_solution,
            number_of_nonzero,
            matrice_size,
            tolerance,
            flag_verbose);

        if((time_cusparse_CG < 0.0) && flag_verbose){
            printf("Error in cusparse CG\n");
        }
        else if (flag_verbose){
            printf("Time cusparse CG: %f\n s", time_cusparse_CG);
        }

        //copy dense matrix
        copy_array<double>(dense_matrix, dense_matrix_copy, matrice_size*matrice_size);
        copy_array<double>(rhs, rhs_copy, matrice_size);

        double time_cusolve_dense = solve_cusolver_LU(
            dense_matrix_copy,
            rhs_copy,
            reference_solution,
            matrice_size,
            tolerance,
            flag_verbose);
        if((time_cusolve_dense < 0.0) && flag_verbose){
            printf("Error in cusolver dense\n");
        }
        else if (flag_verbose){
            printf("Time cusolver dense: %f\n s", time_cusolve_dense);
        }

        // char path_save[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/x0.txt";
        // if(!save_text_array<double>(path_save, rhs_copy, matrice_size)){
        //     printf("Error saving dense matrix\n");
        //     return false;
        // }

    }
    free(dense_matrix_copy);
    free(rhs_copy);
    free(dense_matrix);
    free(data);
    free(indices);
    free(indptr);
    free(rhs);

    return 0;
}








