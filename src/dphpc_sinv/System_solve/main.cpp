/*
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-09

Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
*/

#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <fstream>
#include <string>
#include <mkl.h>

#include <Eigen/Dense>

#include "system_solve_benchmark.h"
#include "utils.h"


char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

int main(int argc, char *argv[]) {

    if(argc != 5){
        std::printf("Usage: ./system_solve_benchmark -nmeas <number of measurements> -step <kmc step to measure>\n");
        return 0;
    }

    int nmeas = std::stoi(getCmdOption(argv, argv + argc, "-nmeas"));
    if(nmeas < 0){
        std::printf("Number of measurements must be positive\n");
        return 0;
    }

    

    int step_to_measure = std::stoi(getCmdOption(argv, argv + argc, "-step"));
    std::printf("Number of measurements: %d\n", nmeas);
    std::printf("KMC step to measure: %d\n", step_to_measure);

    if(step_to_measure == 0){
        std::printf("No previous step exists\n");
        return 0;
    }


    // Get matrix parameters
    char path_data[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/matrix_sparse_data0.txt";
    char path_indices[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/matrix_sparse_indices0.txt";
    char path_indptr[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/matrix_sparse_indptr0.txt";
    char path_rhs[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/rhs_0.txt";
    char path_reference_solution[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/x_ref0.txt";
    
    char path_reference_solution_previous_step[] = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/x_ref0.txt";
    
    int matrix_size = 7165;
    int number_of_nonzero = 182287;
    double abstol = 1e-16;
    double reltol = 1e-14;
    double residual_tol_CG = 5e-18;
    double residual_tol_ILU_CG = 5e-18;
    bool flag_verbose = false;
    bool flag_failed = false;
    int number_measurements = 210;


    //print the matrix parameters
    std::printf("Matrix parameters:\n");
    std::printf("Matrix size: %d\n", matrix_size);
    std::printf("Number of nonzero: %d\n", number_of_nonzero);

    double *dense_matrix = (double*)malloc(matrix_size*matrix_size*sizeof(double));
    double *data = (double*)malloc(number_of_nonzero*sizeof(double));
    int *indices = (int*)malloc(number_of_nonzero*sizeof(int));
    int *indptr = (int*)malloc((matrix_size+1)*sizeof(int));
    double *rhs = (double*)malloc(matrix_size*sizeof(double));
    double *dense_matrix_copy = (double*)malloc(matrix_size*matrix_size*sizeof(double));
    double *rhs_copy = (double*)malloc(matrix_size*sizeof(double));
    double *reference_solution = (double*)malloc(matrix_size*sizeof(double));
    


    if(!load_text_array<double>(path_data, data, number_of_nonzero)){
        if(flag_verbose){
            std::printf("Error loading data\n");
        }
        flag_failed = true;
    }
    if(!load_text_array<int>(path_indices, indices, number_of_nonzero)){
        if(flag_verbose){
            std::printf("Error loading indices\n");
        }
        flag_failed = true;
    }
    if(!load_text_array<int>(path_indptr, indptr, matrix_size+1)){
        if(flag_verbose){
            std::printf("Error loading indptr\n");
        }
        flag_failed = true;
    }
    if(!load_text_array<double>(path_rhs, rhs, matrix_size)){
        if(flag_verbose){
            std::printf("Error loading rhs\n");
        }
        flag_failed = true;
    }
    if(!load_text_array<double>(path_reference_solution, reference_solution, matrix_size)){
        if(flag_verbose){
            std::printf("Error loading reference solution\n");
        }
        flag_failed = true;
    }




    mkl_set_num_threads(14);
    if(!flag_failed){

        bool correct_measurement = true;

        double times_gesv[number_measurements];
        double times_posv[number_measurements];
        double times_gbsv[number_measurements];
        double times_pbsv[number_measurements];
        double times_cusparse_CG[number_measurements];
        double times_cusparse_ILU_CG[number_measurements];
        double times_cusparse_CG_guess[number_measurements];
        double times_cusolver_dense_LU[number_measurements];
        double times_cusolver_dense_CHOL[number_measurements];
        double times_cusolver_sparse_CHOL[number_measurements];
        



        sparse_to_dense<double>(
            dense_matrix,
            data,
            indices,
            indptr,
            matrix_size);

        int ku = 0;
        int kl = 0;
        calc_bandwidth<double>(
            dense_matrix,
            matrix_size,
            &ku,
            &kl);
        if(ku != kl && !assert_symmetric<double>(dense_matrix, matrix_size, abstol, reltol)){
            std::printf("Error: matrix is not symmetric\n");
            correct_measurement = false;
        }
        else{
            std::printf("Matrix is symmetric\n");
        }

        
        int kd = ku;
        std::printf("Upper Bandwidth: %d\n", ku);
        std::printf("Lower Bandwidth: %d\n", kl);

        double *matrix_band_LU = (double*)malloc((2*ku+kl+1)*matrix_size*sizeof(double));
        double *matrix_band_LU_copy = (double*)malloc((2*ku+kl+1)*matrix_size*sizeof(double));
        double *matrix_band_CHOL = (double*)malloc((kd+1)*matrix_size*sizeof(double));
        double *matrix_band_CHOL_copy = (double*)malloc((kd+1)*matrix_size*sizeof(double));


        dense_to_band_for_LU<double>(
            dense_matrix,
            matrix_band_LU,
            matrix_size,
            ku,
            kl);


        for(int i = 0; i < number_measurements; i++){
            copy_array<double>(matrix_band_LU, matrix_band_LU_copy, matrix_size*(2*ku+kl+1));
            copy_array<double>(rhs, rhs_copy, matrix_size);
            times_gbsv[i] = solve_mkl_dgbsv(
                matrix_band_LU_copy,
                rhs_copy,
                reference_solution,
                matrix_size,
                ku,
                kl,
                abstol,
                reltol,
                flag_verbose);
            if(times_gbsv[i] < 0.0){
                std::printf("Error in MKL gbsv\n");
                correct_measurement = false;
            }
            else{
                std::printf("Time MKL gbsv: %f\n", times_gbsv[i]);
            }
        }



        dense_to_band_for_U_CHOL<double>(
            dense_matrix,
            matrix_band_CHOL,
            matrix_size,
            kd);

        for(int i = 0; i < number_measurements; i++){
            copy_array<double>(matrix_band_CHOL, matrix_band_CHOL_copy, matrix_size*(kd+1));
            copy_array<double>(rhs, rhs_copy, matrix_size);
            times_pbsv[i] = solve_mkl_dpbsv(
                matrix_band_CHOL_copy,
                rhs_copy,
                reference_solution,
                matrix_size,
                kd,
                abstol,
                reltol,
                flag_verbose);
            if(times_pbsv[i] < 0.0){
                std::printf("Error in MKL gbsv\n");
                correct_measurement = false;
            }
            else{
                std::printf("Time MKL pbsv: %f\n", times_pbsv[i]);
            }
        }

        for(int i = 0; i < number_measurements; i++){
            copy_array<double>(dense_matrix, dense_matrix_copy, matrix_size*matrix_size);
            copy_array<double>(rhs, rhs_copy, matrix_size);

            times_gesv[i] = solve_mkl_dgesv(
                dense_matrix_copy,
                rhs_copy,
                reference_solution,
                matrix_size,
                abstol,
                reltol,
                flag_verbose);
            if(times_gesv[i] < 0.0){
                std::printf("Error in MKL dgesv\n");
                correct_measurement = false;
            }
            else{
                std::printf("Time MKL dgesv: %f\n", times_gesv[i]);
            }
        }


        for(int i = 0; i < number_measurements; i++){
            copy_array<double>(dense_matrix, dense_matrix_copy, matrix_size*matrix_size);
            copy_array<double>(rhs, rhs_copy, matrix_size);

            times_posv[i] = solve_mkl_dposv(
                dense_matrix_copy,
                rhs_copy,
                reference_solution,
                matrix_size,
                abstol,
                reltol,
                flag_verbose);
            if(times_posv[i] < 0.0){
                std::printf("Error in MKL dposv\n");
                correct_measurement = false;
            }
            else{
                std::printf("Time MKL dposv: %f\n", times_posv[i]);
            }
        }

        for(int i = 0; i < number_measurements; i++){
            copy_array<double>(rhs, rhs_copy, matrix_size);
            double starting_guess[matrix_size];
            // bit overkill to set zero in every measurement
            for(int j = 0; j < matrix_size; j++){
                starting_guess[j] = 0.0;
            }
            times_cusparse_CG[i] = solve_cusparse_CG(
                data,
                indices,
                indptr,
                rhs_copy,
                reference_solution,
                starting_guess,
                number_of_nonzero,
                matrix_size,
                abstol,
                reltol,
                residual_tol_CG,
                flag_verbose);
            if(times_cusparse_CG[i] < 0.0){
                std::printf("Error in cusparse CG\n");
                correct_measurement = false;
            }
            else{
                std::printf("Time cusparse CG: %f\n", times_cusparse_CG[i]);
            }

        }

        for(int i = 0; i < number_measurements; i++){
            copy_array<double>(rhs, rhs_copy, matrix_size);

            times_cusparse_ILU_CG[i] = solve_cusparse_ILU_CG(
                data,
                indices,
                indptr,
                rhs_copy,
                reference_solution,
                number_of_nonzero,
                matrix_size,
                abstol,
                reltol,
                residual_tol_ILU_CG,
                flag_verbose);
            if(times_cusparse_ILU_CG[i] < 0.0){
                std::printf("Error in cusparse ILU CG\n");
                correct_measurement = false;
            }
            else{
                std::printf("Time cusparse ILU CG: %f\n", times_cusparse_ILU_CG[i]);
            }

        }

        for(int i = 0; i < number_measurements; i++){
            copy_array<double>(dense_matrix, dense_matrix_copy, matrix_size*matrix_size);
            copy_array<double>(rhs, rhs_copy, matrix_size);

            times_cusolver_dense_LU[i] = solve_cusolver_LU(
                dense_matrix_copy,
                rhs_copy,
                reference_solution,
                matrix_size,
                abstol,
                reltol,
                flag_verbose);
            if(times_cusolver_dense_LU[i] < 0.0){
                std::printf("Error in cusolver dense LU\n");
                correct_measurement = false;
            }
            else{
                std::printf("Time cusolver dense LU: %f\n", times_cusolver_dense_LU[i]);
            }

        }


        for(int i = 0; i < number_measurements; i++){
            copy_array<double>(dense_matrix, dense_matrix_copy, matrix_size*matrix_size);
            copy_array<double>(rhs, rhs_copy, matrix_size);

            times_cusolver_dense_CHOL[i] = solve_cusolver_CHOL(
                dense_matrix_copy,
                rhs_copy,
                reference_solution,
                matrix_size,
                abstol,
                reltol,
                flag_verbose);
            if(times_cusolver_dense_CHOL[i] < 0.0){
                std::printf("Error in cusolver dense CHOL\n");
                correct_measurement = false;
            }
            else{
                std::printf("Time cusolver dense CHOL: %f\n", times_cusolver_dense_CHOL[i]);
            }

        }


        for(int i = 0; i < number_measurements; i++){
            copy_array<double>(rhs, rhs_copy, matrix_size);

            times_cusolver_sparse_CHOL[i] = solve_cusolver_CHOL(
                data,
                indices,
                indptr,
                rhs_copy,
                reference_solution,
                number_of_nonzero,
                matrix_size,
                abstol,
                reltol,
                flag_verbose);
            if(times_cusolver_sparse_CHOL[i] < 0.0){
                std::printf("Error in cusolver sparse\n");
                correct_measurement = false;
            }
            else{
                std::printf("Time cusolver sparse: %f\n", times_cusolver_sparse_CHOL[i]);
            }

            if(!correct_measurement){
                std::printf("Error in one of the measurements\n");
            }
            else{
                std::printf("All measurements correct\n");
            }

        }


        std::ofstream outputFile_times;
        outputFile_times.open("times.txt");

        if(outputFile_times.is_open()){
            for(int i = 0; i < number_measurements; i++){
                outputFile_times << times_gesv[i] << " ";
            }
            outputFile_times << '\n';
            for(int i = 0; i < number_measurements; i++){
                outputFile_times << times_posv[i] << " ";
            }
            outputFile_times << '\n';
            for(int i = 0; i < number_measurements; i++){
                outputFile_times << times_gbsv[i] << " ";
            }
            outputFile_times << '\n';
            for(int i = 0; i < number_measurements; i++){
                outputFile_times << times_pbsv[i] << " ";
            }
            outputFile_times << '\n';
            for(int i = 0; i < number_measurements; i++){
                outputFile_times << times_cusparse_CG[i] << " ";
            }
            outputFile_times << '\n';
            for(int i = 0; i < number_measurements; i++){
                outputFile_times << times_cusparse_ILU_CG[i] << " ";
            }
            outputFile_times << '\n';
            for(int i = 0; i < number_measurements; i++){
                outputFile_times << times_cusolver_dense_LU[i] << " ";
            }
            outputFile_times << '\n';
            for(int i = 0; i < number_measurements; i++){
                outputFile_times << times_cusolver_dense_CHOL[i] << " ";
            }
            outputFile_times << '\n';
            for(int i = 0; i < number_measurements; i++){
                outputFile_times << times_cusolver_sparse_CHOL[i] << " ";
            }
            outputFile_times << '\n';
        }
        else{
            std::printf("Error opening file\n");
        }


        free(matrix_band_LU);
        free(matrix_band_LU_copy);
        free(matrix_band_CHOL);
        free(matrix_band_CHOL_copy);
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








