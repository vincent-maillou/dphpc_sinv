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
#include <iostream>
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

int main(int argc, char *argv[]){

    if(argc != 11){
        std::cout << "Usage: ./main" << std::endl <<
        "-nmeas <number of measurements>" << std::endl <<
        "-steps <kmc steps to measure>" << std::endl <<
        "-abstol <absolute tolerance to compare reference>" << std::endl << 
        "-reltol <relative tolerance to compare reference>" << std::endl <<
        "-restol <CG limit for residual>" << std::endl;
        return 0;
    }

    int nmeas = std::stoi(getCmdOption(argv, argv + argc, "-nmeas"));
    if(nmeas < 0){
        std::printf("Number of measurements must be positive\n");
        return 0;
    }

    
    int steps_to_measure = std::stoi(getCmdOption(argv, argv + argc, "-steps"));

    std::printf("Number of measurements: %d\n", nmeas);
    std::printf("KMC steps to measure: %d\n", steps_to_measure);

    double abstol = std::stod(getCmdOption(argv, argv + argc, "-abstol"));
    double reltol = std::stod(getCmdOption(argv, argv + argc, "-reltol"));
    double restol = std::stod(getCmdOption(argv, argv + argc, "-restol"));
    if(abstol < 0){
        std::printf("Absolute tolerance must be positive\n");
        return 0;
    }
    else{
        std::printf("Absolute tolerance: %f\n", abstol);
    }
    if(reltol < 0){
        std::printf("Relative tolerance must be positive\n");
        return 0;
    }
    else{
        std::printf("Relative tolerance: %f\n", reltol);
    }
    if(restol < 0){
        std::printf("Residual tolerance must be positive\n");
        return 0;
    }
    else{
        std::printf("Residual tolerance: %f\n", restol);
    }


    int matrix_size = 7165;
    int number_of_nonzero = 182287;
    double residual_tol_ILU_CG = 5e-18;
    bool flag_verbose = false;
    bool flag_failed = false;


    //print the matrix parameters
    std::printf("Matrix parameters:\n");
    std::printf("Matrix size: %d\n", matrix_size);
    std::printf("Number of nonzero: %d\n", number_of_nonzero);

    double *dense_matrix = (double*)malloc(matrix_size*matrix_size*sizeof(double));
    double *dense_matrix_ref = (double*)malloc(matrix_size*matrix_size*sizeof(double));
    double *data = (double*)malloc(number_of_nonzero*sizeof(double));
    int *indices = (int*)malloc(number_of_nonzero*sizeof(int));
    int *indptr = (int*)malloc((matrix_size+1)*sizeof(int));
    double *rhs = (double*)malloc(matrix_size*sizeof(double));
    double *dense_matrix_copy = (double*)malloc(matrix_size*matrix_size*sizeof(double));
    double *rhs_copy = (double*)malloc(matrix_size*sizeof(double));
    double *reference_solution = (double*)malloc(matrix_size*sizeof(double));
    double *previous_solution = (double*)malloc(matrix_size*sizeof(double));


    int kku = 483;
    int kkl = 483;
    int kkd = 483;

    double *matrix_band_LU = (double*)malloc((2*kku+kkl+1)*matrix_size*sizeof(double));
    double *matrix_band_LU_copy = (double*)malloc((2*kku+kkl+1)*matrix_size*sizeof(double));
    double *matrix_band_CHOL = (double*)malloc((kkd+1)*matrix_size*sizeof(double));
    double *matrix_band_CHOL_copy = (double*)malloc((kkd+1)*matrix_size*sizeof(double));

    bool measurements_correct = true;
    for(int step = 0; step <= steps_to_measure; step++){

        int step_to_measure_previous = step - 1;
        if(step == 0){
            std::printf("No previous step exists\n");
            std::printf("Use solution of step itself\n");
            step_to_measure_previous = 0;
        }


        // Get matrix parameters
        std::string base_path = "/usr/scratch/mont-fort17/almaeder/manasa_kmc_matrices/test_matrices/";
        std::string path_data = base_path + "matrix_sparse_data";
        std::string path_indices = base_path + "matrix_sparse_indices";
        std::string path_indptr = base_path + "matrix_sparse_indptr";
        std::string path_rhs = base_path + "rhs";
        std::string path_reference_solution = base_path + "solution";
        std::string path_reference_solution_previous_step = base_path + "solution";
        std::string path_dense = base_path + "matrix_dense";

        path_data += std::to_string(step) + ".bin";
        path_indices += std::to_string(step) + ".txt";
        path_indptr += std::to_string(step) + ".txt";
        path_rhs += std::to_string(step) + ".bin";
        path_reference_solution += std::to_string(step) + ".bin";
        path_reference_solution_previous_step += std::to_string(step_to_measure_previous) + ".bin";
        path_dense += std::to_string(step) + ".bin";

        std::cout << path_data << std::endl;
        std::cout << path_indices << std::endl;
        std::cout << path_indptr << std::endl;
        std::cout << path_rhs << std::endl;
        std::cout << path_reference_solution << std::endl;
        std::cout << path_reference_solution_previous_step << std::endl;




        if(!load_binary_array<double>(path_data, data, number_of_nonzero)){
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
        if(!load_binary_array<double>(path_rhs, rhs, matrix_size)){
            if(flag_verbose){
                std::printf("Error loading rhs\n");
            }
            flag_failed = true;
        }
        if(!load_binary_array<double>(path_reference_solution, reference_solution, matrix_size)){
            if(flag_verbose){
                std::printf("Error loading reference solution\n");
            }
            flag_failed = true;
        }
        if(!load_binary_array<double>(path_reference_solution_previous_step, previous_solution, matrix_size)){
            if(flag_verbose){
                std::printf("Error loading reference solution previous step\n");
            }
            flag_failed = true;
        }
        if(!load_binary_array<double>(path_dense, dense_matrix_ref, matrix_size*matrix_size)){
            if(flag_verbose){
                std::printf("Error loading dense matrix\n");
            }
            flag_failed = true;
        }

        bool previous = assert_array_magnitude(
            reference_solution,
            previous_solution,
            abstol,
            reltol,
            matrix_size
        );
        if(!previous){
            std::printf("Reference solution and previous solution are not the same\n");
        }
        else{
            std::printf("Reference solution and previous solution are the same\n");
        }

        mkl_set_num_threads(14);
        
        if(!flag_failed){
            bool correct_measurement = true;
            bool reference_correct = true;

            double times_gesv[nmeas];
            double times_posv[nmeas];
            double times_gbsv[nmeas];
            double times_pbsv[nmeas];
            double times_cusparse_CG[nmeas];
            double times_cusparse_ILU_CG[nmeas];
            double times_cusparse_CG_guess[nmeas];
            double times_cusparse_CG_jacobi[nmeas];
            double times_cusparse_CG_jacobi_guess[nmeas];
            double times_cusolver_dense_LU[nmeas];
            double times_cusolver_dense_CHOL[nmeas];
            double times_cusolver_sparse_CHOL[nmeas];
            
            int cg_steps[5];

            sparse_to_dense<double>(
                dense_matrix,
                data,
                indices,
                indptr,
                matrix_size);

            if(!assert_array_elementwise<double>(dense_matrix,
                                        dense_matrix_ref,
                                        abstol,
                                        reltol,
                                        matrix_size*matrix_size)){
                std::printf("Error: dense matrix is not the same as reference\n");
                correct_measurement = false;
                reference_correct = false;
            }
            else{
                std::printf("Dense matrix is the same as reference\n");
            }

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



            for(int i = 0; i < nmeas; i++){
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

            if(!assert_array_elementwise<double>(dense_matrix,
                                        dense_matrix_ref,
                                        abstol,
                                        reltol,
                                        matrix_size*matrix_size)){
                std::printf("Error: dense matrix is not the same as reference\n");
                correct_measurement = false;
                reference_correct = false;
            }
            else{
                std::printf("Dense matrix is the same as reference\n");
            }

            for(int i = 0; i < nmeas; i++){
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


            dense_to_band_for_LU<double>(
                dense_matrix,
                matrix_band_LU,
                matrix_size,
                ku,
                kl);


            for(int i = 0; i < nmeas; i++){
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

            for(int i = 0; i < nmeas; i++){
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


            for(int i = 0; i < nmeas; i++){
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
                    restol,
                    flag_verbose,
                    &cg_steps[0]);
                if(times_cusparse_CG[i] < 0.0){
                    std::printf("Error in cusparse CG\n");
                    correct_measurement = false;
                }
                else{
                    std::printf("Time cusparse CG: %f\n", times_cusparse_CG[i]);
                }

            }


            for(int i = 0; i < nmeas; i++){
                copy_array<double>(rhs, rhs_copy, matrix_size);
                double starting_guess[matrix_size];
                for(int j = 0; j < matrix_size; j++){
                    starting_guess[j] = previous_solution[j];
                }
                times_cusparse_CG_guess[i] = solve_cusparse_CG(
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
                    restol,
                    flag_verbose,
                    &cg_steps[1]);
                if(times_cusparse_CG_guess[i] < 0.0){
                    std::printf("Error in cusparse CG with guess\n");
                    correct_measurement = false;
                }
                else{
                    std::printf("Time cusparse CG with guess: %f\n", times_cusparse_CG_guess[i]);
                }
            }



            for(int i = 0; i < nmeas; i++){
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
                    flag_verbose,
                    &cg_steps[4]);
                if(times_cusparse_ILU_CG[i] < 0.0){
                    std::printf("Error in cusparse ILU CG\n");
                    correct_measurement = false;
                }
                else{
                    std::printf("Time cusparse ILU CG: %f\n", times_cusparse_ILU_CG[i]);
                }

            }


            for(int i = 0; i < nmeas; i++){
                copy_array<double>(rhs, rhs_copy, matrix_size);
                double starting_guess[matrix_size];
                // bit overkill to set zero in every measurement
                for(int j = 0; j < matrix_size; j++){
                    starting_guess[j] = previous_solution[j];
                }
                times_cusparse_CG_jacobi[i] = solve_cusparse_CG_jacobi(
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
                    restol,
                    flag_verbose,
                    &cg_steps[2]);
                if(times_cusparse_CG_jacobi[i] < 0.0){
                    std::printf("Error in cusparse jacobi CG\n");
                    correct_measurement = false;
                }
                else{
                    std::printf("Time cusparse jacobi CG: %f\n", times_cusparse_CG_jacobi[i]);
                }

            }

            for(int i = 0; i < nmeas; i++){
                copy_array<double>(rhs, rhs_copy, matrix_size);
                double starting_guess[matrix_size];
                // bit overkill to set zero in every measurement
                for(int j = 0; j < matrix_size; j++){
                    starting_guess[j] = 0.0;
                }
                times_cusparse_CG_jacobi_guess[i] = solve_cusparse_CG_jacobi(
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
                    restol,
                    flag_verbose,
                    &cg_steps[3]);
                if(times_cusparse_CG_jacobi_guess[i] < 0.0){
                    std::printf("Error in cusparse jacobi CG with guess\n");
                    correct_measurement = false;
                }
                else{
                    std::printf("Time cusparse jacobi CG with guess: %f\n", times_cusparse_CG_jacobi_guess[i]);
                }

            }



            for(int i = 0; i < nmeas; i++){
                copy_array<double>(dense_matrix, dense_matrix_copy, matrix_size*matrix_size);
                copy_array<double>(rhs, rhs_copy, matrix_size);

                times_cusolver_dense_LU[i] = solve_cusolver_dense_LU(
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


            for(int i = 0; i < nmeas; i++){
                copy_array<double>(dense_matrix, dense_matrix_copy, matrix_size*matrix_size);
                copy_array<double>(rhs, rhs_copy, matrix_size);

                times_cusolver_dense_CHOL[i] = solve_cusolver_dense_CHOL(
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


            for(int i = 0; i < nmeas; i++){
                copy_array<double>(rhs, rhs_copy, matrix_size);

                times_cusolver_sparse_CHOL[i] = solve_cusolver_sparse_CHOL(
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
                    std::printf("Error in cusolver sparse CHOL\n");
                    correct_measurement = false;
                }
                else{
                    std::printf("Time cusolver sparse CHOL: %f\n", times_cusolver_sparse_CHOL[i]);
                }

            }

            measurements_correct = measurements_correct && correct_measurement;
            if(!correct_measurement){
                std::printf("Error in one of the measurements\n");
            }
            else{
                std::printf("All measurements correct\n");
            }
            if(!reference_correct){
                std::printf("Error in reference\n");
            }
            else{
                std::printf("Reference correct\n");
            }

            std::ofstream outputFile_steps;
            std::string steps = "steps";
            steps += std::to_string(step);
            steps +=  ".txt";
            outputFile_steps.open(steps);
            if(outputFile_steps.is_open()){
                for(int i = 0; i < 5; i++){
                    outputFile_steps << cg_steps[i] << " ";
                }
                outputFile_steps << '\n';
            }
            else{
                std::printf("Error opening file\n");
            }
            outputFile_steps.close();

            std::ofstream outputFile_times;
            std::string file = "times";
            file += std::to_string(step);
            file +=  ".txt";
            outputFile_times.open(file);

            if(outputFile_times.is_open()){
                for(int i = 0; i < nmeas; i++){
                    outputFile_times << times_gesv[i] << " ";
                }
                outputFile_times << '\n';
                for(int i = 0; i < nmeas; i++){
                    outputFile_times << times_posv[i] << " ";
                }
                outputFile_times << '\n';
                for(int i = 0; i < nmeas; i++){
                    outputFile_times << times_gbsv[i] << " ";
                }
                outputFile_times << '\n';
                for(int i = 0; i < nmeas; i++){
                    outputFile_times << times_pbsv[i] << " ";
                }
                outputFile_times << '\n';
                for(int i = 0; i < nmeas; i++){
                    outputFile_times << times_cusparse_CG[i] << " ";
                }
                outputFile_times << '\n';
                for(int i = 0; i < nmeas; i++){
                    outputFile_times << times_cusparse_CG_guess[i] << " ";
                }
                outputFile_times << '\n';
                for(int i = 0; i < nmeas; i++){
                    outputFile_times << times_cusparse_CG_jacobi[i] << " ";
                }
                outputFile_times << '\n';
                for(int i = 0; i < nmeas; i++){
                    outputFile_times << times_cusparse_CG_jacobi_guess[i] << " ";
                }
                outputFile_times << '\n';
                for(int i = 0; i < nmeas; i++){
                    outputFile_times << times_cusparse_ILU_CG[i] << " ";
                }
                outputFile_times << '\n';
                for(int i = 0; i < nmeas; i++){
                    outputFile_times << times_cusolver_dense_LU[i] << " ";
                }
                outputFile_times << '\n';
                for(int i = 0; i < nmeas; i++){
                    outputFile_times << times_cusolver_dense_CHOL[i] << " ";
                }
                outputFile_times << '\n';
                for(int i = 0; i < nmeas; i++){
                    outputFile_times << times_cusolver_sparse_CHOL[i] << " ";
                }
                outputFile_times << '\n';
            }
            else{
                std::printf("Error opening file\n");
            }
            outputFile_times.close();


        }
    }

    if(!measurements_correct){
        std::printf("Error in one of the measurements\n");
    }
    else{
        std::printf("All measurements correct\n");
    }

    free(matrix_band_LU);
    free(matrix_band_LU_copy);
    free(matrix_band_CHOL);
    free(matrix_band_CHOL_copy);
    free(dense_matrix_copy);
    free(rhs_copy);
    free(dense_matrix);
    free(dense_matrix_ref);
    free(data);
    free(indices);
    free(indptr);
    free(rhs);

    return 0;
}








