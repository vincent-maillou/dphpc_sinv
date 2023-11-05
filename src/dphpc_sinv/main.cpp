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
    // // Get matrix parameters
    // char f_matparam[] = "../../tests/tests_cases/mat_parameters_0.txt";
    // unsigned int matrice_size;
    // unsigned int blocksize;

    // load_matrix_parameters(f_matparam, &matrice_size, &blocksize);

    // unsigned int n_blocks = matrice_size / blocksize;

    // // Print the matrix parameters
    // printf("Matrix parameters:\n");
    // printf("    Matrix size: %d\n", matrice_size);
    // printf("    Block size: %d\n", blocksize);


    // // Load matrix to invert
    // std::complex<double>* matrix_diagblk = (std::complex<double>*) malloc(blocksize * matrice_size * sizeof(std::complex<double>));
    // char f_mat_diagblk[] = "../../tests/tests_cases/matrix_0_diagblk.bin";
    // load_binary_matrix(f_mat_diagblk, matrix_diagblk, blocksize, matrice_size);

    // std::complex<double>* matrix_upperblk = (std::complex<double>*) malloc(blocksize * (matrice_size-blocksize) * sizeof(std::complex<double>));
    // char f_mat_upperblk[] = "../../tests/tests_cases/matrix_0_upperblk.bin";
    // load_binary_matrix(f_mat_upperblk, matrix_upperblk, blocksize, matrice_size-blocksize);

    // std::complex<double>* matrix_lowerblk = (std::complex<double>*) malloc(blocksize * (matrice_size-blocksize) * sizeof(std::complex<double>));
    // char f_mat_lowerblk[] = "../../tests/tests_cases/matrix_0_lowerblk.bin";
    // load_binary_matrix(f_mat_lowerblk, matrix_lowerblk, blocksize, matrice_size-blocksize);


    // // Load reference solution of the matrix inverse
    // std::complex<double>* matrix_inv_diagblk = (std::complex<double>*) malloc(blocksize * matrice_size * sizeof(std::complex<double>));
    // char f_mat_inv_diagblk[] = "../../tests/tests_cases/matrix_0_inverse_diagblk.bin";
    // load_binary_matrix(f_mat_inv_diagblk, matrix_inv_diagblk, blocksize, matrice_size);

    // std::complex<double>* matrix_inv_upperblk = (std::complex<double>*) malloc(blocksize * (matrice_size-blocksize) * sizeof(std::complex<double>));
    // char f_mat_inv_upperblk[] = "../../tests/tests_cases/matrix_0_inverse_upperblk.bin";
    // load_binary_matrix(f_mat_inv_upperblk, matrix_inv_upperblk, blocksize, matrice_size-blocksize);
    
    // std::complex<double>* matrix_inv_lowerblk = (std::complex<double>*) malloc(blocksize * (matrice_size-blocksize) * sizeof(std::complex<double>));
    // char f_mat_inv_lowerblk[] = "../../tests/tests_cases/matrix_0_inverse_lowerblk.bin";
    // load_binary_matrix(f_mat_inv_lowerblk, matrix_inv_lowerblk, blocksize, matrice_size-blocksize);


    // // ----- END OF INIT SECTION -----

    // // Transform C style storage to vector of eigen matrices
    // std::vector<Eigen::MatrixXcd> eig_diagblk;
    // std::vector<Eigen::MatrixXcd> eig_upperblk;
    // std::vector<Eigen::MatrixXcd> eig_lowerblk;

    // for (unsigned int i = 0; i < n_blocks; ++i) {
    //     eig_diagblk.push_back(Eigen::Map<Eigen::MatrixXcd>(reinterpret_cast<std::complex<double>*>(matrix_diagblk + i*blocksize*blocksize), blocksize, blocksize));
    //     if(i < n_blocks-1){
    //         eig_upperblk.push_back(Eigen::Map<Eigen::MatrixXcd>(reinterpret_cast<std::complex<double>*>(matrix_upperblk + i*blocksize*(matrice_size-blocksize)), blocksize, blocksize));
    //         eig_lowerblk.push_back(Eigen::Map<Eigen::MatrixXcd>(reinterpret_cast<std::complex<double>*>(matrix_lowerblk + i*blocksize*(matrice_size-blocksize)), blocksize, blocksize));
    //     }
    // }

    // // Pre-allocate memory for the inverted blocks
    // std::vector<Eigen::MatrixXcd> eig_inv_diagblk(n_blocks);
    // std::vector<Eigen::MatrixXcd> eig_inv_upperblk(n_blocks-1);
    // std::vector<Eigen::MatrixXcd> eig_inv_lowerblk(n_blocks-1);

    // // Resize the inverted blocks to match the size of the original blocks
    // for (unsigned int i = 0; i < n_blocks; i++) {
    //     eig_inv_diagblk[i].resize(blocksize, blocksize);
    //     if(i < n_blocks-1){
    //         eig_inv_upperblk[i].resize(blocksize, matrice_size-blocksize);
    //         eig_inv_lowerblk[i].resize(blocksize, matrice_size-blocksize);
    //     }
    // }

    // // 0. Inverse of the first block
    // eig_inv_diagblk[0] = eig_diagblk[0].inverse();

    // // 1. Forward substitution (performed left to right)
    // for (unsigned int i = 1; i < n_blocks; ++i) {
    //     eig_inv_diagblk[i] = eig_diagblk[i] - eig_lowerblk[i-1] * eig_inv_diagblk[i-1] * eig_upperblk[i-1];
    // }

    // // 2. Backward substitution (performed right to left)
    // for(int i = n_blocks-2; i >= 0; --i){
    //     Eigen::MatrixXcd tmp_lowerfactor = eig_inv_diagblk[i+1] * eig_lowerblk[i] * eig_inv_diagblk[i];

    //     eig_inv_lowerblk[i] = -tmp_lowerfactor;
    //     eig_inv_upperblk[i] = -eig_inv_diagblk[i] * eig_upperblk[i] * eig_inv_diagblk[i+1];

    //     eig_inv_diagblk[i] += eig_inv_diagblk[i] * eig_upperblk[i] * tmp_lowerfactor;
    // }


    // // Store results back into C style storage
    // for(unsigned int i = 0; i < n_blocks; ++i){
    //     memcpy(matrix_diagblk + i*blocksize*blocksize, eig_inv_diagblk[i].data(), blocksize*blocksize*sizeof(std::complex<double>));
    //     if(i < n_blocks-1){
    //         memcpy(matrix_upperblk + i*blocksize*(matrice_size-blocksize), eig_inv_upperblk[i].data(), blocksize*(matrice_size-blocksize)*sizeof(std::complex<double>));
    //         memcpy(matrix_lowerblk + i*blocksize*(matrice_size-blocksize), eig_inv_lowerblk[i].data(), blocksize*(matrice_size-blocksize)*sizeof(std::complex<double>));
    //     }
    // }



    // // ----- RESULT CHECKING SECTION -----

    // if(!are_equals(matrix_diagblk, matrix_inv_diagblk, matrice_size, blocksize)){
    //     printf("Error: matrix_diagblk and matrix_inv_diagblk are not equal\n");

    //     print_matrix(matrix_diagblk, blocksize, blocksize);
    //     print_matrix(matrix_inv_diagblk, blocksize, blocksize);
    // }

    // if(!are_equals(matrix_upperblk, matrix_inv_upperblk, matrice_size, blocksize)){
    //     printf("Error: matrix_upperblk and matrix_inv_upperblk are not equal\n");
    // }

    // if(!are_equals(matrix_lowerblk, matrix_inv_lowerblk, matrice_size, blocksize)){
    //     printf("Error: matrix_lowerblk and matrix_inv_lowerblk are not equal\n");
    // }
    // // removed return 1
    // // else we have a memory leak

    // // ----- CLEANING SECTION -----

    // free_matrix(matrix_diagblk);
    // free_matrix(matrix_upperblk);
    // free_matrix(matrix_lowerblk);
    // free_matrix(matrix_inv_diagblk);
    // free_matrix(matrix_inv_upperblk);
    // free_matrix(matrix_inv_lowerblk);

    if(!benchmark_manasa()){
        printf("Error: benchmark_manasa failed\n");
    }

    return 0;
}








