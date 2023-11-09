/*
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-09

Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
*/

#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <iostream>

#include <Eigen/Dense>

#include "utils.h"

int main() {
    // Get matrix parameters
    char f_matparam[] = "../../../tests/tests_cases/mat_parameters_0.txt";
    unsigned int matrix_size;
    unsigned int blocksize;
    bool verif = false;

    load_matrix_parameters(f_matparam, &matrix_size, &blocksize);

    unsigned int n_blocks = matrix_size / blocksize;
    unsigned int off_diag_size = matrix_size - blocksize;

    // Print the matrix parameters
    printf("Matrix parameters:\n");
    printf("    Matrix size: %d\n", matrix_size);
    printf("    Block size: %d\n", blocksize);


    // Load matrix to invert
    std::complex<double>* matrix_diagblk = (std::complex<double>*) malloc(blocksize * matrix_size * sizeof(std::complex<double>));
    char f_mat_diagblk[] = "../../../tests/tests_cases/matrix_0_diagblk.bin";
    load_binary_matrix(f_mat_diagblk, matrix_diagblk, blocksize, matrix_size);

    std::complex<double>* matrix_upperblk = (std::complex<double>*) malloc(blocksize * (off_diag_size) * sizeof(std::complex<double>));
    char f_mat_upperblk[] = "../../../tests/tests_cases/matrix_0_upperblk.bin";
    load_binary_matrix(f_mat_upperblk, matrix_upperblk, blocksize, off_diag_size);

    std::complex<double>* matrix_lowerblk = (std::complex<double>*) malloc(blocksize * (off_diag_size) * sizeof(std::complex<double>));
    char f_mat_lowerblk[] = "../../../tests/tests_cases/matrix_0_lowerblk.bin";
    load_binary_matrix(f_mat_lowerblk, matrix_lowerblk, blocksize, off_diag_size);


    if(verif){
        for(unsigned int i = 0; i < blocksize; i++){
            for(unsigned int j = 0; j < matrix_size; j++){
                std::cout << "matrix_diagblk[" << i << "][" << j << "] = " << matrix_diagblk[i*matrix_size+j] << std::endl;
            }
        }

        for(unsigned int i = 0; i < blocksize; i++){
            for(unsigned int j = 0; j < off_diag_size; j++){
                std::cout << "matrix_upperblk[" << i << "][" << j << "] = " << matrix_upperblk[i*off_diag_size+j] << std::endl;
            }
        }

        for(unsigned int i = 0; i < blocksize; i++){
            for(unsigned int j = 0; j < off_diag_size; j++){
                std::cout << "matrix_lowerblk[" << i << "][" << j << "] = " << matrix_lowerblk[i*off_diag_size+j] << std::endl;
            }
        }
    }


    // ----- END OF INIT SECTION -----

    // Transform C style storage to vector of eigen matrices
    std::vector<Eigen::MatrixXcd> eig_diagblk;
    std::vector<Eigen::MatrixXcd> eig_upperblk;
    std::vector<Eigen::MatrixXcd> eig_lowerblk;

    for (unsigned int i = 0; i < n_blocks; ++i) {
        eig_diagblk.push_back(Eigen::Map<Eigen::MatrixXcd, 0, Eigen::Stride<1, 10>>
            (reinterpret_cast<std::complex<double>*>(matrix_diagblk + i*blocksize),
            blocksize, blocksize));

        if(i < n_blocks-1){
            eig_upperblk.push_back(Eigen::Map<Eigen::MatrixXcd, 0, Eigen::Stride<1, 8>>
                (reinterpret_cast<std::complex<double>*>(matrix_upperblk + i*blocksize), blocksize, blocksize));

            eig_lowerblk.push_back(Eigen::Map<Eigen::MatrixXcd, 0, Eigen::Stride<1, 8>>
                (reinterpret_cast<std::complex<double>*>(matrix_lowerblk + i*blocksize), blocksize, blocksize));
        }
    }

    // print eigen matrices
    if(verif){
        for (unsigned int i = 0; i < n_blocks; ++i) {
            std::cout << "eig_diagblk[" << i << "] = " << std::endl << eig_diagblk[i] << std::endl << std::endl;
            if(i < n_blocks-1){
                std::cout << "eig_upperblk[" << i << "] = " << std::endl << eig_upperblk[i] << std::endl << std::endl;
                std::cout << "eig_lowerblk[" << i << "] = " << std::endl << eig_lowerblk[i] << std::endl << std::endl;
            }
        }
    }


    /* // Pre-allocate memory for the inverted blocks
    std::vector<Eigen::MatrixXcd> eig_inv_diagblk(n_blocks);
    std::vector<Eigen::MatrixXcd> eig_inv_upperblk(n_blocks-1);
    std::vector<Eigen::MatrixXcd> eig_inv_lowerblk(n_blocks-1);

    // Resize the inverted blocks to match the size of the original blocks
    for (unsigned int i = 0; i < n_blocks; i++) {
        eig_inv_diagblk[i].resize(blocksize, blocksize);
        if(i < n_blocks-1){
            eig_inv_upperblk[i].resize(blocksize, off_diag_size);
            eig_inv_lowerblk[i].resize(blocksize, off_diag_size);
        }
    }

    

    // 0. Inverse of the first block
    eig_inv_diagblk[0] = eig_diagblk[0].inverse();

    // 1. Forward substitution (performed left to right)
    for (unsigned int i = 1; i < n_blocks; ++i) {
        eig_inv_diagblk[i] = (eig_diagblk[i] - eig_lowerblk[i-1] * eig_inv_diagblk[i-1] * eig_upperblk[i-1]).inverse();
    }

    // 2. Backward substitution (performed right to left)
    for(int i = n_blocks-2; i >= 0; --i){
        Eigen::MatrixXcd tmp_lowerfactor = eig_inv_diagblk[i+1] * eig_lowerblk[i] * eig_inv_diagblk[i];

        eig_inv_lowerblk[i] = -tmp_lowerfactor;
        eig_inv_upperblk[i] = -eig_inv_diagblk[i] * eig_upperblk[i] * eig_inv_diagblk[i+1];

        eig_inv_diagblk[i] += eig_inv_diagblk[i] * eig_upperblk[i] * tmp_lowerfactor;
    } */







    // Pre-allocate memory for the inverted blocks
    std::vector<Eigen::MatrixXcd> eig_inv_diagblk(n_blocks);
    std::vector<Eigen::MatrixXcd> eig_inv_upperblk(n_blocks-1);
    std::vector<Eigen::MatrixXcd> eig_inv_lowerblk(n_blocks-1);

    // 0. Inverse of the first block
    eig_inv_diagblk[0] = eig_diagblk[0].inverse();

    // 1. Forward substitution (performed left to right)
    for (unsigned int i = 1; i < n_blocks; ++i) {
        eig_inv_diagblk[i] = (eig_diagblk[i] - eig_lowerblk[i-1] * eig_inv_diagblk[i-1] * eig_upperblk[i-1]).inverse();
    }

    // 2. Backward substitution (performed right to left)
    for(int i = n_blocks-2; i >= 0; --i){
        Eigen::MatrixXcd tmp_lowerfactor = eig_inv_diagblk[i+1] * eig_lowerblk[i] * eig_inv_diagblk[i];

        eig_inv_lowerblk[i] = -tmp_lowerfactor;
        eig_inv_upperblk[i] = -eig_inv_diagblk[i] * eig_upperblk[i] * eig_inv_diagblk[i+1];

        eig_inv_diagblk[i] += eig_inv_diagblk[i] * eig_upperblk[i] * tmp_lowerfactor;
    }







    /* // Store results back into C style storage
    for(unsigned int i = 0; i < n_blocks; ++i){
        memcpy(matrix_diagblk + i*blocksize*blocksize, eig_inv_diagblk[i].data(), blocksize*blocksize*sizeof(std::complex<double>));
        if(i < n_blocks-1){
            memcpy(matrix_upperblk + i*blocksize*(off_diag_size), eig_inv_upperblk[i].data(), blocksize*(off_diag_size)*sizeof(std::complex<double>));
            memcpy(matrix_lowerblk + i*blocksize*(off_diag_size), eig_inv_lowerblk[i].data(), blocksize*(off_diag_size)*sizeof(std::complex<double>));
        }
    } */



    // ----- RESULT CHECKING SECTION -----

    // Load reference solution of the matrix inverse
    /* std::complex<double>* matrix_inv_diagblk = (std::complex<double>*) malloc(blocksize * matrix_size * sizeof(std::complex<double>));
    char f_mat_inv_diagblk[] = "../../../tests/tests_cases/matrix_0_inverse_diagblk.bin";
    load_binary_matrix(f_mat_inv_diagblk, matrix_inv_diagblk, blocksize, matrix_size);

    std::complex<double>* matrix_inv_upperblk = (std::complex<double>*) malloc(blocksize * (off_diag_size) * sizeof(std::complex<double>));
    char f_mat_inv_upperblk[] = "../../../tests/tests_cases/matrix_0_inverse_upperblk.bin";
    load_binary_matrix(f_mat_inv_upperblk, matrix_inv_upperblk, blocksize, off_diag_size);
    
    std::complex<double>* matrix_inv_lowerblk = (std::complex<double>*) malloc(blocksize * (off_diag_size) * sizeof(std::complex<double>));
    char f_mat_inv_lowerblk[] = "../../../tests/tests_cases/matrix_0_inverse_lowerblk.bin";
    load_binary_matrix(f_mat_inv_lowerblk, matrix_inv_lowerblk, blocksize, off_diag_size); */


    /* if(!are_equals(matrix_diagblk, matrix_inv_diagblk, matrix_size, blocksize)){
        printf("Error: matrix_diagblk and matrix_inv_diagblk are not equal\n");

        print_matrix(matrix_diagblk, blocksize, blocksize);
        print_matrix(matrix_inv_diagblk, blocksize, blocksize);
    }
    else{
        printf("matrix_diagblk and matrix_inv_diagblk are equal\n");
    }

    if(!are_equals(matrix_upperblk, matrix_inv_upperblk, matrix_size, blocksize)){
        printf("Error: matrix_upperblk and matrix_inv_upperblk are not equal\n");
    }
    else{
        printf("matrix_upperblk and matrix_inv_upperblk are equal\n");
    }

    if(!are_equals(matrix_lowerblk, matrix_inv_lowerblk, matrix_size, blocksize)){
        printf("Error: matrix_lowerblk and matrix_inv_lowerblk are not equal\n");
    }
    else{
        printf("matrix_lowerblk and matrix_inv_lowerblk are equal\n");
    } */


    // removed return 1
    // else we have a memory leak

    // ----- CLEANING SECTION -----

    /*free_matrix(matrix_diagblk);
    free_matrix(matrix_upperblk);
    free_matrix(matrix_lowerblk);
    free_matrix(matrix_inv_diagblk);
    free_matrix(matrix_inv_upperblk);
    free_matrix(matrix_inv_lowerblk);*/


    return 0;
}








