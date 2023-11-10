// Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.

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

    // ----- END OF INIT SECTION -----



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



    // ----- RESULT CHECKING SECTION -----

    // Load reference solution of the matrix inverse
    std::complex<double>* matrix_inv_diagblk = (std::complex<double>*) malloc(blocksize * matrix_size * sizeof(std::complex<double>));
    char f_mat_inv_diagblk[] = "../../../tests/tests_cases/matrix_0_inverse_diagblk.bin";
    load_binary_matrix(f_mat_inv_diagblk, matrix_inv_diagblk, blocksize, matrix_size);

    std::complex<double>* matrix_inv_upperblk = (std::complex<double>*) malloc(blocksize * (off_diag_size) * sizeof(std::complex<double>));
    char f_mat_inv_upperblk[] = "../../../tests/tests_cases/matrix_0_inverse_upperblk.bin";
    load_binary_matrix(f_mat_inv_upperblk, matrix_inv_upperblk, blocksize, off_diag_size);
    
    std::complex<double>* matrix_inv_lowerblk = (std::complex<double>*) malloc(blocksize * (off_diag_size) * sizeof(std::complex<double>));
    char f_mat_inv_lowerblk[] = "../../../tests/tests_cases/matrix_0_inverse_lowerblk.bin";
    load_binary_matrix(f_mat_inv_lowerblk, matrix_inv_lowerblk, blocksize, off_diag_size);


    // Convert back std::vector<Eigen::Matrix> to std::vector<std::complex<double>>
    std::complex<double>* eigen_inverted_diagblk = (std::complex<double>*) malloc(blocksize * matrix_size * sizeof(std::complex<double>));
    std::complex<double>* eigen_inverted_upperblk = (std::complex<double>*) malloc(blocksize * off_diag_size * sizeof(std::complex<double>));
    std::complex<double>* eigen_inverted_lowerblk = (std::complex<double>*) malloc(blocksize * off_diag_size * sizeof(std::complex<double>));

    for(unsigned int i = 0; i < blocksize; i++){
        for(unsigned int j = 0; j < blocksize; j++){
            for(unsigned int k = 0; k < n_blocks; ++k){
                //std::cout << "vectorized eig_inv_diagblk[" << i << "][" << j+k*blocksize << "] = " << eig_inv_diagblk[k].data()[i+j*blocksize] << std::endl;
                unsigned int row = i*matrix_size;
                unsigned int col = j+k*blocksize;
                eigen_inverted_diagblk[row+col] = eig_inv_diagblk[k].data()[i+j*blocksize];
            }

            for(unsigned int k = 0; k < n_blocks-1; ++k){
                unsigned int row = i*off_diag_size;
                unsigned int col = j+k*blocksize;
                eigen_inverted_upperblk[row+col] = eig_inv_upperblk[k].data()[i+j*blocksize];
                eigen_inverted_lowerblk[row+col] = eig_inv_lowerblk[k].data()[i+j*blocksize];
            }
        }
    }


    if(!are_equals(eigen_inverted_diagblk, matrix_inv_diagblk, matrix_size, blocksize)){
        printf("Error: eigen_inverted_diagblk and matrix_inv_diagblk are not equal\n");

        print_matrix(eigen_inverted_diagblk, blocksize, blocksize);
        print_matrix(matrix_inv_diagblk, blocksize, blocksize);
    }
    else{
        printf("eigen_inverted_diagblk and matrix_inv_diagblk are equal\n");
    }

    if(!are_equals(eigen_inverted_upperblk, matrix_inv_upperblk, off_diag_size, blocksize)){
        printf("Error: eigen_inverted_upperblk and matrix_inv_upperblk are not equal\n");
    }
    else{
        printf("eigen_inverted_upperblk and matrix_inv_upperblk are equal\n");
    }

    if(!are_equals(eigen_inverted_lowerblk, matrix_inv_lowerblk, off_diag_size, blocksize)){
        printf("Error: eigen_inverted_lowerblk and matrix_inv_lowerblk are not equal\n");
    }
    else{
        printf("eigen_inverted_lowerblk and matrix_inv_lowerblk are equal\n");
    }

    return 0;
}








