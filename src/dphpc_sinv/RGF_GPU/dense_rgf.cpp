// Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.

#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <iostream>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>
#include <cuda/std/complex>
#include <Eigen/Dense>

#include "utils.h"


#define cudaErrchk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDAassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


using complex_h = std::complex<double>;
using complex_d = cuda::std::complex<double>;
int main() {
    // Get matrix parameters
    char f_matparam[] = "../../../tests/tests_cases/dense_blocks_matrix_0_parameters.txt";
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
    char f_mat_diagblk[] = "../../../tests/tests_cases/dense_blocks_matrix_0_diagblk.bin";
    load_binary_matrix(f_mat_diagblk, matrix_diagblk, blocksize, matrix_size);

    std::complex<double>* matrix_upperblk = (std::complex<double>*) malloc(blocksize * (off_diag_size) * sizeof(std::complex<double>));
    char f_mat_upperblk[] = "../../../tests/tests_cases/dense_blocks_matrix_0_upperblk.bin";
    load_binary_matrix(f_mat_upperblk, matrix_upperblk, blocksize, off_diag_size);

    std::complex<double>* matrix_lowerblk = (std::complex<double>*) malloc(blocksize * (off_diag_size) * sizeof(std::complex<double>));
    char f_mat_lowerblk[] = "../../../tests/tests_cases/dense_blocks_matrix_0_lowerblk.bin";
    load_binary_matrix(f_mat_lowerblk, matrix_lowerblk, blocksize, off_diag_size);

    /*
    Matrices are saved in the following way:

    matrix_diagblk = [A_0, A_1, ..., A_n]
    matrix_upperblk = [B_0, B_1, ..., B_n-1]
    matrix_lowerblk = [C_0, C_1, ..., C_n-1]

    where A_i, B_i, C_i are block matrices of size blocksize x blocksize

    The three above arrays are in Row-Major order which means the blocks are not contiguous in memory.

    Below they will be transformed to the following layout:

    matrix_diagblk_cmajor = [A_0;
                             A_1;
                             ...;
                             A_n]
    matrix_upperblk_cmajor = [B_0;
                              B_1;
                              ...;
                              B_n-1]
    matrix_lowerblk_cmajor = [C_0;
                              C_1;
                              ...;
                              C_n-1]
    */


    complex_h* matrix_diagblk_cmajor = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
    complex_h* matrix_upperblk_cmajor = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
    complex_h* matrix_lowerblk_cmajor = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));

    for(unsigned int i = 0; i < blocksize * matrix_size; i++){
        // block index
        int k = i / (blocksize * blocksize);
        // index inside block
        int h = i % (blocksize * blocksize);
        // row inside block
        int m = h / blocksize;
        // col inside block
        int n = h % blocksize;
        matrix_diagblk_cmajor[i] = matrix_diagblk[m*matrix_size + k*blocksize + n];
    }
    for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
        // block index
        int k = i / (blocksize * blocksize);
        // index inside block
        int h = i % (blocksize * blocksize);
        // row inside block
        int m = h / blocksize;
        // col inside block
        int n = h % blocksize;
        matrix_upperblk_cmajor[i] = matrix_upperblk[m*off_diag_size + k*blocksize + n];
        matrix_lowerblk_cmajor[i] = matrix_lowerblk[m*off_diag_size + k*blocksize + n];
    }
    for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
        std::cout << "matrix_upperblk_cmajor[" << i << "] = " << matrix_upperblk_cmajor[i] << std::endl;
    }


    complex_d* matrix_diagblk_d = NULL;
    complex_d* matrix_upperblk_d = NULL;
    complex_d* matrix_lowerblk_d = NULL;
    
    cudaStream_t stream = NULL;
    cudaErrchk(cudaStreamCreate(&stream));

    cudaErrchk(cudaMalloc((void**)&matrix_diagblk_d, blocksize * matrix_size * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&matrix_upperblk_d, blocksize * (off_diag_size) * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&matrix_lowerblk_d, blocksize * (off_diag_size) * sizeof(complex_d)));

    cudaErrchk(cudaMemcpy(matrix_diagblk_d, reinterpret_cast<const complex_d*>(matrix_diagblk_cmajor),
                blocksize * matrix_size * sizeof(complex_d), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(matrix_upperblk_d, reinterpret_cast<const complex_d*>(matrix_upperblk_cmajor),
                blocksize * (off_diag_size) * sizeof(complex_d), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(matrix_lowerblk_d, reinterpret_cast<const complex_d*>(matrix_lowerblk_cmajor),
                blocksize * (off_diag_size) * sizeof(complex_d), cudaMemcpyHostToDevice));

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
    char f_mat_inv_diagblk[] = "../../../tests/tests_cases/dense_blocks_matrix_0_inverse_diagblk.bin";
    load_binary_matrix(f_mat_inv_diagblk, matrix_inv_diagblk, blocksize, matrix_size);

    std::complex<double>* matrix_inv_upperblk = (std::complex<double>*) malloc(blocksize * (off_diag_size) * sizeof(std::complex<double>));
    char f_mat_inv_upperblk[] = "../../../tests/tests_cases/dense_blocks_matrix_0_inverse_upperblk.bin";
    load_binary_matrix(f_mat_inv_upperblk, matrix_inv_upperblk, blocksize, off_diag_size);
    
    std::complex<double>* matrix_inv_lowerblk = (std::complex<double>*) malloc(blocksize * (off_diag_size) * sizeof(std::complex<double>));
    char f_mat_inv_lowerblk[] = "../../../tests/tests_cases/dense_blocks_matrix_0_inverse_lowerblk.bin";
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

    if (stream) {
        cudaErrchk(cudaStreamDestroy(stream));
    }
    if(matrix_diagblk_d) {
        cudaErrchk(cudaFree(matrix_diagblk_d));
    }
    if(matrix_upperblk_d) {
        cudaErrchk(cudaFree(matrix_upperblk_d));
    }
    if(matrix_lowerblk_d) {
        cudaErrchk(cudaFree(matrix_lowerblk_d));
    }

    return 0;
}








