#include "batched_dense_rgf.h"
#include "dense_rgf.h"

#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <iostream>
#include <string>

int main() {


    if (cudaSetDevice(0) != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device.");
    }

    if(sizeof(complex_h) != sizeof(complex_d)){
        printf("Error: complex_h and complex_d have different sizes\n");
        return 1;
    }
    else{
        printf("complex_h and complex_d have the same size\n");
    }
 
    // std::string base_path = "../../../tests/tests_cases/";
    std::string base_path = "/usr/scratch/mont-fort17/almaeder/rgf_test/";

    // Get matrix parameters
    std::string parameter_path = base_path + "batched_matrix_parameters.txt";
    unsigned int matrix_size;
    unsigned int blocksize;
    unsigned int batch_size;

    load_matrix_parameters_batched(parameter_path.c_str(), &matrix_size, &blocksize, &batch_size);

    unsigned int n_blocks = matrix_size / blocksize;
    unsigned int off_diag_size = matrix_size - blocksize;

    // Print the matrix parameters
    printf("Matrix parameters:\n");
    printf("    Matrix size: %d\n", matrix_size);
    printf("    Block size: %d\n", blocksize);
    printf("    Number of blocks: %d\n", n_blocks);
    printf("    Batch size: %d\n", batch_size);

    complex_h *matrices_diagblk_h[batch_size];
    complex_h *matrices_upperblk_h[batch_size];
    complex_h *matrices_lowerblk_h[batch_size];

    complex_h *inv_matrices_diagblk_ref[batch_size];
    complex_h *inv_matrices_upperblk_ref[batch_size];
    complex_h *inv_matrices_lowerblk_ref[batch_size];
    bool not_failed = true;
    for(int k = 0; k < 100; k++){

        
        for(unsigned int batch = 0; batch < batch_size; batch++){

            // Load matrix to invert
            complex_h* matrix_diagblk = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
            std::string diagblk_path = base_path + "dense_blocks_matrix_"+ std::to_string(batch) +"_diagblk.bin";
            load_binary_matrix(diagblk_path.c_str(), matrix_diagblk, blocksize, matrix_size);

            complex_h* matrix_upperblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
            std::string upperblk_path = base_path + "dense_blocks_matrix_"+ std::to_string(batch) +"_upperblk.bin";
            load_binary_matrix(upperblk_path.c_str(), matrix_upperblk, blocksize, off_diag_size);

            complex_h* matrix_lowerblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
            std::string lowerblk_path = base_path + "dense_blocks_matrix_"+ std::to_string(batch) +"_lowerblk.bin";
            load_binary_matrix(lowerblk_path.c_str(), matrix_lowerblk, blocksize, off_diag_size);

            // Load reference solution of the matrix inverse
            complex_h* matrix_inv_diagblk_ref = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
            std::string inv_diagblk_path = base_path + "dense_blocks_matrix_" + std::to_string(batch) + "_inverse_diagblk.bin";
            load_binary_matrix(inv_diagblk_path.c_str(), matrix_inv_diagblk_ref, blocksize, matrix_size);

            complex_h* matrix_inv_upperblk_ref = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
            std::string inv_upperblk_path = base_path + "dense_blocks_matrix_" + std::to_string(batch) + "_inverse_upperblk.bin";
            load_binary_matrix(inv_upperblk_path.c_str(), matrix_inv_upperblk_ref, blocksize, off_diag_size);
            
            complex_h* matrix_inv_lowerblk_ref = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
            std::string inv_lowerblk_path = base_path + "dense_blocks_matrix_" + std::to_string(batch) + "_inverse_lowerblk.bin";
            load_binary_matrix(inv_lowerblk_path.c_str(), matrix_inv_lowerblk_ref, blocksize, off_diag_size);

            /*
            Matrices are saved in the following way:

            matrix_diagblk = [A_0, A_1, ..., A_n]
            matrix_upperblk = [B_0, B_1, ..., B_n-1]
            matrix_lowerblk = [C_0, C_1, ..., C_n-1]

            where A_i, B_i, C_i are block matrices of size blocksize x blocksize

            The three above arrays are in Row-Major order which means the blocks are not contiguous in memory.

            Below they will be transformed to the following layout:

            matrix_diagblk_h = [A_0;
                                A_1;
                                ...;
                                A_n]
            matrix_upperblk_h = [B_0;
                                    B_1;
                                    ...;
                                    B_n-1]
            matrix_lowerblk_h = [C_0;
                                    C_1;
                                    ...;
                                    C_n-1]

            where blocks are in column major order
            */


            complex_h* matrix_diagblk_h = NULL;
            complex_h* matrix_upperblk_h = NULL;
            complex_h* matrix_lowerblk_h = NULL;
            cudaMallocHost((void**)&matrix_diagblk_h, blocksize * matrix_size * sizeof(complex_h));
            cudaMallocHost((void**)&matrix_upperblk_h, blocksize * off_diag_size * sizeof(complex_h));
            cudaMallocHost((void**)&matrix_lowerblk_h, blocksize * off_diag_size * sizeof(complex_h));
            complex_h* inv_diagblk_ref = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
            complex_h* inv_upperblk_ref = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
            complex_h* inv_lowerblk_ref = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));

            for(unsigned int i = 0; i < blocksize * matrix_size; i++){
                // block index
                int k = i / (blocksize * blocksize);
                // index inside block
                int h = i % (blocksize * blocksize);
                // row inside block
                int m = h % blocksize;
                // col inside block
                int n = h / blocksize;
                matrix_diagblk_h[i] = matrix_diagblk[m*matrix_size + k*blocksize + n];
                inv_diagblk_ref[i] = matrix_inv_diagblk_ref[m*matrix_size + k*blocksize + n];
            }
            for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
                // block index
                int k = i / (blocksize * blocksize);
                // index inside block
                int h = i % (blocksize * blocksize);
                // row inside block
                int m = h % blocksize;
                // col inside block
                int n = h / blocksize;
                matrix_upperblk_h[i] = matrix_upperblk[m*off_diag_size + k*blocksize + n];
                matrix_lowerblk_h[i] = matrix_lowerblk[m*off_diag_size + k*blocksize + n];
                inv_upperblk_ref[i] = matrix_inv_upperblk_ref[m*off_diag_size + k*blocksize + n];
                inv_lowerblk_ref[i] = matrix_inv_lowerblk_ref[m*off_diag_size + k*blocksize + n];
            }
            matrices_diagblk_h[batch] = matrix_diagblk_h;
            matrices_upperblk_h[batch] = matrix_upperblk_h;
            matrices_lowerblk_h[batch] = matrix_lowerblk_h;
            inv_matrices_diagblk_ref[batch] = inv_diagblk_ref;
            inv_matrices_upperblk_ref[batch] = inv_upperblk_ref;
            inv_matrices_lowerblk_ref[batch] = inv_lowerblk_ref;

            // for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
            //     std::cout << "matrix_upperblk_h[" << i << "] = " << matrix_upperblk_h[i] << std::endl;
            // }

            // allocate memory for the inverse
            complex_h* inv_diagblk_h = NULL;
            complex_h* inv_upperblk_h = NULL;
            complex_h* inv_lowerblk_h = NULL;

            cudaMallocHost((void**)&inv_diagblk_h, blocksize * matrix_size * sizeof(complex_h));
            cudaMallocHost((void**)&inv_upperblk_h, blocksize * off_diag_size * sizeof(complex_h));
            cudaMallocHost((void**)&inv_lowerblk_h, blocksize * off_diag_size * sizeof(complex_h));


            rgf_dense_matrix_fits_gpu_memory(blocksize, matrix_size,
                                            matrix_diagblk_h,
                                            matrix_upperblk_h,
                                            matrix_lowerblk_h,
                                            inv_diagblk_h,
                                            inv_upperblk_h,
                                            inv_lowerblk_h);

            double norm_diagblk = 0.0;
            double norm_upperblk = 0.0;
            double norm_lowerblk = 0.0;
            double diff_diagblk = 0.0;
            double diff_upperblk = 0.0;
            double diff_lowerblk = 0.0;
            for(unsigned int i = 0; i < blocksize * matrix_size; i++){
                norm_diagblk += std::abs(inv_diagblk_ref[i]);
                diff_diagblk += std::abs(inv_diagblk_h[i] - inv_diagblk_ref[i]);
            }
            for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
                norm_upperblk += std::abs(inv_upperblk_ref[i]);
                norm_lowerblk += std::abs(inv_lowerblk_ref[i]);
                diff_upperblk += std::abs(inv_upperblk_h[i] - inv_upperblk_ref[i]);
                diff_lowerblk += std::abs(inv_lowerblk_h[i] - inv_lowerblk_ref[i]);
            }
            double eps = 1e-7;
            if(diff_diagblk/norm_diagblk > eps || diff_upperblk/norm_upperblk > eps || diff_lowerblk/norm_lowerblk > eps){
                printf("fits FAILED \n");
                std::cout << diff_diagblk/norm_diagblk << std::endl;
                not_failed = false;
            }

            // set inverse to zero
            for(unsigned int i = 0; i < blocksize * matrix_size; i++){
                inv_diagblk_h[i] = 0.0;
            }
            for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
                inv_upperblk_h[i] = 0.0;
                inv_lowerblk_h[i] = 0.0;
            }
            rgf_dense_matrix_fits_gpu_memory_with_copy_compute_overlap(blocksize, matrix_size,
                                            matrix_diagblk_h,
                                            matrix_upperblk_h,
                                            matrix_lowerblk_h,
                                            inv_diagblk_h,
                                            inv_upperblk_h,
                                            inv_lowerblk_h);

            norm_diagblk = 0.0;
            norm_upperblk = 0.0;
            norm_lowerblk = 0.0;
            diff_diagblk = 0.0;
            diff_upperblk = 0.0;
            diff_lowerblk = 0.0;
            for(unsigned int i = 0; i < blocksize * matrix_size; i++){
                norm_diagblk += std::abs(inv_diagblk_ref[i]);
                diff_diagblk += std::abs(inv_diagblk_h[i] - inv_diagblk_ref[i]);
            }
            for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
                norm_upperblk += std::abs(inv_upperblk_ref[i]);
                norm_lowerblk += std::abs(inv_lowerblk_ref[i]);
                diff_upperblk += std::abs(inv_upperblk_h[i] - inv_upperblk_ref[i]);
                diff_lowerblk += std::abs(inv_lowerblk_h[i] - inv_lowerblk_ref[i]);
            }
            eps = 1e-7;
            if(diff_diagblk/norm_diagblk > eps || diff_upperblk/norm_upperblk > eps || diff_lowerblk/norm_lowerblk > eps){
                printf("fits overlap FAILED \n");
                std::cout << diff_diagblk/norm_diagblk << std::endl;
                not_failed = false;
            }

            // set inverse to zero
            for(unsigned int i = 0; i < blocksize * matrix_size; i++){
                inv_diagblk_h[i] = 0.0;
            }
            for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
                inv_upperblk_h[i] = 0.0;
                inv_lowerblk_h[i] = 0.0;
            }
            rgf_dense_matrix_does_not_fit_gpu_memory(blocksize, matrix_size,
                                            matrix_diagblk_h,
                                            matrix_upperblk_h,
                                            matrix_lowerblk_h,
                                            inv_diagblk_h,
                                            inv_upperblk_h,
                                            inv_lowerblk_h);

            norm_diagblk = 0.0;
            norm_upperblk = 0.0;
            norm_lowerblk = 0.0;
            diff_diagblk = 0.0;
            diff_upperblk = 0.0;
            diff_lowerblk = 0.0;
            for(unsigned int i = 0; i < blocksize * matrix_size; i++){
                norm_diagblk += std::abs(inv_diagblk_ref[i]);
                diff_diagblk += std::abs(inv_diagblk_h[i] - inv_diagblk_ref[i]);
            }
            for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
                norm_upperblk += std::abs(inv_upperblk_ref[i]);
                norm_lowerblk += std::abs(inv_lowerblk_ref[i]);
                diff_upperblk += std::abs(inv_upperblk_h[i] - inv_upperblk_ref[i]);
                diff_lowerblk += std::abs(inv_lowerblk_h[i] - inv_lowerblk_ref[i]);
            }
            eps = 1e-7;
            if(diff_diagblk/norm_diagblk > eps || diff_upperblk/norm_upperblk > eps || diff_lowerblk/norm_lowerblk > eps){
                printf("not fit FAILED \n");
                std::cout << diff_diagblk/norm_diagblk << std::endl;
                not_failed = false;
            }

            // set inverse to zero
            for(unsigned int i = 0; i < blocksize * matrix_size; i++){
                inv_diagblk_h[i] = 0.0;
            }
            for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
                inv_upperblk_h[i] = 0.0;
                inv_lowerblk_h[i] = 0.0;
            }
            rgf_dense_matrix_does_not_fit_gpu_memory_with_copy_compute_overlap(blocksize, matrix_size,
                                            matrix_diagblk_h,
                                            matrix_upperblk_h,
                                            matrix_lowerblk_h,
                                            inv_diagblk_h,
                                            inv_upperblk_h,
                                            inv_lowerblk_h);

            norm_diagblk = 0.0;
            norm_upperblk = 0.0;
            norm_lowerblk = 0.0;
            diff_diagblk = 0.0;
            diff_upperblk = 0.0;
            diff_lowerblk = 0.0;
            for(unsigned int i = 0; i < blocksize * matrix_size; i++){
                norm_diagblk += std::abs(inv_diagblk_ref[i]);
                diff_diagblk += std::abs(inv_diagblk_h[i] - inv_diagblk_ref[i]);
            }
            for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
                norm_upperblk += std::abs(inv_upperblk_ref[i]);
                norm_lowerblk += std::abs(inv_lowerblk_ref[i]);
                diff_upperblk += std::abs(inv_upperblk_h[i] - inv_upperblk_ref[i]);
                diff_lowerblk += std::abs(inv_lowerblk_h[i] - inv_lowerblk_ref[i]);
            }
            eps = 1e-7;
            if(diff_diagblk/norm_diagblk > eps || diff_upperblk/norm_upperblk > eps || diff_lowerblk/norm_lowerblk > eps){
                printf("not fit overlap FAILED \n");
                std::cout << diff_diagblk/norm_diagblk << std::endl;
                not_failed = false;
            }


            // if(diff_diagblk/norm_diagblk > eps){
            //     printf("Error: batch_inv_diagblk_h and inv_diagblk_ref are not equal\n");
            // }
            // else{
            //     printf("batch_inv_diagblk_h and inv_diagblk_ref are equal\n");
            // }
            // std::cout << diff_diagblk/norm_diagblk << std::endl;

            // if(diff_upperblk/norm_upperblk > eps){
            //     printf("Error: batch_inv_upperblk_h and inv_upperblk_ref are not equal\n");
            // }
            // else{
            //     printf("batch_inv_upperblk_h and inv_upperblk_ref are equal\n");
            // }
            // std::cout << diff_upperblk/norm_upperblk << std::endl;

            // if(diff_lowerblk/norm_lowerblk > eps){
            //     printf("Error: batch_inv_lowerblk_h and inv_lowerblk_ref are not equal\n");
            // }
            // else{
            //     printf("batch_inv_lowerblk_h and inv_lowerblk_ref are equal\n");
            // }
            // std::cout << diff_lowerblk/norm_lowerblk << std::endl;


            if(inv_diagblk_h){
                cudaFreeHost(inv_diagblk_h);
            }
            if(inv_upperblk_h){
                cudaFreeHost(inv_upperblk_h);
            }
            if(inv_lowerblk_h){
                cudaFreeHost(inv_lowerblk_h);
            }
            // free non contiguous memory
            if(matrix_diagblk){
                free(matrix_diagblk);
            }
            if(matrix_upperblk){
                free(matrix_upperblk);
            }
            if(matrix_lowerblk){
                free(matrix_lowerblk);
            }
            if(matrix_inv_diagblk_ref){
                free(matrix_inv_diagblk_ref);
            }
            if(matrix_inv_upperblk_ref){
                free(matrix_inv_upperblk_ref);
            }
            if(matrix_inv_lowerblk_ref){
                free(matrix_inv_lowerblk_ref);
            }

        }


        // transform to batched blocks
        complex_h *batch_diagblk_h[n_blocks];
        complex_h *batch_upperblk_h[n_blocks-1];
        complex_h *batch_lowerblk_h[n_blocks-1];
        complex_h *batched_inv_matrices_diagblk_ref[n_blocks];
        complex_h *batched_inv_matrices_upperblk_ref[n_blocks-1];
        complex_h *batched_inv_matrices_lowerblk_ref[n_blocks-1];
        complex_h *batch_inv_diagblk_h[n_blocks];
        complex_h *batch_inv_upperblk_h[n_blocks-1];
        complex_h *batch_inv_lowerblk_h[n_blocks-1];

        for(unsigned int i = 0; i < n_blocks; i++){
            cudaErrchk(cudaMallocHost((void**)&batch_diagblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batched_inv_matrices_diagblk_ref[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_inv_diagblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));    
            for(unsigned int batch = 0; batch < batch_size; batch++){
                for(unsigned int j = 0; j < blocksize * blocksize; j++){
                    batch_diagblk_h[i][batch * blocksize * blocksize + j] =
                        matrices_diagblk_h[batch][i * blocksize * blocksize + j];
                    batched_inv_matrices_diagblk_ref[i][batch * blocksize * blocksize + j] =
                        inv_matrices_diagblk_ref[batch][i * blocksize * blocksize + j];
                }
            }
        


        }
        for(unsigned int i = 0; i < n_blocks - 1; i++){
            cudaErrchk(cudaMallocHost((void**)&batch_upperblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_lowerblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batched_inv_matrices_upperblk_ref[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batched_inv_matrices_lowerblk_ref[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_inv_upperblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_inv_lowerblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            for(unsigned int batch = 0; batch < batch_size; batch++){
                for(unsigned int j = 0; j < blocksize * blocksize; j++){
                    batch_upperblk_h[i][batch * blocksize * blocksize + j] =
                        matrices_upperblk_h[batch][i * blocksize * blocksize + j];
                    batch_lowerblk_h[i][batch * blocksize * blocksize + j] =
                        matrices_lowerblk_h[batch][i * blocksize * blocksize + j];
                    batched_inv_matrices_upperblk_ref[i][batch * blocksize * blocksize + j] =
                        inv_matrices_upperblk_ref[batch][i * blocksize * blocksize + j];
                    batched_inv_matrices_lowerblk_ref[i][batch * blocksize * blocksize + j] =
                        inv_matrices_lowerblk_ref[batch][i * blocksize * blocksize + j];
                }
            }
        
        }

        rgf_batched(
            blocksize, matrix_size, batch_size,
            batch_diagblk_h, batch_upperblk_h, batch_lowerblk_h,
            batch_inv_diagblk_h, batch_inv_upperblk_h, batch_inv_lowerblk_h
        );

        double norm_diagblk = 0.0;
        double norm_upperblk = 0.0;
        double norm_lowerblk = 0.0;
        double diff_diagblk = 0.0;
        double diff_upperblk = 0.0;
        double diff_lowerblk = 0.0;
        for(unsigned int i = 0; i < n_blocks; i++){
            for(unsigned j = 0; j < batch_size * blocksize * blocksize; j++){
                norm_diagblk += std::abs(batch_inv_diagblk_h[i][j]);
                diff_diagblk += std::abs(batch_inv_diagblk_h[i][j] - batched_inv_matrices_diagblk_ref[i][j]);
            }

        }
        for(unsigned int i = 0; i < n_blocks - 1; i++){
            for(unsigned int j = 0; j < batch_size * blocksize * blocksize; j++){
                norm_upperblk += std::abs(batch_inv_upperblk_h[i][j]);
                norm_lowerblk += std::abs(batch_inv_lowerblk_h[i][j]);
                diff_upperblk += std::abs(batch_inv_upperblk_h[i][j] - batched_inv_matrices_upperblk_ref[i][j]);
                diff_lowerblk += std::abs(batch_inv_lowerblk_h[i][j] - batched_inv_matrices_lowerblk_ref[i][j]);
            }
        }


        double eps = 1e-7;
        if(diff_diagblk/norm_diagblk > eps || diff_upperblk/norm_upperblk > eps || diff_lowerblk/norm_lowerblk > eps){
            printf("Batch FAILED\n");
            std::cout << diff_diagblk/norm_diagblk << std::endl;
            not_failed = false;
        }
        // if(diff_diagblk/norm_diagblk > eps){
        //     printf("Error: batch_inv_diagblk_h and batched_inv_matrices_diagblk_ref are not equal\n");
        // }
        // else{
        //     printf("batch_inv_diagblk_h and batched_inv_matrices_diagblk_ref are equal\n");
        // }
        // std::cout << diff_diagblk/norm_diagblk << std::endl;
        // if(diff_upperblk/norm_upperblk > eps){
        //     printf("Error: batch_inv_upperblk_h and batched_inv_matrices_upperblk_ref are not equal\n");
        // }
        // else{
        //     printf("batch_inv_upperblk_h and batched_inv_matrices_upperblk_ref are equal\n");
        // }
        // std::cout << diff_upperblk/norm_upperblk << std::endl;
        // if(diff_lowerblk/norm_lowerblk > eps){
        //     printf("Error: batch_inv_lowerblk_h and batched_inv_matrices_lowerblk_ref are not equal\n");
        // }
        // else{
        //     printf("batch_inv_lowerblk_h and batched_inv_matrices_lowerblk_ref are equal\n");
        // }
        // std::cout << diff_lowerblk/norm_lowerblk << std::endl;


        // set inverse to zero
        for(unsigned int i = 0; i < n_blocks; i++){
            for(unsigned j = 0; j < batch_size * blocksize * blocksize; j++){
                batch_inv_diagblk_h[i][j] = 0.0;
            }

        }
        for(unsigned int i = 0; i < n_blocks - 1; i++){
            for(unsigned int j = 0; j < batch_size * blocksize * blocksize; j++){
                batch_inv_upperblk_h[i][j] = 0.0;
                batch_inv_lowerblk_h[i][j] = 0.0;
            }
        }


        rgf_multiple_energy_points_for_loop(
            blocksize, matrix_size, batch_size,
            batch_diagblk_h, batch_upperblk_h, batch_lowerblk_h,
            batch_inv_diagblk_h, batch_inv_upperblk_h, batch_inv_lowerblk_h
        );

        norm_diagblk = 0.0;
        norm_upperblk = 0.0;
        norm_lowerblk = 0.0;
        diff_diagblk = 0.0;
        diff_upperblk = 0.0;
        diff_lowerblk = 0.0;

        for(unsigned int i = 0; i < n_blocks; i++){
            for(unsigned j = 0; j < batch_size * blocksize * blocksize; j++){
                norm_diagblk += std::abs(batch_inv_diagblk_h[i][j]);
                diff_diagblk += std::abs(batch_inv_diagblk_h[i][j] - batched_inv_matrices_diagblk_ref[i][j]);
            }

        }
        for(unsigned int i = 0; i < n_blocks - 1; i++){
            for(unsigned int j = 0; j < batch_size * blocksize * blocksize; j++){
                norm_upperblk += std::abs(batch_inv_upperblk_h[i][j]);
                norm_lowerblk += std::abs(batch_inv_lowerblk_h[i][j]);
                diff_upperblk += std::abs(batch_inv_upperblk_h[i][j] - batched_inv_matrices_upperblk_ref[i][j]);
                diff_lowerblk += std::abs(batch_inv_lowerblk_h[i][j] - batched_inv_matrices_lowerblk_ref[i][j]);
            }
        }

        if(diff_diagblk/norm_diagblk > eps || diff_upperblk/norm_upperblk > eps || diff_lowerblk/norm_lowerblk > eps){
            printf("FOR FAILED\n");
            not_failed = false;
        }
        // if(diff_diagblk/norm_diagblk > eps){
        //     printf("Error: for batch_inv_diagblk_h and batched_inv_matrices_diagblk_ref are not equal\n");
        // }
        // else{
        //     printf("for batch_inv_diagblk_h and batched_inv_matrices_diagblk_ref are equal\n");
        // }
        // std::cout << diff_diagblk/norm_diagblk << std::endl;
        // if(diff_upperblk/norm_upperblk > eps){
        //     printf("Error: for batch_inv_upperblk_h and batched_inv_matrices_upperblk_ref are not equal\n");
        // }
        // else{
        //     printf("for batch_inv_upperblk_h and batched_inv_matrices_upperblk_ref are equal\n");
        // }
        // std::cout << diff_upperblk/norm_upperblk << std::endl;
        // if(diff_lowerblk/norm_lowerblk > eps){
        //     printf("Error: for batch_inv_lowerblk_h and batched_inv_matrices_lowerblk_ref are not equal\n");
        // }
        // else{
        //     printf("for batch_inv_lowerblk_h and batched_inv_matrices_lowerblk_ref are equal\n");
        // }
        // std::cout << diff_lowerblk/norm_lowerblk << std::endl;



        // free contiguous memory
        for(unsigned int batch = 0; batch < batch_size; batch++){
            if(matrices_diagblk_h[batch]){
                cudaFreeHost(matrices_diagblk_h[batch]);
            }
            if(matrices_upperblk_h[batch]){
                cudaFreeHost(matrices_upperblk_h[batch]);
            }
            if(matrices_lowerblk_h[batch]){
                cudaFreeHost(matrices_lowerblk_h[batch]);
            }
            if(inv_matrices_diagblk_ref[batch]){
                free(inv_matrices_diagblk_ref[batch]);
            }
            if(inv_matrices_upperblk_ref[batch]){
                free(inv_matrices_upperblk_ref[batch]);
            }
            if(inv_matrices_lowerblk_ref[batch]){
                free(inv_matrices_lowerblk_ref[batch]);
            }
        }
        for(unsigned int i = 0; i < n_blocks; i++){
            if(batch_diagblk_h[i]){
                cudaFreeHost(batch_diagblk_h[i]);
            }
            if(batched_inv_matrices_diagblk_ref[i]){
                cudaFreeHost(batched_inv_matrices_diagblk_ref[i]);
            }
            if(batch_inv_diagblk_h[i]){
                cudaFreeHost(batch_inv_diagblk_h[i]);
            }
        }
        for(unsigned int i = 0; i < n_blocks - 1; i++){
            if(batch_upperblk_h[i]){
                cudaFreeHost(batch_upperblk_h[i]);
            }
            if(batch_lowerblk_h[i]){
                cudaFreeHost(batch_lowerblk_h[i]);
            }
            if(batched_inv_matrices_upperblk_ref[i]){
                cudaFreeHost(batched_inv_matrices_upperblk_ref[i]);
            }
            if(batched_inv_matrices_lowerblk_ref[i]){
                cudaFreeHost(batched_inv_matrices_lowerblk_ref[i]);
            }
            if(batch_inv_upperblk_h[i]){
                cudaFreeHost(batch_inv_upperblk_h[i]);
            }
            if(batch_inv_lowerblk_h[i]){
                cudaFreeHost(batch_inv_lowerblk_h[i]);
            }
        }
    }
    if(!not_failed){
        printf("FAILED\n");
    }
    else{
        printf("PASSED\n");
    }
    return 0;
}








