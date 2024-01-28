// Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
#include "single_retarded.h"
#include "single_lesser_greater.h"
#include "single_lesser_greater_retarded.h"
#include "batched_retarded.h"
#include "batched_lesser_greater.h"
#include "batched_lesser_greater_retarded.h"
#include "cudaerrchk.h"

#include <omp.h>
#include <fstream>


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
 
    std::string base_path = "/usr/scratch/mont-fort23/almaeder/rgf_test/";
    std::string time_path = "/usr/scratch/mont-fort233/almaeder/rgf_times/";

    // Get matrix parameters
    // std::string parameter_path = base_path + "batched_matrix_parameters.txt";
    // unsigned int matrix_size;
    // unsigned int blocksize;
    // unsigned int batch_size;

    // load_matrix_parameters_batched(parameter_path.c_str(), &matrix_size, &blocksize, &batch_size);


    // int nb_test = 8;
    // int bs_test = 4;
    // int n_blocks_input[nb_test] = {3*4, 3*8, 3*16, 3*32, 3*64, 3*128, 3*256, 3*512};
    // int blocksize_input[bs_test] = {64, 128, 256, 512};
    int batch = 0;
    int nb_test = 1;
    int bs_test = 1;
    int n_blocks_input[nb_test] =  {3};
    int blocksize_input[bs_test] = {256};
    int number_of_measurements = 1;

    for(int nb = 0; nb < nb_test; nb++){
        for(int bs = 0; bs < bs_test; bs++){
            unsigned int matrix_size = n_blocks_input[nb]*blocksize_input[bs];
            unsigned int blocksize = blocksize_input[bs];
            unsigned int batch_size = 1;

            double memconsumption = (3 * matrix_size * blocksize) * 16.0 / (1e9);
            int memconsumption_int = std::floor(memconsumption);
            std::cout << "memconsumption: " << memconsumption_int << std::endl;

            unsigned int n_blocks = matrix_size / blocksize;
            unsigned int off_diag_size = matrix_size - blocksize;

            // Print the matrix parameters
            printf("Matrix parameters:\n");
            printf("    Matrix size: %d\n", matrix_size);
            printf("    Block size: %d\n", blocksize);
            printf("    Number of blocks: %d\n", n_blocks);


            // Load matrix to invert
            complex_h* system_matrix_diagblk = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
            std::string diagblk_path = base_path + "system_matrix_"+ std::to_string(batch) +"_diagblk_"
            + std::to_string(matrix_size) + "_"+ std::to_string(blocksize) + "_" + std::to_string(batch_size) +
            ".bin";
            load_binary_matrix(diagblk_path.c_str(), system_matrix_diagblk, blocksize, matrix_size);

            complex_h* system_matrix_upperblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
            std::string upperblk_path = base_path + "system_matrix_"+ std::to_string(batch) +"_upperblk_"
            + std::to_string(matrix_size) + "_"+ std::to_string(blocksize) + "_" + std::to_string(batch_size) +
            ".bin";
            load_binary_matrix(upperblk_path.c_str(), system_matrix_upperblk, blocksize, off_diag_size);

            complex_h* system_matrix_lowerblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
            std::string lowerblk_path = base_path + "system_matrix_"+ std::to_string(batch) +"_lowerblk_"
            + std::to_string(matrix_size) + "_"+ std::to_string(blocksize) + "_" + std::to_string(batch_size) +
            ".bin";
            load_binary_matrix(lowerblk_path.c_str(), system_matrix_lowerblk, blocksize, off_diag_size);

            /*
            Matrices are saved in the following way:

            system_matrix_diagblk = [A_0, A_1, ..., A_n]
            system_matrix_upperblk = [B_0, B_1, ..., B_n-1]
            system_matrix_lowerblk = [C_0, C_1, ..., C_n-1]

            where A_i, B_i, C_i are block matrices of size blocksize x blocksize

            The three above arrays are in Row-Major order which means the blocks are not contiguous in memory.

            Below they will be transformed to the following layout:

            system_matrix_diagblk_h = [A_0;
                                A_1;
                                ...;
                                A_n]
            system_matrix_upperblk_h = [B_0;
                                    B_1;
                                    ...;
                                    B_n-1]
            system_matrix_lowerblk_h = [C_0;
                                    C_1;
                                    ...;
                                    C_n-1]

            where blocks are in column major order
            */


            complex_h* system_matrix_diagblk_h = NULL;
            complex_h* system_matrix_upperblk_h = NULL;
            complex_h* system_matrix_lowerblk_h = NULL;
            cudaMallocHost((void**)&system_matrix_diagblk_h, blocksize * matrix_size * sizeof(complex_h));
            cudaMallocHost((void**)&system_matrix_upperblk_h, blocksize * off_diag_size * sizeof(complex_h));
            cudaMallocHost((void**)&system_matrix_lowerblk_h, blocksize * off_diag_size * sizeof(complex_h));

            transform_diagblk<complex_h>(system_matrix_diagblk, system_matrix_diagblk_h, blocksize, matrix_size);
            transform_offblk<complex_h>(system_matrix_upperblk, system_matrix_upperblk_h, blocksize, off_diag_size);
            transform_offblk<complex_h>(system_matrix_lowerblk, system_matrix_lowerblk_h, blocksize, off_diag_size);


            // allocate memory for the inv
            complex_h* retarded_inv_diagblk_h = NULL;
            complex_h* retarded_inv_upperblk_h = NULL;
            complex_h* retarded_inv_lowerblk_h = NULL;

            cudaMallocHost((void**)&retarded_inv_diagblk_h, blocksize * matrix_size * sizeof(complex_h));
            cudaMallocHost((void**)&retarded_inv_upperblk_h, blocksize * off_diag_size * sizeof(complex_h));
            cudaMallocHost((void**)&retarded_inv_lowerblk_h, blocksize * off_diag_size * sizeof(complex_h));

            double times_retarded_fits_gpu_memory[number_of_measurements];
            double times_retarded_fits_gpu_memory_with_copy_compute_overlap[number_of_measurements];
            double times_retarded_does_not_fit_gpu_memory[number_of_measurements];
            double times_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap[number_of_measurements];
            double time;

            // for(int i = 0; i < number_of_measurements; i++){
            //     if(memconsumption_int > 7){
            //         std::cout << "break" << std::endl;
            //         break;
            //     }
            //     std::cout << "fits_gpu_memory Measurement " << i << std::endl;
            //     time = -omp_get_wtime();
            //     rgf_retarded_fits_gpu_memory(
            //         blocksize, matrix_size,
            //         system_matrix_diagblk_h,
            //         system_matrix_upperblk_h,
            //         system_matrix_lowerblk_h,
            //         retarded_inv_diagblk_h,
            //         retarded_inv_upperblk_h,
            //         retarded_inv_lowerblk_h);
            //     time += omp_get_wtime();
            //     std::cout << time << std::endl;
            //     times_retarded_fits_gpu_memory[i] = time;
            // }
            // for(int i = 0; i < number_of_measurements; i++){
            //     if(memconsumption_int > 7){
            //         std::cout << "break" << std::endl;
            //         break;
            //     }
            //     std::cout << "fits_gpu_memory_with_copy_compute_overlap Measurement " << i << std::endl;
            //     time = -omp_get_wtime();
            //     rgf_retarded_fits_gpu_memory_with_copy_compute_overlap(
            //         blocksize, matrix_size,
            //         system_matrix_diagblk_h,
            //         system_matrix_upperblk_h,
            //         system_matrix_lowerblk_h,
            //         retarded_inv_diagblk_h,
            //         retarded_inv_upperblk_h,
            //         retarded_inv_lowerblk_h);
            //     time += omp_get_wtime();
            //     std::cout << time << std::endl;

            //     times_retarded_fits_gpu_memory_with_copy_compute_overlap[i] = time;
            // }
            // for(int i = 0; i < number_of_measurements; i++){
            //     std::cout << "does_not_fit_gpu_memory Measurement " << i << std::endl;
            //     time = -omp_get_wtime();
            //     rgf_retarded_does_not_fit_gpu_memory(
            //         blocksize, matrix_size,
            //         system_matrix_diagblk_h,
            //         system_matrix_upperblk_h,
            //         system_matrix_lowerblk_h,
            //         retarded_inv_diagblk_h,
            //         retarded_inv_upperblk_h,
            //         retarded_inv_lowerblk_h);
            //     time += omp_get_wtime();
            //     std::cout << time << std::endl;

            //     times_retarded_does_not_fit_gpu_memory[i] = time;
            // }
            for(int i = 0; i < number_of_measurements; i++){
                std::cout << "does_not_fit_gpu_memory_with_copy_compute_overlap Measurement " << i << std::endl;
                time = -omp_get_wtime();
                rgf_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap(
                    blocksize, matrix_size,
                    system_matrix_diagblk_h,
                    system_matrix_upperblk_h,
                    system_matrix_lowerblk_h,
                    retarded_inv_diagblk_h,
                    retarded_inv_upperblk_h,
                    retarded_inv_lowerblk_h);
                time += omp_get_wtime();
                std::cout << time << std::endl;
                times_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap[i] = time;
            }
            

            // std::ofstream outputFile_times_retarded_fits_gpu_memory;
            // std::string filename = time_path + "times_retarded_fits_gpu_memory_" + std::to_string(matrix_size) + "_"   + std::to_string(n_blocks) + "_"  + std::to_string(blocksize) + ".txt";
            // outputFile_times_retarded_fits_gpu_memory.open(filename);
            // if(outputFile_times_retarded_fits_gpu_memory.is_open()){
            //     for(int i = 0; i < number_of_measurements; i++){
            //         outputFile_times_retarded_fits_gpu_memory << times_retarded_fits_gpu_memory[i] << std::endl;
            //     }
            // }
            // else{
            //     std::cout << "Unable to open file" << std::endl;
            // }
            // outputFile_times_retarded_fits_gpu_memory.close();

            // std::ofstream outputFile_times_retarded_fits_gpu_memory_with_copy_compute_overlap;
            // filename = time_path + "times_retarded_fits_gpu_memory_with_copy_compute_overlap_" + std::to_string(matrix_size) + "_"   + std::to_string(n_blocks) + "_"  + std::to_string(blocksize) + ".txt";
            // outputFile_times_retarded_fits_gpu_memory_with_copy_compute_overlap.open(filename);
            // if(outputFile_times_retarded_fits_gpu_memory_with_copy_compute_overlap.is_open()){
            //     for(int i = 0; i < number_of_measurements; i++){
            //         outputFile_times_retarded_fits_gpu_memory_with_copy_compute_overlap << times_retarded_fits_gpu_memory_with_copy_compute_overlap[i] << std::endl;
            //     }
            // }
            // else{
            //     std::cout << "Unable to open file" << std::endl;
            // }
            // outputFile_times_retarded_fits_gpu_memory_with_copy_compute_overlap.close();

            // std::ofstream outputFile_times_retarded_does_not_fit_gpu_memory;
            // filename = time_path + "times_retarded_does_not_fit_gpu_memory_" + std::to_string(matrix_size) + "_"   + std::to_string(n_blocks) + "_"  + std::to_string(blocksize) + ".txt";
            // outputFile_times_retarded_does_not_fit_gpu_memory.open(filename);
            // if(outputFile_times_retarded_does_not_fit_gpu_memory.is_open()){
            //     for(int i = 0; i < number_of_measurements; i++){
            //         outputFile_times_retarded_does_not_fit_gpu_memory << times_retarded_does_not_fit_gpu_memory[i] << std::endl;
            //     }
            // }
            // else{
            //     std::cout << "Unable to open file" << std::endl;
            // }
            // outputFile_times_retarded_does_not_fit_gpu_memory.close();

            // std::ofstream outputFile_times_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap;
            // filename = time_path + "times_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap_" + std::to_string(matrix_size) + "_"   + std::to_string(n_blocks) + "_"  + std::to_string(blocksize) + ".txt";
            // outputFile_times_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap.open(filename);
            // if(outputFile_times_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap.is_open()){
            //     for(int i = 0; i < number_of_measurements; i++){
            //         outputFile_times_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap << times_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap[i] << std::endl;
            //     }
            // }
            // else{
            //     std::cout << "Unable to open file" << std::endl;
            // }
            // outputFile_times_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap.close();


            cudaFreeHost(system_matrix_diagblk_h);
            cudaFreeHost(system_matrix_upperblk_h);
            cudaFreeHost(system_matrix_lowerblk_h);


            // free non contiguous memory
            free(system_matrix_diagblk);
            free(system_matrix_upperblk);
            free(system_matrix_lowerblk);

            // free contiguous memory
            cudaFreeHost(retarded_inv_diagblk_h);
            cudaFreeHost(retarded_inv_upperblk_h);
            cudaFreeHost(retarded_inv_lowerblk_h);
            


        }
    }

    return 0;
}








