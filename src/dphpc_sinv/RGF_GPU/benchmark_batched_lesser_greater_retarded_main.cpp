// Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
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
    std::string time_path = "/usr/scratch/mont-fort23/almaeder/rgf_times_batched/";
    int nbatch_test = 10;
    int nb_test = 7;
    int bs_test = 6;
    int batch_sizes_input[nbatch_test] = {1, 2, 4, 8, 16, 32, 48, 64, 96, 128};
    int n_blocks_input[nb_test] = {2, 4, 6, 8, 10, 12, 14};
    int blocksize_input[bs_test] = {64, 128, 256, 512, 768, 1024};

    // int nb_test = 1;
    // int bs_test = 1;
    // int n_blocks_input[nb_test] = {3};
    // //int blocksize_input[bs_test] = {64, 128, 256, 512};
    // int blocksize_input[bs_test] = {1024};
    int number_of_measurements = 22;    

    for(int bas = 0; bas < nbatch_test; bas++){
        unsigned int batch_size = batch_sizes_input[bas];
        for(int nb = 0; nb < nb_test; nb++){
            for(int bs = 0; bs < bs_test; bs++){
                unsigned int matrix_size = n_blocks_input[nb]*blocksize_input[bs];
                unsigned int blocksize = blocksize_input[bs];
                

                double memconsumption = 2.0 * batch_size * (3.0 * matrix_size * blocksize) * 16.0 / (1e9);
                double memconsumption2 = 27*batch_size * blocksize * blocksize * 16.0 / (1e9);
                int memconsumption_int = std::floor(memconsumption);
                int memconsumption_int2 = std::floor(memconsumption2);
                std::cout << "memconsumption: " << memconsumption_int << std::endl;
                std::cout << "memconsumption2: " << memconsumption_int2 << std::endl;

                if(memconsumption_int > 64 || memconsumption_int2 > 12){
                    std::cout << "Memory consumption too high, skipping" << std::endl;
                    continue;
                }

                unsigned int n_blocks = matrix_size / blocksize;

                // Print the matrix parameters
                printf("Matrix parameters:\n");
                printf("    Matrix size: %d\n", matrix_size);
                printf("    Block size: %d\n", blocksize);
                printf("    Number of blocks: %d\n", n_blocks);
                printf("    Batch size: %d\n", batch_size);

                complex_h* batch_system_matrices_diagblk_h[n_blocks];
                complex_h* batch_system_matrices_upperblk_h[n_blocks-1];
                complex_h* batch_system_matrices_lowerblk_h[n_blocks-1];
                complex_h* batch_self_energy_matrices_lesser_diagblk_h[n_blocks];
                complex_h* batch_self_energy_matrices_lesser_upperblk_h[n_blocks-1];
                complex_h* batch_self_energy_matrices_greater_diagblk_h[n_blocks];
                complex_h* batch_self_energy_matrices_greater_upperblk_h[n_blocks-1];
                complex_h* batch_lesser_inv_matrices_diagblk_h[n_blocks];
                complex_h* batch_lesser_inv_matrices_upperblk_h[n_blocks-1];
                complex_h* batch_greater_inv_matrices_diagblk_h[n_blocks];
                complex_h* batch_greater_inv_matrices_upperblk_h[n_blocks-1];
                complex_h* batch_retarded_inv_matrices_diagblk_h[n_blocks];
                complex_h* batch_retarded_inv_matrices_upperblk_h[n_blocks-1];
                complex_h* batch_retarded_inv_matrices_lowerblk_h[n_blocks-1];

                std::string diagblk_path = base_path + "system_matrix_diagblk_" + std::to_string(blocksize) +
                    "_" + std::to_string(batch_size) +
                    ".bin";
                std::string upperblk_path = base_path + "system_matrix_upperblk_" + std::to_string(blocksize) +
                    "_" + std::to_string(batch_size) +
                    ".bin";
                std::string lowerblk_path = base_path + "system_matrix_lowerblk_" + std::to_string(blocksize) +
                    "_" + std::to_string(batch_size) +
                    ".bin";
                std::string self_energy_diagblk_path = base_path + "self_energy_diagblk_"
                    + std::to_string(blocksize) + "_" + std::to_string(batch_size) +
                    ".bin";
                std::string self_energy_upperblk_path = base_path + "self_energy_upperblk_"
                    + std::to_string(blocksize) + "_" + std::to_string(batch_size) +
                    ".bin";
                
                for(unsigned int i = 0; i < n_blocks; i++){
                    cudaErrchk(cudaMallocHost((void**)&batch_system_matrices_diagblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
                    cudaErrchk(cudaMallocHost((void**)&batch_self_energy_matrices_lesser_diagblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
                    cudaErrchk(cudaMallocHost((void**)&batch_self_energy_matrices_greater_diagblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
                    cudaErrchk(cudaMallocHost((void**)&batch_lesser_inv_matrices_diagblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
                    cudaErrchk(cudaMallocHost((void**)&batch_greater_inv_matrices_diagblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
                    cudaErrchk(cudaMallocHost((void**)&batch_retarded_inv_matrices_diagblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
                    load_binary_array<complex_h>(diagblk_path, batch_system_matrices_diagblk_h[i], blocksize*blocksize*batch_size);
                    load_binary_array<complex_h>(self_energy_diagblk_path, batch_self_energy_matrices_lesser_diagblk_h[i], blocksize*blocksize*batch_size);
                    load_binary_array<complex_h>(self_energy_diagblk_path, batch_self_energy_matrices_greater_diagblk_h[i], blocksize*blocksize*batch_size);

                }
                for(unsigned int i = 0; i < n_blocks-1; i++){
                    cudaErrchk(cudaMallocHost((void**)&batch_system_matrices_upperblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
                    cudaErrchk(cudaMallocHost((void**)&batch_system_matrices_lowerblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
                    cudaErrchk(cudaMallocHost((void**)&batch_self_energy_matrices_lesser_upperblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
                    cudaErrchk(cudaMallocHost((void**)&batch_self_energy_matrices_greater_upperblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
                    cudaErrchk(cudaMallocHost((void**)&batch_lesser_inv_matrices_upperblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
                    cudaErrchk(cudaMallocHost((void**)&batch_greater_inv_matrices_upperblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
                    cudaErrchk(cudaMallocHost((void**)&batch_retarded_inv_matrices_upperblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
                    cudaErrchk(cudaMallocHost((void**)&batch_retarded_inv_matrices_lowerblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
                    load_binary_array<complex_h>(upperblk_path, batch_system_matrices_upperblk_h[i], blocksize*blocksize*batch_size);
                    load_binary_array<complex_h>(lowerblk_path, batch_system_matrices_lowerblk_h[i], blocksize*blocksize*batch_size);
                    load_binary_array<complex_h>(self_energy_upperblk_path, batch_self_energy_matrices_lesser_upperblk_h[i], blocksize*blocksize*batch_size);
                    load_binary_array<complex_h>(self_energy_upperblk_path, batch_self_energy_matrices_greater_upperblk_h[i], blocksize*blocksize*batch_size);
                }


                double times_lesser_greater_retarded_for[number_of_measurements];
                double times_lesser_greater_retarded_batched[number_of_measurements];
                double times_lesser_greater_retarded_memcpy[number_of_measurements];
                double time;


                for(int i = 0; i < number_of_measurements; i++){
                    std::cout << "Batched Measurement " << i << std::endl;
                    time = -omp_get_wtime();
                    rgf_lesser_greater_retarded_batched(blocksize, matrix_size, batch_size,
                                                batch_system_matrices_diagblk_h,
                                                batch_system_matrices_upperblk_h,
                                                batch_system_matrices_lowerblk_h,
                                                batch_self_energy_matrices_lesser_diagblk_h,
                                                batch_self_energy_matrices_lesser_upperblk_h,
                                                batch_self_energy_matrices_greater_diagblk_h,
                                                batch_self_energy_matrices_greater_upperblk_h,
                                                batch_lesser_inv_matrices_diagblk_h,
                                                batch_lesser_inv_matrices_upperblk_h,
                                                batch_greater_inv_matrices_diagblk_h,
                                                batch_greater_inv_matrices_upperblk_h,
                                                batch_retarded_inv_matrices_diagblk_h,
                                                batch_retarded_inv_matrices_upperblk_h,
                                                batch_retarded_inv_matrices_lowerblk_h);
                    time += omp_get_wtime();
                    std::cout << "Time: " << time << std::endl;
                    times_lesser_greater_retarded_batched[i] = time;
                    
                }

                for(int i = 0; i < number_of_measurements; i++){
                    std::cout << "For Loop Measurement " << i << std::endl;
                    time = -omp_get_wtime();
                    rgf_lesser_greater_retarded_for(blocksize, matrix_size, batch_size,
                                                batch_system_matrices_diagblk_h,
                                                batch_system_matrices_upperblk_h,
                                                batch_system_matrices_lowerblk_h,
                                                batch_self_energy_matrices_lesser_diagblk_h,
                                                batch_self_energy_matrices_lesser_upperblk_h,
                                                batch_self_energy_matrices_greater_diagblk_h,
                                                batch_self_energy_matrices_greater_upperblk_h,
                                                batch_lesser_inv_matrices_diagblk_h,
                                                batch_lesser_inv_matrices_upperblk_h,
                                                batch_greater_inv_matrices_diagblk_h,
                                                batch_greater_inv_matrices_upperblk_h,
                                                batch_retarded_inv_matrices_diagblk_h,
                                                batch_retarded_inv_matrices_upperblk_h,
                                                batch_retarded_inv_matrices_lowerblk_h);
                    time += omp_get_wtime();
                    std::cout << "Time: " << time << std::endl;
                    times_lesser_greater_retarded_for[i] = time;
                }


                for(int i = 0; i < number_of_measurements; i++){
                    std::cout << "Batched memcpy Measurement " << i << std::endl;
                    time = -omp_get_wtime();
                    rgf_lesser_greater_retarded_batched_memcpy(blocksize, matrix_size, batch_size,
                                                batch_system_matrices_diagblk_h,
                                                batch_system_matrices_upperblk_h,
                                                batch_system_matrices_lowerblk_h,
                                                batch_self_energy_matrices_lesser_diagblk_h,
                                                batch_self_energy_matrices_lesser_upperblk_h,
                                                batch_self_energy_matrices_greater_diagblk_h,
                                                batch_self_energy_matrices_greater_upperblk_h,
                                                batch_lesser_inv_matrices_diagblk_h,
                                                batch_lesser_inv_matrices_upperblk_h,
                                                batch_greater_inv_matrices_diagblk_h,
                                                batch_greater_inv_matrices_upperblk_h,
                                                batch_retarded_inv_matrices_diagblk_h,
                                                batch_retarded_inv_matrices_upperblk_h,
                                                batch_retarded_inv_matrices_lowerblk_h);
                    time += omp_get_wtime();
                    std::cout << "Time: " << time << std::endl;
                    times_lesser_greater_retarded_memcpy[i] = time;
                    
                }


                std::ofstream outputFile_times_lesser_greater_retarded_for;
                std::string filename = time_path + "times_lesser_greater_retarded_for_" + std::to_string(matrix_size) + "_"   + std::to_string(n_blocks) + "_"  + std::to_string(batch_size) + ".txt";
                outputFile_times_lesser_greater_retarded_for.open(filename);
                if(outputFile_times_lesser_greater_retarded_for.is_open()){
                    for(int i = 0; i < number_of_measurements; i++){
                        outputFile_times_lesser_greater_retarded_for << times_lesser_greater_retarded_for[i] << std::endl;
                    }
                }
                else{
                    std::cout << "Unable to open file" << std::endl;
                }
                outputFile_times_lesser_greater_retarded_for.close();

                std::ofstream outputFile_times_lesser_greater_retarded_batched;
                filename = time_path + "times_lesser_greater_retarded_batched_" + std::to_string(matrix_size) + "_"   + std::to_string(n_blocks) + "_"  + std::to_string(batch_size) + ".txt";
                outputFile_times_lesser_greater_retarded_batched.open(filename);
                if(outputFile_times_lesser_greater_retarded_batched.is_open()){
                    for(int i = 0; i < number_of_measurements; i++){
                        outputFile_times_lesser_greater_retarded_batched << times_lesser_greater_retarded_batched[i] << std::endl;
                    }
                }
                else{
                    std::cout << "Unable to open file" << std::endl;
                }
                outputFile_times_lesser_greater_retarded_batched.close();

                std::ofstream outputFile_times_lesser_greater_retarded_memcpy;
                filename = time_path + "times_lesser_greater_retarded_memcpy_" + std::to_string(matrix_size) + "_"   + std::to_string(n_blocks) + "_"  + std::to_string(batch_size) + ".txt";
                outputFile_times_lesser_greater_retarded_memcpy.open(filename);
                if(outputFile_times_lesser_greater_retarded_memcpy.is_open()){
                    for(int i = 0; i < number_of_measurements; i++){
                        outputFile_times_lesser_greater_retarded_memcpy << times_lesser_greater_retarded_memcpy[i] << std::endl;
                    }
                }
                else{
                    std::cout << "Unable to open file" << std::endl;
                }
                outputFile_times_lesser_greater_retarded_memcpy.close();

                for(unsigned int i = 0; i < n_blocks; i++){
                    cudaFreeHost(batch_system_matrices_diagblk_h[i]);
                    cudaFreeHost(batch_self_energy_matrices_lesser_diagblk_h[i]);
                    cudaFreeHost(batch_self_energy_matrices_greater_diagblk_h[i]);
                    cudaFreeHost(batch_retarded_inv_matrices_diagblk_h[i]);
                    cudaFreeHost(batch_lesser_inv_matrices_diagblk_h[i]);
                    cudaFreeHost(batch_greater_inv_matrices_diagblk_h[i]);
                }
                for(unsigned int i = 0; i < n_blocks-1; i++){
                    cudaFreeHost(batch_system_matrices_upperblk_h[i]);
                    cudaFreeHost(batch_system_matrices_lowerblk_h[i]);
                    cudaFreeHost(batch_self_energy_matrices_lesser_upperblk_h[i]);
                    cudaFreeHost(batch_self_energy_matrices_greater_upperblk_h[i]);
                    cudaFreeHost(batch_retarded_inv_matrices_upperblk_h[i]);
                    cudaFreeHost(batch_retarded_inv_matrices_lowerblk_h[i]);
                    cudaFreeHost(batch_lesser_inv_matrices_upperblk_h[i]);
                    cudaFreeHost(batch_greater_inv_matrices_upperblk_h[i]);
                }

            }
        }
    }
    return 0;
}








