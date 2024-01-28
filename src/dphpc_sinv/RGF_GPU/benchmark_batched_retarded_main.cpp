// Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
#include "batched_retarded.h"
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
    int nb_test = 1;
    int bs_test = 1;
    int n_blocks_input[nb_test] = {3*8};
    int blocksize_input[bs_test] = {1024};
    // int nb_test = 1;
    // int bs_test = 1;
    // int n_blocks_input[nb_test] =  {3*256};
    // int blocksize_input[bs_test] = {256};
    int number_of_measurements = 22;
    unsigned int batch_size = 100;
    for(int nb = 0; nb < nb_test; nb++){
        for(int bs = 0; bs < bs_test; bs++){
            unsigned int matrix_size = n_blocks_input[nb]*blocksize_input[bs];
            unsigned int blocksize = blocksize_input[bs];
            

            double memconsumption = batch_size * (3 * matrix_size * blocksize) * 16.0 / (1e9);
            int memconsumption_int = std::floor(memconsumption);
            std::cout << "memconsumption: " << memconsumption_int << std::endl;

            if(memconsumption_int > 64){
                std::cout << "Memory consumption too high, skipping" << std::endl;
                continue;
            }


            unsigned int n_blocks = matrix_size / blocksize;
            unsigned int off_diag_size = matrix_size - blocksize;

            // Print the matrix parameters
            printf("Matrix parameters:\n");
            printf("    Matrix size: %d\n", matrix_size);
            printf("    Block size: %d\n", blocksize);
            printf("    Number of blocks: %d\n", n_blocks);
            printf("    Batch size: %d\n", batch_size);

            complex_h *system_matrices_diagblk_h[batch_size];
            complex_h *system_matrices_upperblk_h[batch_size];
            complex_h *system_matrices_lowerblk_h[batch_size];

            for(unsigned int batch = 0; batch < batch_size; batch++){
                std::cout << "Loading batch " << batch << std::endl;
                // Load matrix to invert
                complex_h* system_matrix_diagblk = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
                std::string diagblk_path = base_path + "system_matrix_"+ std::to_string(0) +"_diagblk_"
                + std::to_string(matrix_size) + "_"+ std::to_string(blocksize) + "_" + std::to_string(batch_size) +
                ".bin";
                load_binary_matrix(diagblk_path.c_str(), system_matrix_diagblk, blocksize, matrix_size);

                complex_h* system_matrix_upperblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
                std::string upperblk_path = base_path + "system_matrix_"+ std::to_string(0) +"_upperblk_"
                + std::to_string(matrix_size) + "_"+ std::to_string(blocksize) + "_" + std::to_string(batch_size) +
                ".bin";
                load_binary_matrix(upperblk_path.c_str(), system_matrix_upperblk, blocksize, off_diag_size);

                complex_h* system_matrix_lowerblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
                std::string lowerblk_path = base_path + "system_matrix_"+ std::to_string(0) +"_lowerblk_"
                + std::to_string(matrix_size) + "_"+ std::to_string(blocksize) + "_" + std::to_string(batch_size) +
                ".bin";
                load_binary_matrix(lowerblk_path.c_str(), system_matrix_lowerblk, blocksize, off_diag_size);

                complex_h* system_matrix_diagblk_h = NULL;
                complex_h* system_matrix_upperblk_h = NULL;
                complex_h* system_matrix_lowerblk_h = NULL;

                cudaMallocHost((void**)&system_matrix_diagblk_h, blocksize * matrix_size * sizeof(complex_h));
                cudaMallocHost((void**)&system_matrix_upperblk_h, blocksize * off_diag_size * sizeof(complex_h));
                cudaMallocHost((void**)&system_matrix_lowerblk_h, blocksize * off_diag_size * sizeof(complex_h));

                transform_diagblk<complex_h>(system_matrix_diagblk, system_matrix_diagblk_h, blocksize, matrix_size);
                
                transform_offblk<complex_h>(system_matrix_upperblk, system_matrix_upperblk_h, blocksize, off_diag_size);
                transform_offblk<complex_h>(system_matrix_lowerblk, system_matrix_lowerblk_h, blocksize, off_diag_size);

                system_matrices_diagblk_h[batch] = system_matrix_diagblk_h;
                system_matrices_upperblk_h[batch] = system_matrix_upperblk_h;
                system_matrices_lowerblk_h[batch] = system_matrix_lowerblk_h;

                free(system_matrix_diagblk);
                free(system_matrix_upperblk);
                free(system_matrix_lowerblk);

            }

            // transform to batched blocks
            complex_h* batch_system_matrices_diagblk_h[n_blocks];
            complex_h* batch_system_matrices_upperblk_h[n_blocks-1];
            complex_h* batch_system_matrices_lowerblk_h[n_blocks-1];

            complex_h* batch_retarded_inv_matrices_diagblk_h[n_blocks];
            complex_h* batch_retarded_inv_matrices_upperblk_h[n_blocks-1];
            complex_h* batch_retarded_inv_matrices_lowerblk_h[n_blocks-1];

            for(unsigned int i = 0; i < n_blocks; i++){
                cudaErrchk(cudaMallocHost((void**)&batch_system_matrices_diagblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
                cudaErrchk(cudaMallocHost((void**)&batch_retarded_inv_matrices_diagblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
                for(unsigned int batch = 0; batch < batch_size; batch++){
                    for(unsigned int j = 0; j < blocksize * blocksize; j++){
                        batch_system_matrices_diagblk_h[i][batch * blocksize * blocksize + j] = system_matrices_diagblk_h[batch][i * blocksize * blocksize + j];
                    }
                }
            }
            for(unsigned int i = 0; i < n_blocks-1; i++){
                cudaErrchk(cudaMallocHost((void**)&batch_system_matrices_upperblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
                cudaErrchk(cudaMallocHost((void**)&batch_system_matrices_lowerblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
                cudaErrchk(cudaMallocHost((void**)&batch_retarded_inv_matrices_upperblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
                cudaErrchk(cudaMallocHost((void**)&batch_retarded_inv_matrices_lowerblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
                for(unsigned int batch = 0; batch < batch_size; batch++){
                    for(unsigned int j = 0; j < blocksize * blocksize; j++){
                        batch_system_matrices_upperblk_h[i][batch * blocksize * blocksize + j] = system_matrices_upperblk_h[batch][i * blocksize * blocksize + j];
                        batch_system_matrices_lowerblk_h[i][batch * blocksize * blocksize + j] = system_matrices_lowerblk_h[batch][i * blocksize * blocksize + j];
                    }
                }
            }


            double times_retarded_for[number_of_measurements];
            double times_retarded_batched[number_of_measurements];
            double time;

            for(int i = 0; i < number_of_measurements; i++){
                std::cout << "For Loop Measurement " << i << std::endl;
                time = -omp_get_wtime();
                rgf_retarded_for(blocksize, matrix_size, batch_size,
                                            batch_system_matrices_diagblk_h,
                                            batch_system_matrices_upperblk_h,
                                            batch_system_matrices_lowerblk_h,
                                            batch_retarded_inv_matrices_diagblk_h,
                                            batch_retarded_inv_matrices_upperblk_h,
                                            batch_retarded_inv_matrices_lowerblk_h);
                time += omp_get_wtime();
                std::cout << "Time: " << time << std::endl;
                times_retarded_for[i] = time;
            }


            for(int i = 0; i < number_of_measurements; i++){
                std::cout << "Batched Measurement " << i << std::endl;
                time = -omp_get_wtime();
                rgf_retarded_batched(blocksize, matrix_size, batch_size,
                                            batch_system_matrices_diagblk_h,
                                            batch_system_matrices_upperblk_h,
                                            batch_system_matrices_lowerblk_h,
                                            batch_retarded_inv_matrices_diagblk_h,
                                            batch_retarded_inv_matrices_upperblk_h,
                                            batch_retarded_inv_matrices_lowerblk_h);
                time += omp_get_wtime();
                std::cout << "Time: " << time << std::endl;
                times_retarded_batched[i] = time;
                
            }



            std::ofstream outputFile_times_retarded_for;
            std::string filename = time_path + "times_retarded_for_" + std::to_string(matrix_size) + "_"   + std::to_string(n_blocks) + "_"  + std::to_string(batch_size) + ".txt";
            outputFile_times_retarded_for.open(filename);
            if(outputFile_times_retarded_for.is_open()){
                for(int i = 0; i < number_of_measurements; i++){
                    outputFile_times_retarded_for << times_retarded_for[i] << std::endl;
                }
            }
            else{
                std::cout << "Unable to open file" << std::endl;
            }
            outputFile_times_retarded_for.close();

            std::ofstream outputFile_times_retarded_batched;
            filename = time_path + "times_retarded_batched_" + std::to_string(matrix_size) + "_"   + std::to_string(n_blocks) + "_"  + std::to_string(batch_size) + ".txt";
            outputFile_times_retarded_batched.open(filename);
            if(outputFile_times_retarded_batched.is_open()){
                for(int i = 0; i < number_of_measurements; i++){
                    outputFile_times_retarded_batched << times_retarded_batched[i] << std::endl;
                }
            }
            else{
                std::cout << "Unable to open file" << std::endl;
            }
            outputFile_times_retarded_batched.close();

            for(unsigned int i = 0; i < n_blocks; i++){
                cudaFreeHost(batch_system_matrices_diagblk_h[i]);
                cudaFreeHost(batch_retarded_inv_matrices_diagblk_h[i]);
            }
            for(unsigned int i = 0; i < n_blocks-1; i++){
                cudaFreeHost(batch_system_matrices_upperblk_h[i]);
                cudaFreeHost(batch_system_matrices_lowerblk_h[i]);
                cudaFreeHost(batch_retarded_inv_matrices_upperblk_h[i]);
                cudaFreeHost(batch_retarded_inv_matrices_lowerblk_h[i]);
            }

            for(unsigned int batch = 0; batch < batch_size; batch++){
                cudaFreeHost(system_matrices_diagblk_h[batch]);
                cudaFreeHost(system_matrices_upperblk_h[batch]);
                cudaFreeHost(system_matrices_lowerblk_h[batch]);
            }
        }
    }
    return 0;
}








