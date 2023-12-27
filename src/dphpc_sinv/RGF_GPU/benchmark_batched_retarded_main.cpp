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

    complex_h *system_matrices_diagblk_h[batch_size];
    complex_h *system_matrices_upperblk_h[batch_size];
    complex_h *system_matrices_lowerblk_h[batch_size];

    complex_h *retarded_inv_matrices_diagblk_ref[batch_size];
    complex_h *retarded_inv_matrices_upperblk_ref[batch_size];
    complex_h *retarded_inv_matrices_lowerblk_ref[batch_size];

    for(unsigned int batch = 0; batch < batch_size; batch++){
        std::cout << "Loading batch " << batch << std::endl;
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


        complex_h* retarded_inv_diagblk = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
        std::string retarded_diagblk_path = base_path + "retarded_"+ std::to_string(batch) +"_inv_diagblk_"
        + std::to_string(matrix_size) + "_"+ std::to_string(blocksize) + "_" + std::to_string(batch_size) +
        ".bin";
        load_binary_matrix(retarded_diagblk_path.c_str(), retarded_inv_diagblk, blocksize, matrix_size);

        complex_h* retarded_inv_upperblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
        std::string retarded_upperblk_path = base_path + "retarded_"+ std::to_string(batch) +"_inv_upperblk_"
        + std::to_string(matrix_size) + "_"+ std::to_string(blocksize) + "_" + std::to_string(batch_size) +
        ".bin";
        load_binary_matrix(retarded_upperblk_path.c_str(), retarded_inv_upperblk, blocksize, off_diag_size);

        complex_h* retarded_inv_lowerblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
        std::string retarded_lowerblk_path = base_path + "retarded_"+ std::to_string(batch) +"_inv_lowerblk_"
        + std::to_string(matrix_size) + "_"+ std::to_string(blocksize) + "_" + std::to_string(batch_size) +
        ".bin";
        load_binary_matrix(retarded_lowerblk_path.c_str(), retarded_inv_lowerblk, blocksize, off_diag_size);

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
        complex_h* retarded_inv_diagblk_ref = NULL;
        complex_h* retarded_inv_upperblk_ref = NULL;
        complex_h* retarded_inv_lowerblk_ref = NULL;
        cudaMallocHost((void**)&system_matrix_diagblk_h, blocksize * matrix_size * sizeof(complex_h));
        cudaMallocHost((void**)&system_matrix_upperblk_h, blocksize * off_diag_size * sizeof(complex_h));
        cudaMallocHost((void**)&system_matrix_lowerblk_h, blocksize * off_diag_size * sizeof(complex_h));
        cudaMallocHost((void**)&retarded_inv_diagblk_ref, blocksize * matrix_size * sizeof(complex_h));
        cudaMallocHost((void**)&retarded_inv_upperblk_ref, blocksize * off_diag_size * sizeof(complex_h));
        cudaMallocHost((void**)&retarded_inv_lowerblk_ref, blocksize * off_diag_size * sizeof(complex_h));

        transform_diagblk<complex_h>(system_matrix_diagblk, system_matrix_diagblk_h, blocksize, matrix_size);
        transform_diagblk<complex_h>(retarded_inv_diagblk, retarded_inv_diagblk_ref, blocksize, matrix_size);
        
        transform_offblk<complex_h>(system_matrix_upperblk, system_matrix_upperblk_h, blocksize, off_diag_size);
        transform_offblk<complex_h>(system_matrix_lowerblk, system_matrix_lowerblk_h, blocksize, off_diag_size);
        transform_offblk<complex_h>(retarded_inv_upperblk, retarded_inv_upperblk_ref, blocksize, off_diag_size);
        transform_offblk<complex_h>(retarded_inv_lowerblk, retarded_inv_lowerblk_ref, blocksize, off_diag_size);

        system_matrices_diagblk_h[batch] = system_matrix_diagblk_h;
        system_matrices_upperblk_h[batch] = system_matrix_upperblk_h;
        system_matrices_lowerblk_h[batch] = system_matrix_lowerblk_h;

        retarded_inv_matrices_diagblk_ref[batch] = retarded_inv_diagblk_ref;
        retarded_inv_matrices_upperblk_ref[batch] = retarded_inv_upperblk_ref;
        retarded_inv_matrices_lowerblk_ref[batch] = retarded_inv_lowerblk_ref;


        // free non contiguous memory
        free(system_matrix_diagblk);
        free(system_matrix_upperblk);
        free(system_matrix_lowerblk);
        free(retarded_inv_diagblk);
        free(retarded_inv_upperblk);
        free(retarded_inv_lowerblk);
    }

    // transform to batched blocks
    complex_h* batch_system_matrices_diagblk_h[n_blocks];
    complex_h* batch_system_matrices_upperblk_h[n_blocks-1];
    complex_h* batch_system_matrices_lowerblk_h[n_blocks-1];
    complex_h* batch_retarded_inv_matrices_diagblk_ref[n_blocks];
    complex_h* batch_retarded_inv_matrices_upperblk_ref[n_blocks-1];
    complex_h* batch_retarded_inv_matrices_lowerblk_ref[n_blocks-1];
    complex_h* batch_retarded_inv_matrices_diagblk_h[n_blocks];
    complex_h* batch_retarded_inv_matrices_upperblk_h[n_blocks-1];
    complex_h* batch_retarded_inv_matrices_lowerblk_h[n_blocks-1];

    for(unsigned int i = 0; i < n_blocks; i++){
        cudaErrchk(cudaMallocHost((void**)&batch_system_matrices_diagblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
        cudaErrchk(cudaMallocHost((void**)&batch_retarded_inv_matrices_diagblk_ref[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
        cudaErrchk(cudaMallocHost((void**)&batch_retarded_inv_matrices_diagblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
        for(unsigned int batch = 0; batch < batch_size; batch++){
            for(unsigned int j = 0; j < blocksize * blocksize; j++){
                batch_system_matrices_diagblk_h[i][batch * blocksize * blocksize + j] = system_matrices_diagblk_h[batch][i * blocksize * blocksize + j];
                batch_retarded_inv_matrices_diagblk_ref[i][batch * blocksize * blocksize + j] = retarded_inv_matrices_diagblk_ref[batch][i * blocksize * blocksize + j];
            }
        }
    }
    for(unsigned int i = 0; i < n_blocks-1; i++){
        cudaErrchk(cudaMallocHost((void**)&batch_system_matrices_upperblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
        cudaErrchk(cudaMallocHost((void**)&batch_system_matrices_lowerblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
        cudaErrchk(cudaMallocHost((void**)&batch_retarded_inv_matrices_upperblk_ref[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
        cudaErrchk(cudaMallocHost((void**)&batch_retarded_inv_matrices_lowerblk_ref[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
        cudaErrchk(cudaMallocHost((void**)&batch_retarded_inv_matrices_upperblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
        cudaErrchk(cudaMallocHost((void**)&batch_retarded_inv_matrices_lowerblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
        for(unsigned int batch = 0; batch < batch_size; batch++){
            for(unsigned int j = 0; j < blocksize * blocksize; j++){
                batch_system_matrices_upperblk_h[i][batch * blocksize * blocksize + j] = system_matrices_upperblk_h[batch][i * blocksize * blocksize + j];
                batch_system_matrices_lowerblk_h[i][batch * blocksize * blocksize + j] = system_matrices_lowerblk_h[batch][i * blocksize * blocksize + j];
                batch_retarded_inv_matrices_upperblk_ref[i][batch * blocksize * blocksize + j] = retarded_inv_matrices_upperblk_ref[batch][i * blocksize * blocksize + j];
                batch_retarded_inv_matrices_lowerblk_ref[i][batch * blocksize * blocksize + j] = retarded_inv_matrices_lowerblk_ref[batch][i * blocksize * blocksize + j];
            }
        }
    }

    int number_of_measurements = 110;
    double times_retarded_for[number_of_measurements];
    double times_retarded_batched[number_of_measurements];
    double times_retarded_batched_strided[number_of_measurements];
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
        times_retarded_batched[i] = time;
        
    }

    for(int i = 0; i < number_of_measurements; i++){
        std::cout << "Batched Strided Measurement " << i << std::endl;
        time = -omp_get_wtime();
        rgf_retarded_batched_strided(blocksize, matrix_size, batch_size,
                                    batch_system_matrices_diagblk_h,
                                    batch_system_matrices_upperblk_h,
                                    batch_system_matrices_lowerblk_h,
                                    batch_retarded_inv_matrices_diagblk_h,
                                    batch_retarded_inv_matrices_upperblk_h,
                                    batch_retarded_inv_matrices_lowerblk_h);
        time += omp_get_wtime();
        times_retarded_batched_strided[i] = time;
        
    }

    std::string time_path = "/usr/scratch/mont-fort17/almaeder/rgf_bench/times/";

    std::ofstream outputFile_times_retarded_for;
    std::string filename = time_path + "times_retarded_for_" + std::to_string(matrix_size) + "_"   + std::to_string(n_blocks) + "_"  + std::to_string(batch_size) + "_"+ ".txt";
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
    filename = time_path + "times_retarded_batched_" + std::to_string(matrix_size) + "_"   + std::to_string(n_blocks) + "_"  + std::to_string(batch_size) + "_"+ ".txt";
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

    std::ofstream outputFile_times_retarded_batched_strided;
    filename = time_path + "times_retarded_batched_strided_" + std::to_string(matrix_size) + "_"   + std::to_string(n_blocks) + "_"  + std::to_string(batch_size) + "_"+ ".txt";
    outputFile_times_retarded_batched_strided.open(filename);
    if(outputFile_times_retarded_batched_strided.is_open()){
        for(int i = 0; i < number_of_measurements; i++){
            outputFile_times_retarded_batched_strided << times_retarded_batched_strided[i] << std::endl;
        }
    }
    else{
        std::cout << "Unable to open file" << std::endl;
    }
    outputFile_times_retarded_batched_strided.close();

    //free batched memory
    for(unsigned int i = 0; i < n_blocks; i++){
        cudaFreeHost(batch_system_matrices_diagblk_h[i]);
        cudaFreeHost(batch_retarded_inv_matrices_diagblk_ref[i]);
        cudaFreeHost(batch_retarded_inv_matrices_diagblk_h[i]);
    }
    for(unsigned int i = 0; i < n_blocks-1; i++){
        cudaFreeHost(batch_system_matrices_upperblk_h[i]);
        cudaFreeHost(batch_system_matrices_lowerblk_h[i]);
        cudaFreeHost(batch_retarded_inv_matrices_upperblk_ref[i]);
        cudaFreeHost(batch_retarded_inv_matrices_lowerblk_ref[i]);
        cudaFreeHost(batch_retarded_inv_matrices_upperblk_h[i]);
        cudaFreeHost(batch_retarded_inv_matrices_lowerblk_h[i]);
    }

    // free contiguous memory
    for(unsigned int batch = 0; batch < batch_size; batch++){
        cudaFreeHost(system_matrices_diagblk_h[batch]);
        cudaFreeHost(system_matrices_upperblk_h[batch]);
        cudaFreeHost(system_matrices_lowerblk_h[batch]);
        cudaFreeHost(retarded_inv_matrices_diagblk_ref[batch]);
        cudaFreeHost(retarded_inv_matrices_upperblk_ref[batch]);
        cudaFreeHost(retarded_inv_matrices_lowerblk_ref[batch]);
    }
    return 0;
}








