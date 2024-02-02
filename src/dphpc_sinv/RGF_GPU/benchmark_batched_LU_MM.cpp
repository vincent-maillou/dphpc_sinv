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
    int bs_test = 6;
    int nbatch_test = 10;
    int batch_sizes_input[nbatch_test] = {1, 2, 4, 8, 16, 32, 48, 64, 96, 128};
    int blocksize_input[bs_test] = {64, 128, 256, 512, 768, 1024};

    // int nb_test = 1;
    // int bs_test = 1;
    // int n_blocks_input[nb_test] = {3};
    // //int blocksize_input[bs_test] = {64, 128, 256, 512};
    // int blocksize_input[bs_test] = {1024};
    int number_of_measurements = 22;    

    for(int bas = 0; bas < nbatch_test; bas++){
        for(int bs = 0; bs < bs_test; bs++){
            unsigned int batch_size = batch_sizes_input[bas];                
            unsigned int blocksize = blocksize_input[bs];
            

            double memconsumption = 5*batch_size * blocksize * blocksize * 16.0 / (1e9);
            int memconsumption_int = std::floor(memconsumption);
            std::cout << "memconsumption: " << memconsumption_int << std::endl;

            if(memconsumption_int > 12){
                std::cout << "Memory consumption too high, skipping" << std::endl;
                continue;
            }

            // Print the matrix parameters
            printf("    Block size: %d\n", blocksize);
            printf("    Batch size: %d\n", batch_size);


            std::string diagblk_path = base_path + "system_matrix_diagblk_" + std::to_string(blocksize) +
                "_" + std::to_string(batch_size) +
                ".bin";

            complex_h *mm_A_h;
            complex_h *mm_B_h;
            complex_h *mm_C_h;

            complex_h *mm_LU_in_h;
            complex_h *mm_LU_out_h;
            cudaErrchk(cudaMallocHost((void**)&mm_A_h, blocksize*blocksize*batch_size*sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&mm_B_h, blocksize*blocksize*batch_size*sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&mm_C_h, blocksize*blocksize*batch_size*sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&mm_LU_in_h, blocksize*blocksize*batch_size*sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&mm_LU_out_h, blocksize*blocksize*batch_size*sizeof(complex_h)));

            load_binary_array<complex_h>(diagblk_path, mm_A_h, blocksize*blocksize*batch_size);
            load_binary_array<complex_h>(diagblk_path, mm_B_h, blocksize*blocksize*batch_size);
            load_binary_array<complex_h>(diagblk_path, mm_C_h, blocksize*blocksize*batch_size);
            load_binary_array<complex_h>(diagblk_path, mm_LU_in_h, blocksize*blocksize*batch_size);
            load_binary_array<complex_h>(diagblk_path, mm_LU_out_h, blocksize*blocksize*batch_size);

            complex_d *mm_A_d;
            complex_d *mm_B_d;
            complex_d *mm_C_d;
            complex_d *mm_LU_in_d;
            complex_d *mm_LU_out_d;

            cudaErrchk(cudaMalloc((void**)&mm_A_d, blocksize*blocksize*batch_size*sizeof(complex_d)));
            cudaErrchk(cudaMalloc((void**)&mm_B_d, blocksize*blocksize*batch_size*sizeof(complex_d)));
            cudaErrchk(cudaMalloc((void**)&mm_C_d, blocksize*blocksize*batch_size*sizeof(complex_d)));
            cudaErrchk(cudaMalloc((void**)&mm_LU_in_d, blocksize*blocksize*batch_size*sizeof(complex_d)));
            cudaErrchk(cudaMalloc((void**)&mm_LU_out_d, blocksize*blocksize*batch_size*sizeof(complex_d)));

            cudaErrchk(cudaMemcpy(mm_A_d, mm_A_h, blocksize*blocksize*batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));
            cudaErrchk(cudaMemcpy(mm_B_d, mm_B_h, blocksize*blocksize*batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));
            cudaErrchk(cudaMemcpy(mm_C_d, mm_C_h, blocksize*blocksize*batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));
            cudaErrchk(cudaMemcpy(mm_LU_in_d, mm_LU_in_h, blocksize*blocksize*batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));
            cudaErrchk(cudaMemcpy(mm_LU_out_d, mm_LU_out_h, blocksize*blocksize*batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));

            complex_d *mm_A_ptr_h[batch_size];
            complex_d *mm_B_ptr_h[batch_size];
            complex_d *mm_C_ptr_h[batch_size];
            complex_d *mm_LU_in_ptr_h[batch_size];
            complex_d *mm_LU_out_ptr_h[batch_size];

            for(unsigned int i = 0; i < batch_size; i++){
                mm_A_ptr_h[i] = mm_A_d + i*blocksize*blocksize;
                mm_B_ptr_h[i] = mm_B_d + i*blocksize*blocksize;
                mm_C_ptr_h[i] = mm_C_d + i*blocksize*blocksize;
                mm_LU_in_ptr_h[i] = mm_LU_in_d + i*blocksize*blocksize;
                mm_LU_out_ptr_h[i] = mm_LU_out_d + i*blocksize*blocksize;
            }

            complex_d **mm_A_ptr_d;
            complex_d **mm_B_ptr_d;
            complex_d **mm_C_ptr_d;
            complex_d **mm_LU_in_ptr_d;
            complex_d **mm_LU_out_ptr_d;

            cudaErrchk(cudaMalloc((void**)&mm_A_ptr_d, batch_size*sizeof(complex_d*)));
            cudaErrchk(cudaMalloc((void**)&mm_B_ptr_d, batch_size*sizeof(complex_d*)));
            cudaErrchk(cudaMalloc((void**)&mm_C_ptr_d, batch_size*sizeof(complex_d*)));
            cudaErrchk(cudaMalloc((void**)&mm_LU_in_ptr_d, batch_size*sizeof(complex_d*)));
            cudaErrchk(cudaMalloc((void**)&mm_LU_out_ptr_d, batch_size*sizeof(complex_d*)));

            cudaErrchk(cudaMemcpy(mm_A_ptr_d, mm_A_ptr_h, batch_size*sizeof(complex_d*), cudaMemcpyHostToDevice));
            cudaErrchk(cudaMemcpy(mm_B_ptr_d, mm_B_ptr_h, batch_size*sizeof(complex_d*), cudaMemcpyHostToDevice));
            cudaErrchk(cudaMemcpy(mm_C_ptr_d, mm_C_ptr_h, batch_size*sizeof(complex_d*), cudaMemcpyHostToDevice));
            cudaErrchk(cudaMemcpy(mm_LU_in_ptr_d, mm_LU_in_ptr_h, batch_size*sizeof(complex_d*), cudaMemcpyHostToDevice));
            cudaErrchk(cudaMemcpy(mm_LU_out_ptr_d, mm_LU_out_ptr_h, batch_size*sizeof(complex_d*), cudaMemcpyHostToDevice));

            int *ipiv_d = NULL;
            int *info_d = NULL;
            cudaErrchk(cudaMalloc((void**)&info_d, batch_size * sizeof(int)))
            cudaErrchk(cudaMalloc((void**)&ipiv_d, batch_size * blocksize*sizeof(int)));

            cudaStream_t stream;
            cublasHandle_t cublas_handle;
            cublasErrchk(cublasCreate(&cublas_handle));
            cusolverDnHandle_t cusolver_handle;
            cusolverErrchk(cusolverDnCreate(&cusolver_handle));
            cudaErrchk(cudaStreamCreate(&stream));
            cublasErrchk(cublasSetStream(cublas_handle, stream));
            cusolverErrchk(cusolverDnSetStream(cusolver_handle, stream));
            complex_d alpha;
            complex_d beta;
            alpha = make_cuDoubleComplex(1.0, 0.0);
            beta = make_cuDoubleComplex(0.0, 0.0);

            double times_mm_batched[number_of_measurements];
            double times_mm_for[number_of_measurements];
            double times_mm_strided[number_of_measurements];
            double times_inv_batched[number_of_measurements];
            double times_inv_for[number_of_measurements];
            double times_getrf_batched[number_of_measurements];
            double times_getrf_for[number_of_measurements];
            double times_getri_batched[number_of_measurements];
            double times_getrs_for[number_of_measurements];
            double times_solve_batched[number_of_measurements];
            double times_getrs_batched[number_of_measurements];
            double time;
            for(int i = 0; i < number_of_measurements; i++){
                std::cout << "Batched MM Measurement " << i << std::endl;
                time = -omp_get_wtime();
                cudaErrchk(cudaStreamSynchronize(stream));
                cublasErrchk(cublasZgemmBatched(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,   
                    blocksize, blocksize, blocksize,
                    &alpha,
                    mm_A_ptr_d, blocksize,
                    mm_B_ptr_d, blocksize,
                    &beta,
                    mm_C_ptr_d, blocksize, batch_size));
                cudaErrchk(cudaStreamSynchronize(stream));
                time += omp_get_wtime();
                std::cout << "Time: " << time << std::endl;
                times_mm_batched[i] = time;
            }
            for(int i = 0; i < number_of_measurements; i++){
                std::cout << "Strided Batched MM Measurement " << i << std::endl;
                time = -omp_get_wtime();
                cudaErrchk(cudaStreamSynchronize(stream));
                cublasErrchk(cublasZgemmStridedBatched(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,   
                    blocksize, blocksize, blocksize,
                    &alpha,
                    mm_A_d, blocksize,
                    blocksize*blocksize,
                    mm_B_d, blocksize,
                    blocksize*blocksize,
                    &beta,
                    mm_C_d, blocksize,
                    blocksize*blocksize,
                    batch_size));
                cudaErrchk(cudaStreamSynchronize(stream));
                time += omp_get_wtime();
                std::cout << "Time: " << time << std::endl;
                times_mm_strided[i] = time;
            }
            for(int i = 0; i < number_of_measurements; i++){
                std::cout << "For MM Measurement " << i << std::endl;
                time = -omp_get_wtime();
                cudaErrchk(cudaStreamSynchronize(stream));
                for(unsigned int batch = 0; batch < batch_size; batch++){
                    cublasErrchk(cublasZgemm(
                        cublas_handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,   
                        blocksize, blocksize, blocksize,
                        &alpha,
                        mm_A_ptr_h[batch], blocksize,
                        mm_B_ptr_h[batch], blocksize,
                        &beta,
                        mm_C_ptr_h[batch], blocksize));
                }
                cudaErrchk(cudaStreamSynchronize(stream));
                time += omp_get_wtime();
                std::cout << "Time: " << time << std::endl;
                times_mm_for[i] = time;
            }


            for(int i = 0; i < number_of_measurements; i++){
                std::cout << "Batched LU Measurement " << i << std::endl;
                cudaErrchk(cudaMemcpy(mm_LU_in_d, mm_LU_in_h, blocksize*blocksize*batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));
                cudaErrchk(cudaMemcpy(mm_LU_out_d, mm_LU_out_h, blocksize*blocksize*batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));
                time = -omp_get_wtime();
                cudaErrchk(cudaStreamSynchronize(stream));
                cublasErrchk(cublasZgetrfBatched(
                        cublas_handle,
                        blocksize,
                        mm_LU_in_ptr_d,
                        blocksize,
                        ipiv_d,
                        info_d,
                        batch_size));
                // inversion
                cublasErrchk(cublasZgetriBatched(
                        cublas_handle,
                        blocksize,
                        mm_LU_in_ptr_d,
                        blocksize,
                        ipiv_d,
                        mm_LU_out_ptr_d,
                        blocksize,
                        info_d,
                        batch_size));
                cudaErrchk(cudaStreamSynchronize(stream));
                time += omp_get_wtime();
                std::cout << "Time: " << time << std::endl;
                times_inv_batched[i] = time;
            }




            for(int i = 0; i < number_of_measurements; i++){
                std::cout << "Batched getrf Measurement " << i << std::endl;
                cudaErrchk(cudaMemcpy(mm_LU_in_d, mm_LU_in_h, blocksize*blocksize*batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));
                time = -omp_get_wtime();
                cudaErrchk(cudaStreamSynchronize(stream));
                cublasErrchk(cublasZgetrfBatched(
                        cublas_handle,
                        blocksize,
                        mm_LU_in_ptr_d,
                        blocksize,
                        ipiv_d,
                        info_d,
                        batch_size));
                cudaErrchk(cudaStreamSynchronize(stream));
                time += omp_get_wtime();
                std::cout << "Time: " << time << std::endl;
                times_getrf_batched[i] = time;
            }

            for(int i = 0; i < number_of_measurements; i++){
                std::cout << "Batched getri Measurement " << i << std::endl;
                cudaErrchk(cudaMemcpy(mm_LU_out_d, mm_LU_out_h, blocksize*blocksize*batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));
                time = -omp_get_wtime();
                cudaErrchk(cudaStreamSynchronize(stream));
                // inversion
                cublasErrchk(cublasZgetriBatched(
                        cublas_handle,
                        blocksize,
                        mm_LU_in_ptr_d,
                        blocksize,
                        ipiv_d,
                        mm_LU_out_ptr_d,
                        blocksize,
                        info_d,
                        batch_size));
                cudaErrchk(cudaStreamSynchronize(stream));
                time += omp_get_wtime();
                std::cout << "Time: " << time << std::endl;
                times_getri_batched[i] = time;
            }

            complex_h* identity_h;
            cudaErrchk(cudaMallocHost((void**)&identity_h, blocksize * blocksize * sizeof(complex_h)));
            complex_d* identity_d = NULL;
            cudaErrchk(cudaMalloc((void**)&identity_d, blocksize * blocksize * sizeof(complex_d)));


            for(unsigned int i = 0; i < blocksize * blocksize; i++){
                identity_h[i] = 0.0;
                if(i / blocksize == i % blocksize){
                    identity_h[i] = 1.0;
                }
            }
            // init right hand side identity matrix on device for backsub
            cudaErrchk(cudaMemcpy(identity_d, reinterpret_cast<const complex_d*>(identity_h),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice));
            



            //figure out extra amount of memory needed
            complex_d *buffer = NULL;
            int bufferSize = 0;
            cusolverErrchk(cusolverDnZgetrf_bufferSize(cusolver_handle, blocksize, blocksize,
                                                    (complex_d *)mm_LU_in_d,
                                                    blocksize, &bufferSize));
            cudaErrchk(cudaMalloc((void**)&buffer, sizeof(complex_d) * bufferSize));


            for(int i = 0; i < number_of_measurements; i++){
                cudaErrchk(cudaMemcpy(mm_LU_in_d, mm_LU_in_h, blocksize*blocksize*batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));
                for(unsigned int batch = 0; batch < batch_size; batch++){
                    cudaErrchk(cudaMemcpy(mm_LU_out_d + batch*blocksize*blocksize, identity_h, blocksize*blocksize*sizeof(complex_d), cudaMemcpyHostToDevice));
                }
                std::cout << "For LU Measurement " << i << std::endl;
                time = -omp_get_wtime();
                cudaErrchk(cudaStreamSynchronize(stream));
                for(unsigned int batch = 0; batch < batch_size; batch++){
                    cusolverErrchk(cusolverDnZgetrf(cusolver_handle,
                        blocksize, blocksize,
                        mm_LU_in_ptr_h[batch],
                        blocksize, buffer,
                        ipiv_d, info_d));
                    
                    //back substitution
                    cusolverErrchk(cusolverDnZgetrs(cusolver_handle,
                        CUBLAS_OP_N,
                        blocksize, blocksize,
                        mm_LU_in_ptr_h[batch],
                        blocksize, ipiv_d,
                        mm_LU_out_ptr_h[batch],
                        blocksize, info_d));

                }
                cudaErrchk(cudaStreamSynchronize(stream));
                time += omp_get_wtime();
                std::cout << "Time: " << time << std::endl;
                times_inv_for[i] = time;
            }
            for(int i = 0; i < number_of_measurements; i++){
                cudaErrchk(cudaMemcpy(mm_LU_in_d, mm_LU_in_h, blocksize*blocksize*batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));
                std::cout << "For cusolverDnZgetrf " << i << std::endl;
                time = -omp_get_wtime();
                cudaErrchk(cudaStreamSynchronize(stream));
                for(unsigned int batch = 0; batch < batch_size; batch++){
                    cusolverErrchk(cusolverDnZgetrf(cusolver_handle,
                        blocksize, blocksize,
                        mm_LU_in_ptr_h[batch],
                        blocksize, buffer,
                        ipiv_d, info_d));
                }
                cudaErrchk(cudaStreamSynchronize(stream));
                time += omp_get_wtime();
                std::cout << "Time: " << time << std::endl;
                times_getrf_for[i] = time;
            }
            for(int i = 0; i < number_of_measurements; i++){
                for(unsigned int batch = 0; batch < batch_size; batch++){
                    cudaErrchk(cudaMemcpy(mm_LU_out_d + batch*blocksize*blocksize, identity_d, batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));
                }
                std::cout << "For cusolverDnZgetrs Measurement " << i << std::endl;
                time = -omp_get_wtime();
                cudaErrchk(cudaStreamSynchronize(stream));
                for(unsigned int batch = 0; batch < batch_size; batch++){
                    //back substitution
                    cusolverErrchk(cusolverDnZgetrs(cusolver_handle,
                        CUBLAS_OP_N,
                        blocksize, blocksize,
                        mm_LU_in_ptr_h[batch],
                        blocksize, ipiv_d,
                        mm_LU_out_ptr_h[batch],
                        blocksize, info_d));

                }
                cudaErrchk(cudaStreamSynchronize(stream));
                time += omp_get_wtime();
                std::cout << "Time: " << time << std::endl;
                times_getrs_for[i] = time;
            }

            for(int i = 0; i < number_of_measurements; i++){
                std::cout << "Batched solve Measurement " << i << std::endl;
                cudaErrchk(cudaMemcpy(mm_LU_in_d, mm_LU_in_h, blocksize*blocksize*batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));
                for(unsigned int batch = 0; batch < batch_size; batch++){
                    cudaErrchk(cudaMemcpy(mm_LU_out_d + batch*blocksize*blocksize, identity_h, blocksize*blocksize*sizeof(complex_d), cudaMemcpyHostToDevice));
                }
                int info_h;
                time = -omp_get_wtime();
                cudaErrchk(cudaStreamSynchronize(stream));
                cublasErrchk(cublasZgetrfBatched(
                        cublas_handle,
                        blocksize,
                        mm_LU_in_ptr_d,
                        blocksize,
                        ipiv_d,
                        info_d,
                        batch_size));
                // inversion
                cublasErrchk(cublasZgetrsBatched(
                        cublas_handle,
                        CUBLAS_OP_N,
                        blocksize,
                        blocksize,
                        mm_LU_in_ptr_d,
                        blocksize,
                        ipiv_d,
                        mm_LU_out_ptr_d,
                        blocksize,
                        &info_h,
                        batch_size));
                cudaErrchk(cudaStreamSynchronize(stream));
                time += omp_get_wtime();
                std::cout << "Time: " << time << std::endl;
                times_solve_batched[i] = time;
            }

            for(int i = 0; i < number_of_measurements; i++){
                std::cout << "Batched getrs Measurement " << i << std::endl;
                for(unsigned int batch = 0; batch < batch_size; batch++){
                    cudaErrchk(cudaMemcpy(mm_LU_out_d + batch*blocksize*blocksize, identity_h, blocksize*blocksize*sizeof(complex_d), cudaMemcpyHostToDevice));
                }
                int info_h;
                time = -omp_get_wtime();
                cublasErrchk(cublasZgetrsBatched(
                        cublas_handle,
                        CUBLAS_OP_N,
                        blocksize,
                        blocksize,
                        mm_LU_in_ptr_d,
                        blocksize,
                        ipiv_d,
                        mm_LU_out_ptr_d,
                        blocksize,
                        &info_h,
                        batch_size));
                cudaErrchk(cudaStreamSynchronize(stream));
                time += omp_get_wtime();
                std::cout << "Time: " << time << std::endl;
                times_getrs_batched[i] = time;
            }

            std::string filename;
            std::ofstream outputFile;

            filename = time_path + "times_mm_batched_" + std::to_string(blocksize) + "_"  + std::to_string(batch_size) + ".txt";
            outputFile.open(filename);
            if(outputFile.is_open()){
                for(int i = 0; i < number_of_measurements; i++){
                    outputFile << times_mm_batched[i] << std::endl;
                }
            }
            else{
                std::cout << "Unable to open file" << std::endl;
            }
            outputFile.close();

            filename = time_path + "times_mm_strided_" + std::to_string(blocksize) + "_"  + std::to_string(batch_size) + ".txt";
            outputFile.open(filename);
            if(outputFile.is_open()){
                for(int i = 0; i < number_of_measurements; i++){
                    outputFile << times_mm_strided[i] << std::endl;
                }
            }
            else{
                std::cout << "Unable to open file" << std::endl;
            }
            outputFile.close();

            filename = time_path + "times_mm_for_" + std::to_string(blocksize) + "_"  + std::to_string(batch_size) + ".txt";
            outputFile.open(filename);
            if(outputFile.is_open()){
                for(int i = 0; i < number_of_measurements; i++){
                    outputFile << times_mm_for[i] << std::endl;
                }
            }
            else{
                std::cout << "Unable to open file" << std::endl;
            }
            outputFile.close();

            filename = time_path + "times_inv_batched_" + std::to_string(blocksize) + "_"  + std::to_string(batch_size) + ".txt";
            outputFile.open(filename);
            if(outputFile.is_open()){
                for(int i = 0; i < number_of_measurements; i++){
                    outputFile << times_inv_batched[i] << std::endl;
                }
            }
            else{
                std::cout << "Unable to open file" << std::endl;
            }
            outputFile.close();

            filename = time_path + "times_inv_for_" + std::to_string(blocksize) + "_"  + std::to_string(batch_size) + ".txt";
            outputFile.open(filename);
            if(outputFile.is_open()){
                for(int i = 0; i < number_of_measurements; i++){
                    outputFile << times_inv_for[i] << std::endl;
                }
            }
            else{
                std::cout << "Unable to open file" << std::endl;
            }
            outputFile.close();

            filename = time_path + "times_getrf_batched_" + std::to_string(blocksize) + "_"  + std::to_string(batch_size) + ".txt";
            outputFile.open(filename);
            if(outputFile.is_open()){
                for(int i = 0; i < number_of_measurements; i++){
                    outputFile << times_getrf_batched[i] << std::endl;
                }
            }
            else{
                std::cout << "Unable to open file" << std::endl;
            }
            outputFile.close();

            filename = time_path + "times_getrf_for_" + std::to_string(blocksize) + "_"  + std::to_string(batch_size) + ".txt";
            outputFile.open(filename);
            if(outputFile.is_open()){
                for(int i = 0; i < number_of_measurements; i++){
                    outputFile << times_getrf_for[i] << std::endl;
                }
            }
            else{
                std::cout << "Unable to open file" << std::endl;
            }
            outputFile.close();

            filename = time_path + "times_getri_batched_" + std::to_string(blocksize) + "_"  + std::to_string(batch_size) + ".txt"; 
            outputFile.open(filename);
            if(outputFile.is_open()){
                for(int i = 0; i < number_of_measurements; i++){
                    outputFile << times_getri_batched[i] << std::endl;
                }
            }
            else{
                std::cout << "Unable to open file" << std::endl;
            }
            outputFile.close();

            filename = time_path + "times_getrs_for_" + std::to_string(blocksize) + "_"  + std::to_string(batch_size) + ".txt";
            outputFile.open(filename);
            if(outputFile.is_open()){
                for(int i = 0; i < number_of_measurements; i++){
                    outputFile << times_getrs_for[i] << std::endl;
                }
            }
            else{
                std::cout << "Unable to open file" << std::endl;
            }
            outputFile.close();

            filename = time_path + "times_solve_batched_" + std::to_string(blocksize) + "_"  + std::to_string(batch_size) + ".txt";
            outputFile.open(filename);
            if(outputFile.is_open()){ 
                for(int i = 0; i < number_of_measurements; i++){
                    outputFile << times_solve_batched[i] << std::endl;
                }
            }
            else{
                std::cout << "Unable to open file" << std::endl; 
            }
            outputFile.close();

            filename = time_path + "times_getrs_batched_" + std::to_string(blocksize) + "_"  + std::to_string(batch_size) + ".txt";
            outputFile.open(filename);
            if(outputFile.is_open()){ 
                for(int i = 0; i < number_of_measurements; i++){
                    outputFile << times_getrs_batched[i] << std::endl;
                }
            }
            else{
                std::cout << "Unable to open file" << std::endl; 
            }
            outputFile.close();


            cublasErrchk(cublasDestroy(cublas_handle));
            cusolverErrchk(cusolverDnDestroy(cusolver_handle));
            cudaErrchk(cudaStreamDestroy(stream));

            cudaErrchk(cudaFreeHost(mm_A_h));
            cudaErrchk(cudaFreeHost(mm_B_h));
            cudaErrchk(cudaFreeHost(mm_C_h));
            cudaErrchk(cudaFreeHost(mm_LU_in_h));
            cudaErrchk(cudaFreeHost(mm_LU_out_h));

            cudaErrchk(cudaFree(mm_A_d));
            cudaErrchk(cudaFree(mm_B_d));
            cudaErrchk(cudaFree(mm_C_d));
            cudaErrchk(cudaFree(mm_LU_in_d));
            cudaErrchk(cudaFree(mm_LU_out_d));

            cudaErrchk(cudaFree(mm_A_ptr_d));
            cudaErrchk(cudaFree(mm_B_ptr_d));
            cudaErrchk(cudaFree(mm_C_ptr_d));
            cudaErrchk(cudaFree(mm_LU_in_ptr_d));
            cudaErrchk(cudaFree(mm_LU_out_ptr_d));

            cudaErrchk(cudaFree(ipiv_d));
            cudaErrchk(cudaFree(info_d));

            cudaErrchk(cudaFreeHost(identity_h));
            cudaErrchk(cudaFree(identity_d));
            cudaErrchk(cudaFree(buffer));
        }
    }
    return 0;
}








