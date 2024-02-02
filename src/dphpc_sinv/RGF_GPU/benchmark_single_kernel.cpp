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
    // int bs_test = 6;
    // int nbatch_test = 10;
    // int batch_sizes_input[nbatch_test] = {1, 2, 4, 8, 16, 32, 48, 64, 96, 128};
    // int blocksize_input[bs_test] = {64, 128, 256, 512, 768, 1024};

    int bs_test = 1;
    int nbatch_test = 1;
    int batch_sizes_input[nbatch_test] = {1};
    int blocksize_input[bs_test] = {1024};

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


            // std::cout << "Batched MM Measurement " << std::endl;
            // cublasErrchk(cublasZgemmBatched(
            //     cublas_handle,
            //     CUBLAS_OP_N, CUBLAS_OP_N,   
            //     blocksize, blocksize, blocksize,
            //     &alpha,
            //     mm_A_ptr_d, blocksize,
            //     mm_B_ptr_d, blocksize,
            //     &beta,
            //     mm_C_ptr_d, blocksize, batch_size));



            // std::cout << "Strided Batched MM Measurement " <<  std::endl;
            // cublasErrchk(cublasZgemmStridedBatched(
            //     cublas_handle,
            //     CUBLAS_OP_N, CUBLAS_OP_N,   
            //     blocksize, blocksize, blocksize,
            //     &alpha,
            //     mm_A_d, blocksize,
            //     blocksize*blocksize,
            //     mm_B_d, blocksize,
            //     blocksize*blocksize,
            //     &beta,
            //     mm_C_d, blocksize,
            //     blocksize*blocksize,
            //     batch_size));



            // std::cout << "For MM Measurement " << std::endl;
            // for(unsigned int batch = 0; batch < 1; batch++){
            //     cublasErrchk(cublasZgemm(
            //         cublas_handle,
            //         CUBLAS_OP_N, CUBLAS_OP_N,   
            //         blocksize, blocksize, blocksize,
            //         &alpha,
            //         mm_A_ptr_h[batch], blocksize,
            //         mm_B_ptr_h[batch], blocksize,
            //         &beta,
            //         mm_C_ptr_h[batch], blocksize));
            // }


            std::cout << "Batched LU Measurement " <<  std::endl;
            cudaErrchk(cudaMemcpy(mm_LU_in_d, mm_LU_in_h, blocksize*blocksize*batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));
            cudaErrchk(cudaMemcpy(mm_LU_out_d, mm_LU_out_h, blocksize*blocksize*batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));
            cublasErrchk(cublasZgetrfBatched(
                    cublas_handle,
                    blocksize,
                    mm_LU_in_ptr_d,
                    blocksize,
                    ipiv_d,
                    info_d,
                    batch_size));
            // inversion
            // cublasErrchk(cublasZgetriBatched(
            //         cublas_handle,
            //         blocksize,
            //         mm_LU_in_ptr_d,
            //         blocksize,
            //         ipiv_d,
            //         mm_LU_out_ptr_d,
            //         blocksize,
            //         info_d,
            //         batch_size));




            // std::cout << "Batched getrf Measurement " << std::endl;
            // cudaErrchk(cudaMemcpy(mm_LU_in_d, mm_LU_in_h, blocksize*blocksize*batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));
            // cublasErrchk(cublasZgetrfBatched(
            //         cublas_handle,
            //         blocksize,
            //         mm_LU_in_ptr_d,
            //         blocksize,
            //         ipiv_d,
            //         info_d,
            //         batch_size));

            // std::cout << "Batched getri Measurement " <<  std::endl;
            // cudaErrchk(cudaMemcpy(mm_LU_out_d, mm_LU_out_h, blocksize*blocksize*batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));
            // // inversion
            // cublasErrchk(cublasZgetriBatched(
            //         cublas_handle,
            //         blocksize,
            //         mm_LU_in_ptr_d,
            //         blocksize,
            //         ipiv_d,
            //         mm_LU_out_ptr_d,
            //         blocksize,
            //         info_d,
            //         batch_size));




            // cudaErrchk(cudaMemcpy(mm_LU_in_d, mm_LU_in_h, blocksize*blocksize*batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));
            // for(unsigned int batch = 0; batch < batch_size; batch++){
            //     cudaErrchk(cudaMemcpy(mm_LU_out_d + batch*blocksize*blocksize, identity_h, blocksize*blocksize*sizeof(complex_d), cudaMemcpyHostToDevice));
            // }
            // std::cout << "For LU Measurement " << std::endl;
            // for(unsigned int batch = 0; batch < 1; batch++){
            //     cusolverErrchk(cusolverDnZgetrf(cusolver_handle,
            //         blocksize, blocksize,
            //         mm_LU_in_ptr_h[batch],
            //         blocksize, buffer,
            //         ipiv_d, info_d));
                
            //     //back substitution
            //     cusolverErrchk(cusolverDnZgetrs(cusolver_handle,
            //         CUBLAS_OP_N,
            //         blocksize, blocksize,
            //         mm_LU_in_ptr_h[batch],
            //         blocksize, ipiv_d,
            //         mm_LU_out_ptr_h[batch],
            //         blocksize, info_d));

            // }



            // cudaErrchk(cudaMemcpy(mm_LU_in_d, mm_LU_in_h, blocksize*blocksize*batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));
            // std::cout << "For cusolverDnZgetrf " << std::endl;
            // for(unsigned int batch = 0; batch < batch_size; batch++){
            //     cusolverErrchk(cusolverDnZgetrf(cusolver_handle,
            //         blocksize, blocksize,
            //         mm_LU_in_ptr_h[batch],
            //         blocksize, buffer,
            //         ipiv_d, info_d));
            // }



            // for(unsigned int batch = 0; batch < batch_size; batch++){
            //     cudaErrchk(cudaMemcpy(mm_LU_out_d + batch*blocksize*blocksize, identity_d, batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));
            // }
            // std::cout << "For cusolverDnZgetrs Measurement " <<  std::endl;
            // for(unsigned int batch = 0; batch < batch_size; batch++){
            //     //back substitution
            //     cusolverErrchk(cusolverDnZgetrs(cusolver_handle,
            //         CUBLAS_OP_N,
            //         blocksize, blocksize,
            //         mm_LU_in_ptr_h[batch],
            //         blocksize, ipiv_d,
            //         mm_LU_out_ptr_h[batch],
            //         blocksize, info_d));

            // }


            // std::cout << "Batched solve Measurement " << std::endl;
            // cudaErrchk(cudaMemcpy(mm_LU_in_d, mm_LU_in_h, blocksize*blocksize*batch_size*sizeof(complex_d), cudaMemcpyHostToDevice));
            // for(unsigned int batch = 0; batch < batch_size; batch++){
            //     cudaErrchk(cudaMemcpy(mm_LU_out_d + batch*blocksize*blocksize, identity_h, blocksize*blocksize*sizeof(complex_d), cudaMemcpyHostToDevice));
            // }
            // int info_h;
            // cublasErrchk(cublasZgetrfBatched(
            //         cublas_handle,
            //         blocksize,
            //         mm_LU_in_ptr_d,
            //         blocksize,
            //         ipiv_d,
            //         info_d,
            //         batch_size));
            // // inversion
            // cublasErrchk(cublasZgetrsBatched(
            //         cublas_handle,
            //         CUBLAS_OP_N,
            //         blocksize,
            //         blocksize,
            //         mm_LU_in_ptr_d,
            //         blocksize,
            //         ipiv_d,
            //         mm_LU_out_ptr_d,
            //         blocksize,
            //         &info_h,
            //         batch_size));


            // std::cout << "Batched getrs Measurement " << std::endl;
            // for(unsigned int batch = 0; batch < batch_size; batch++){
            //     cudaErrchk(cudaMemcpy(mm_LU_out_d + batch*blocksize*blocksize, identity_h, blocksize*blocksize*sizeof(complex_d), cudaMemcpyHostToDevice));
            // }
            // int info_h;
            // cublasErrchk(cublasZgetrsBatched(
            //         cublas_handle,
            //         CUBLAS_OP_N,
            //         blocksize,
            //         blocksize,
            //         mm_LU_in_ptr_d,
            //         blocksize,
            //         ipiv_d,
            //         mm_LU_out_ptr_d,
            //         blocksize,
            //         &info_h,
            //         batch_size));


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








