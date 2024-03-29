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
    complex_h *self_energy_matrices_lesser_diagblk_h[batch_size];
    complex_h *self_energy_matrices_lesser_upperblk_h[batch_size];
    complex_h *self_energy_matrices_greater_diagblk_h[batch_size];
    complex_h *self_energy_matrices_greater_upperblk_h[batch_size];

    complex_h *lesser_inv_matrices_diagblk_ref[batch_size];
    complex_h *lesser_inv_matrices_upperblk_ref[batch_size];
    complex_h *greater_inv_matrices_diagblk_ref[batch_size];
    complex_h *greater_inv_matrices_upperblk_ref[batch_size];
    complex_h *retarded_inv_matrices_diagblk_ref[batch_size];
    complex_h *retarded_inv_matrices_upperblk_ref[batch_size];
    complex_h *retarded_inv_matrices_lowerblk_ref[batch_size];

    bool not_failed = true;
    double reltol = 1e-7;
    double abstol = 1e-12;
    int n_tests = n_tests;
    bool same_array = true;

    // has to be tested multiple times to capture if synchronization is faulty
    for(int k = 0; k < 100; k++){
        for(unsigned int batch = 0; batch < batch_size; batch++){
            std::cout << "Batch: " << batch << std::endl;
            
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

            // load the self energy
            complex_h* self_energy_lesser_diagblk = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
            std::string self_energy_lesser_diagblk_path = base_path + "self_energy_lesser_"+ std::to_string(batch) +"_diagblk_"
            + std::to_string(matrix_size) + "_"+ std::to_string(blocksize) + "_" + std::to_string(batch_size) +
            ".bin";
            load_binary_matrix(self_energy_lesser_diagblk_path.c_str(), self_energy_lesser_diagblk, blocksize, matrix_size);

            complex_h* self_energy_lesser_upperblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
            std::string self_energy_lesser_upperblk_path = base_path + "self_energy_lesser_"+ std::to_string(batch) +"_upperblk_"
            + std::to_string(matrix_size) + "_"+ std::to_string(blocksize) + "_" + std::to_string(batch_size) +
            ".bin";
            load_binary_matrix(self_energy_lesser_upperblk_path.c_str(), self_energy_lesser_upperblk, blocksize, off_diag_size);



            complex_h* self_energy_greater_diagblk = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
            std::string self_energy_greater_diagblk_path = base_path + "self_energy_greater_"+ std::to_string(batch) +"_diagblk_"
            + std::to_string(matrix_size) + "_"+ std::to_string(blocksize) + "_" + std::to_string(batch_size) +
            ".bin";
            load_binary_matrix(self_energy_greater_diagblk_path.c_str(), self_energy_greater_diagblk, blocksize, matrix_size);

            complex_h* self_energy_greater_upperblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
            std::string self_energy_greater_upperblk_path = base_path + "self_energy_greater_"+ std::to_string(batch) +"_upperblk_"
            + std::to_string(matrix_size) + "_"+ std::to_string(blocksize) + "_" + std::to_string(batch_size) +
            ".bin";
            load_binary_matrix(self_energy_greater_upperblk_path.c_str(), self_energy_greater_upperblk, blocksize, off_diag_size);



            complex_h* lesser_inv_diagblk = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
            std::string lesser_diagblk_path = base_path + "lesser_"+ std::to_string(batch) +"_inv_diagblk_"
            + std::to_string(matrix_size) + "_"+ std::to_string(blocksize) + "_" + std::to_string(batch_size) +
            ".bin";
            load_binary_matrix(lesser_diagblk_path.c_str(), lesser_inv_diagblk, blocksize, matrix_size);

            complex_h* lesser_inv_upperblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
            std::string lesser_upperblk_path = base_path + "lesser_"+ std::to_string(batch) +"_inv_upperblk_"
            + std::to_string(matrix_size) + "_"+ std::to_string(blocksize) + "_" + std::to_string(batch_size) +
            ".bin";
            load_binary_matrix(lesser_upperblk_path.c_str(), lesser_inv_upperblk, blocksize, off_diag_size);



            complex_h* greater_inv_diagblk = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
            std::string greater_diagblk_path = base_path + "greater_"+ std::to_string(batch) +"_inv_diagblk_"
            + std::to_string(matrix_size) + "_"+ std::to_string(blocksize) + "_" + std::to_string(batch_size) +
            ".bin";
            load_binary_matrix(greater_diagblk_path.c_str(), greater_inv_diagblk, blocksize, matrix_size);

            complex_h* greater_inv_upperblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
            std::string greater_upperblk_path = base_path + "greater_"+ std::to_string(batch) +"_inv_upperblk_"
            + std::to_string(matrix_size) + "_"+ std::to_string(blocksize) + "_" + std::to_string(batch_size) +
            ".bin";
            load_binary_matrix(greater_upperblk_path.c_str(), greater_inv_upperblk, blocksize, off_diag_size);


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
            complex_h* self_energy_lesser_diagblk_h = NULL;
            complex_h* self_energy_lesser_upperblk_h = NULL;
            complex_h* self_energy_greater_diagblk_h = NULL;
            complex_h* self_energy_greater_upperblk_h = NULL;
            complex_h* lesser_inv_diagblk_ref = NULL;
            complex_h* lesser_inv_upperblk_ref = NULL;
            complex_h* greater_inv_diagblk_ref = NULL;
            complex_h* greater_inv_upperblk_ref = NULL;
            complex_h* retarded_inv_diagblk_ref = NULL;
            complex_h* retarded_inv_upperblk_ref = NULL;
            complex_h* retarded_inv_lowerblk_ref = NULL;
            cudaMallocHost((void**)&system_matrix_diagblk_h, blocksize * matrix_size * sizeof(complex_h));
            cudaMallocHost((void**)&system_matrix_upperblk_h, blocksize * off_diag_size * sizeof(complex_h));
            cudaMallocHost((void**)&system_matrix_lowerblk_h, blocksize * off_diag_size * sizeof(complex_h));
            cudaMallocHost((void**)&self_energy_lesser_diagblk_h, blocksize * matrix_size * sizeof(complex_h));
            cudaMallocHost((void**)&self_energy_lesser_upperblk_h, blocksize * off_diag_size * sizeof(complex_h));
            cudaMallocHost((void**)&self_energy_greater_diagblk_h, blocksize * matrix_size * sizeof(complex_h));
            cudaMallocHost((void**)&self_energy_greater_upperblk_h, blocksize * off_diag_size * sizeof(complex_h));
            cudaMallocHost((void**)&lesser_inv_diagblk_ref, blocksize * matrix_size * sizeof(complex_h));
            cudaMallocHost((void**)&lesser_inv_upperblk_ref, blocksize * off_diag_size * sizeof(complex_h));
            cudaMallocHost((void**)&greater_inv_diagblk_ref, blocksize * matrix_size * sizeof(complex_h));
            cudaMallocHost((void**)&greater_inv_upperblk_ref, blocksize * off_diag_size * sizeof(complex_h));
            cudaMallocHost((void**)&retarded_inv_diagblk_ref, blocksize * matrix_size * sizeof(complex_h));
            cudaMallocHost((void**)&retarded_inv_upperblk_ref, blocksize * off_diag_size * sizeof(complex_h));
            cudaMallocHost((void**)&retarded_inv_lowerblk_ref, blocksize * off_diag_size * sizeof(complex_h));

            transform_diagblk<complex_h>(system_matrix_diagblk, system_matrix_diagblk_h, blocksize, matrix_size);
            transform_diagblk<complex_h>(self_energy_lesser_diagblk, self_energy_lesser_diagblk_h, blocksize, matrix_size);
            transform_diagblk<complex_h>(self_energy_greater_diagblk, self_energy_greater_diagblk_h, blocksize, matrix_size);
            transform_diagblk<complex_h>(lesser_inv_diagblk, lesser_inv_diagblk_ref, blocksize, matrix_size);
            transform_diagblk<complex_h>(greater_inv_diagblk, greater_inv_diagblk_ref, blocksize, matrix_size);
            transform_diagblk<complex_h>(retarded_inv_diagblk, retarded_inv_diagblk_ref, blocksize, matrix_size);
            
            transform_offblk<complex_h>(system_matrix_upperblk, system_matrix_upperblk_h, blocksize, off_diag_size);
            transform_offblk<complex_h>(system_matrix_lowerblk, system_matrix_lowerblk_h, blocksize, off_diag_size);
            transform_offblk<complex_h>(self_energy_lesser_upperblk, self_energy_lesser_upperblk_h, blocksize, off_diag_size);
            transform_offblk<complex_h>(self_energy_greater_upperblk, self_energy_greater_upperblk_h, blocksize, off_diag_size);
            transform_offblk<complex_h>(lesser_inv_upperblk, lesser_inv_upperblk_ref, blocksize, off_diag_size);
            transform_offblk<complex_h>(greater_inv_upperblk, greater_inv_upperblk_ref, blocksize, off_diag_size);
            transform_offblk<complex_h>(retarded_inv_upperblk, retarded_inv_upperblk_ref, blocksize, off_diag_size);
            transform_offblk<complex_h>(retarded_inv_lowerblk, retarded_inv_lowerblk_ref, blocksize, off_diag_size);

            system_matrices_diagblk_h[batch] = system_matrix_diagblk_h;
            system_matrices_upperblk_h[batch] = system_matrix_upperblk_h;
            system_matrices_lowerblk_h[batch] = system_matrix_lowerblk_h;
            self_energy_matrices_lesser_diagblk_h[batch] = self_energy_lesser_diagblk_h;
            self_energy_matrices_lesser_upperblk_h[batch] = self_energy_lesser_upperblk_h;
            self_energy_matrices_greater_diagblk_h[batch] = self_energy_greater_diagblk_h;
            self_energy_matrices_greater_upperblk_h[batch] = self_energy_greater_upperblk_h;
            
            lesser_inv_matrices_diagblk_ref[batch] = lesser_inv_diagblk_ref;
            lesser_inv_matrices_upperblk_ref[batch] = lesser_inv_upperblk_ref;
            greater_inv_matrices_diagblk_ref[batch] = greater_inv_diagblk_ref;
            greater_inv_matrices_upperblk_ref[batch] = greater_inv_upperblk_ref;
            retarded_inv_matrices_diagblk_ref[batch] = retarded_inv_diagblk_ref;
            retarded_inv_matrices_upperblk_ref[batch] = retarded_inv_upperblk_ref;
            retarded_inv_matrices_lowerblk_ref[batch] = retarded_inv_lowerblk_ref;


            // allocate memory for the inv
            complex_h* lesser_inv_diagblk_h = NULL;
            complex_h* lesser_inv_upperblk_h = NULL;
            complex_h* greater_inv_diagblk_h = NULL;
            complex_h* greater_inv_upperblk_h = NULL;
            complex_h* retarded_inv_diagblk_h = NULL;
            complex_h* retarded_inv_upperblk_h = NULL;
            complex_h* retarded_inv_lowerblk_h = NULL;

            cudaMallocHost((void**)&lesser_inv_diagblk_h, blocksize * matrix_size * sizeof(complex_h));
            cudaMallocHost((void**)&lesser_inv_upperblk_h, blocksize * off_diag_size * sizeof(complex_h));
            cudaMallocHost((void**)&greater_inv_diagblk_h, blocksize * matrix_size * sizeof(complex_h));
            cudaMallocHost((void**)&greater_inv_upperblk_h, blocksize * off_diag_size * sizeof(complex_h));
            cudaMallocHost((void**)&retarded_inv_diagblk_h, blocksize * matrix_size * sizeof(complex_h));
            cudaMallocHost((void**)&retarded_inv_upperblk_h, blocksize * off_diag_size * sizeof(complex_h));
            cudaMallocHost((void**)&retarded_inv_lowerblk_h, blocksize * off_diag_size * sizeof(complex_h));

            rgf_lesser_greater_retarded(blocksize, matrix_size,
                                    system_matrix_diagblk_h,
                                    system_matrix_upperblk_h,
                                    system_matrix_lowerblk_h,
                                    self_energy_lesser_diagblk_h,
                                    self_energy_lesser_upperblk_h,
                                    self_energy_greater_diagblk_h,
                                    self_energy_greater_upperblk_h,
                                    lesser_inv_diagblk_h,
                                    lesser_inv_upperblk_h,
                                    greater_inv_diagblk_h,
                                    greater_inv_upperblk_h,
                                    retarded_inv_diagblk_h,
                                    retarded_inv_upperblk_h,
                                    retarded_inv_lowerblk_h);


            same_array =  true;
            same_array = same_array && assert_array_magnitude<complex_h>(lesser_inv_diagblk_h,
                lesser_inv_diagblk_ref, abstol, reltol, blocksize * matrix_size);
            same_array = same_array && assert_array_magnitude<complex_h>(lesser_inv_upperblk_h,
                lesser_inv_upperblk_ref, abstol, reltol, blocksize * off_diag_size);
            same_array = same_array && assert_array_magnitude<complex_h>(greater_inv_diagblk_h,
                greater_inv_diagblk_ref, abstol, reltol, blocksize * matrix_size);
            same_array = same_array && assert_array_magnitude<complex_h>(greater_inv_upperblk_h,
                greater_inv_upperblk_ref, abstol, reltol, blocksize * off_diag_size);
            same_array = same_array && assert_array_magnitude<complex_h>(retarded_inv_diagblk_h,
                retarded_inv_diagblk_ref, abstol, reltol, blocksize * matrix_size);
            same_array = same_array && assert_array_magnitude<complex_h>(retarded_inv_upperblk_h,
                retarded_inv_upperblk_ref, abstol, reltol, blocksize * off_diag_size);
            same_array = same_array && assert_array_magnitude<complex_h>(retarded_inv_lowerblk_h,
                retarded_inv_lowerblk_ref, abstol, reltol, blocksize * off_diag_size);

            if(!same_array){
                printf("FAILED lesser greater retarded\n");
                not_failed = false;
            }

            // set outputs to zero
            for(unsigned int i = 0; i < blocksize * matrix_size; i++){
                lesser_inv_diagblk_h[i] = 0.0;
                greater_inv_diagblk_h[i] = 0.0;
                retarded_inv_diagblk_h[i] = 0.0;
            }
            for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
                lesser_inv_upperblk_h[i] = 0.0;
                greater_inv_upperblk_h[i] = 0.0;
                retarded_inv_upperblk_h[i] = 0.0;
                retarded_inv_lowerblk_h[i] = 0.0;
            }

            rgf_lesser_greater(
                blocksize, matrix_size,
                system_matrix_diagblk_h,
                system_matrix_upperblk_h,
                system_matrix_lowerblk_h,
                self_energy_lesser_diagblk_h,
                self_energy_lesser_upperblk_h,
                self_energy_greater_diagblk_h,
                self_energy_greater_upperblk_h,
                lesser_inv_diagblk_h,
                lesser_inv_upperblk_h,
                greater_inv_diagblk_h,
                greater_inv_upperblk_h);

            same_array =  true;
            same_array = same_array && assert_array_magnitude<complex_h>(lesser_inv_diagblk_h,
                lesser_inv_diagblk_ref, abstol, reltol, blocksize * matrix_size);
            same_array = same_array && assert_array_magnitude<complex_h>(lesser_inv_upperblk_h,
                lesser_inv_upperblk_ref, abstol, reltol, blocksize * off_diag_size);
            same_array = same_array && assert_array_magnitude<complex_h>(greater_inv_diagblk_h,
                greater_inv_diagblk_ref, abstol, reltol, blocksize * matrix_size);
            same_array = same_array && assert_array_magnitude<complex_h>(greater_inv_upperblk_h,
                greater_inv_upperblk_ref, abstol, reltol, blocksize * off_diag_size);

            if(!same_array){
                printf("FAILED lesser greater\n");
                not_failed = false;
            }

            // set outputs to zero
            for(unsigned int i = 0; i < blocksize * matrix_size; i++){
                lesser_inv_diagblk_h[i] = 0.0;
                greater_inv_diagblk_h[i] = 0.0;
            }
            for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
                lesser_inv_upperblk_h[i] = 0.0;
                greater_inv_upperblk_h[i] = 0.0;
            }
            rgf_retarded_fits_gpu_memory(
                blocksize, matrix_size,
                system_matrix_diagblk_h,
                system_matrix_upperblk_h,
                system_matrix_lowerblk_h,
                retarded_inv_diagblk_h,
                retarded_inv_upperblk_h,
                retarded_inv_lowerblk_h);

            same_array =  true;
            same_array = same_array && assert_array_magnitude<complex_h>(retarded_inv_diagblk_h,
                retarded_inv_diagblk_ref, abstol, reltol, blocksize * matrix_size);
            same_array = same_array && assert_array_magnitude<complex_h>(retarded_inv_upperblk_h,
                retarded_inv_upperblk_ref, abstol, reltol, blocksize * off_diag_size);
            same_array = same_array && assert_array_magnitude<complex_h>(retarded_inv_lowerblk_h,
                retarded_inv_lowerblk_ref, abstol, reltol, blocksize * off_diag_size);
            
            if(!same_array){
                printf("FAILED retarded rgf_retarded_fits_gpu_memory\n");
                not_failed = false;
            }

            // set outputs to zero
            for(unsigned int i = 0; i < blocksize * matrix_size; i++){
                lesser_inv_diagblk_h[i] = 0.0;
                greater_inv_diagblk_h[i] = 0.0;
            }
            for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
                lesser_inv_upperblk_h[i] = 0.0;
                greater_inv_upperblk_h[i] = 0.0;
            }
            rgf_retarded_fits_gpu_memory_with_copy_compute_overlap(
                blocksize, matrix_size,
                system_matrix_diagblk_h,
                system_matrix_upperblk_h,
                system_matrix_lowerblk_h,
                retarded_inv_diagblk_h,
                retarded_inv_upperblk_h,
                retarded_inv_lowerblk_h);

            same_array =  true;
            same_array = same_array && assert_array_magnitude<complex_h>(retarded_inv_diagblk_h,
                retarded_inv_diagblk_ref, abstol, reltol, blocksize * matrix_size);
            same_array = same_array && assert_array_magnitude<complex_h>(retarded_inv_upperblk_h,
                retarded_inv_upperblk_ref, abstol, reltol, blocksize * off_diag_size);
            same_array = same_array && assert_array_magnitude<complex_h>(retarded_inv_lowerblk_h,
                retarded_inv_lowerblk_ref, abstol, reltol, blocksize * off_diag_size);
            
            if(!same_array){
                printf("FAILED retarded rgf_retarded_fits_gpu_memory_with_copy_compute_overlap\n");
                not_failed = false;
            }

            // set outputs to zero
            for(unsigned int i = 0; i < blocksize * matrix_size; i++){
                lesser_inv_diagblk_h[i] = 0.0;
                greater_inv_diagblk_h[i] = 0.0;
            }
            for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
                lesser_inv_upperblk_h[i] = 0.0;
                greater_inv_upperblk_h[i] = 0.0;
            }
            rgf_retarded_does_not_fit_gpu_memory(
                blocksize, matrix_size,
                system_matrix_diagblk_h,
                system_matrix_upperblk_h,
                system_matrix_lowerblk_h,
                retarded_inv_diagblk_h,
                retarded_inv_upperblk_h,
                retarded_inv_lowerblk_h);

            same_array =  true;
            same_array = same_array && assert_array_magnitude<complex_h>(retarded_inv_diagblk_h,
                retarded_inv_diagblk_ref, abstol, reltol, blocksize * matrix_size);
            same_array = same_array && assert_array_magnitude<complex_h>(retarded_inv_upperblk_h,
                retarded_inv_upperblk_ref, abstol, reltol, blocksize * off_diag_size);
            same_array = same_array && assert_array_magnitude<complex_h>(retarded_inv_lowerblk_h,
                retarded_inv_lowerblk_ref, abstol, reltol, blocksize * off_diag_size);
            
            if(!same_array){
                printf("FAILED retarded rgf_retarded_does_not_fit_gpu_memory\n");
                not_failed = false;
            }

            // set outputs to zero
            for(unsigned int i = 0; i < blocksize * matrix_size; i++){
                lesser_inv_diagblk_h[i] = 0.0;
                greater_inv_diagblk_h[i] = 0.0;
            }
            for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
                lesser_inv_upperblk_h[i] = 0.0;
                greater_inv_upperblk_h[i] = 0.0;
            }
            rgf_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap(
                blocksize, matrix_size,
                system_matrix_diagblk_h,
                system_matrix_upperblk_h,
                system_matrix_lowerblk_h,
                retarded_inv_diagblk_h,
                retarded_inv_upperblk_h,
                retarded_inv_lowerblk_h);

            same_array =  true;
            same_array = same_array && assert_array_magnitude<complex_h>(retarded_inv_diagblk_h,
                retarded_inv_diagblk_ref, abstol, reltol, blocksize * matrix_size);
            same_array = same_array && assert_array_magnitude<complex_h>(retarded_inv_upperblk_h,
                retarded_inv_upperblk_ref, abstol, reltol, blocksize * off_diag_size);
            same_array = same_array && assert_array_magnitude<complex_h>(retarded_inv_lowerblk_h,
                retarded_inv_lowerblk_ref, abstol, reltol, blocksize * off_diag_size);
            
            if(!same_array){
                printf("FAILED retarded rgf_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap\n");
                not_failed = false;
            }

            cudaFreeHost(lesser_inv_diagblk_h);
            cudaFreeHost(lesser_inv_upperblk_h);
            cudaFreeHost(greater_inv_diagblk_h);
            cudaFreeHost(greater_inv_upperblk_h);
            cudaFreeHost(retarded_inv_diagblk_h);
            cudaFreeHost(retarded_inv_upperblk_h);
            cudaFreeHost(retarded_inv_lowerblk_h);

            // free non contiguous memory
            free(system_matrix_diagblk);
            free(system_matrix_upperblk);
            free(system_matrix_lowerblk);
            free(self_energy_lesser_diagblk);
            free(self_energy_lesser_upperblk);
            free(self_energy_greater_diagblk);
            free(self_energy_greater_upperblk);
            free(lesser_inv_diagblk);
            free(lesser_inv_upperblk);
            free(greater_inv_diagblk);
            free(greater_inv_upperblk);
            free(retarded_inv_diagblk);
            free(retarded_inv_upperblk);
            free(retarded_inv_lowerblk);
        }

        // transform to batched blocks
        complex_h* batch_system_matrices_diagblk_h[n_blocks];
        complex_h* batch_system_matrices_upperblk_h[n_blocks-1];
        complex_h* batch_system_matrices_lowerblk_h[n_blocks-1];
        complex_h* batch_self_energy_matrices_lesser_diagblk_h[n_blocks];
        complex_h* batch_self_energy_matrices_lesser_upperblk_h[n_blocks-1];
        complex_h* batch_self_energy_matrices_greater_diagblk_h[n_blocks];
        complex_h* batch_self_energy_matrices_greater_upperblk_h[n_blocks-1];
        complex_h* batch_lesser_inv_matrices_diagblk_ref[n_blocks];
        complex_h* batch_lesser_inv_matrices_upperblk_ref[n_blocks-1];
        complex_h* batch_greater_inv_matrices_diagblk_ref[n_blocks];
        complex_h* batch_greater_inv_matrices_upperblk_ref[n_blocks-1];
        complex_h* batch_lesser_inv_matrices_diagblk_h[n_blocks];
        complex_h* batch_lesser_inv_matrices_upperblk_h[n_blocks-1];
        complex_h* batch_greater_inv_matrices_diagblk_h[n_blocks];
        complex_h* batch_greater_inv_matrices_upperblk_h[n_blocks-1];
        complex_h* batch_retarded_inv_matrices_diagblk_ref[n_blocks];
        complex_h* batch_retarded_inv_matrices_upperblk_ref[n_blocks-1];
        complex_h* batch_retarded_inv_matrices_lowerblk_ref[n_blocks-1];
        complex_h* batch_retarded_inv_matrices_diagblk_h[n_blocks];
        complex_h* batch_retarded_inv_matrices_upperblk_h[n_blocks-1];
        complex_h* batch_retarded_inv_matrices_lowerblk_h[n_blocks-1];

        for(unsigned int i = 0; i < n_blocks; i++){
            cudaErrchk(cudaMallocHost((void**)&batch_system_matrices_diagblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_self_energy_matrices_lesser_diagblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_self_energy_matrices_greater_diagblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_lesser_inv_matrices_diagblk_ref[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_greater_inv_matrices_diagblk_ref[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_lesser_inv_matrices_diagblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_greater_inv_matrices_diagblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_retarded_inv_matrices_diagblk_ref[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_retarded_inv_matrices_diagblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            for(unsigned int batch = 0; batch < batch_size; batch++){
                for(unsigned int j = 0; j < blocksize * blocksize; j++){
                    batch_system_matrices_diagblk_h[i][batch * blocksize * blocksize + j] = system_matrices_diagblk_h[batch][i * blocksize * blocksize + j];
                    batch_self_energy_matrices_lesser_diagblk_h[i][batch * blocksize * blocksize + j] = self_energy_matrices_lesser_diagblk_h[batch][i * blocksize * blocksize + j];
                    batch_self_energy_matrices_greater_diagblk_h[i][batch * blocksize * blocksize + j] = self_energy_matrices_greater_diagblk_h[batch][i * blocksize * blocksize + j];
                    batch_lesser_inv_matrices_diagblk_ref[i][batch * blocksize * blocksize + j] = lesser_inv_matrices_diagblk_ref[batch][i * blocksize * blocksize + j];
                    batch_greater_inv_matrices_diagblk_ref[i][batch * blocksize * blocksize + j] = greater_inv_matrices_diagblk_ref[batch][i * blocksize * blocksize + j];
                    batch_retarded_inv_matrices_diagblk_ref[i][batch * blocksize * blocksize + j] = retarded_inv_matrices_diagblk_ref[batch][i * blocksize * blocksize + j];
                }
            }
        }
        for(unsigned int i = 0; i < n_blocks-1; i++){
            cudaErrchk(cudaMallocHost((void**)&batch_system_matrices_upperblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_system_matrices_lowerblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_self_energy_matrices_lesser_upperblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_self_energy_matrices_greater_upperblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_lesser_inv_matrices_upperblk_ref[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_greater_inv_matrices_upperblk_ref[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_lesser_inv_matrices_upperblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_greater_inv_matrices_upperblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_retarded_inv_matrices_upperblk_ref[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_retarded_inv_matrices_lowerblk_ref[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_retarded_inv_matrices_upperblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            cudaErrchk(cudaMallocHost((void**)&batch_retarded_inv_matrices_lowerblk_h[i], batch_size * blocksize * blocksize * sizeof(complex_h)));
            for(unsigned int batch = 0; batch < batch_size; batch++){
                for(unsigned int j = 0; j < blocksize * blocksize; j++){
                    batch_system_matrices_upperblk_h[i][batch * blocksize * blocksize + j] = system_matrices_upperblk_h[batch][i * blocksize * blocksize + j];
                    batch_system_matrices_lowerblk_h[i][batch * blocksize * blocksize + j] = system_matrices_lowerblk_h[batch][i * blocksize * blocksize + j];
                    batch_self_energy_matrices_lesser_upperblk_h[i][batch * blocksize * blocksize + j] = self_energy_matrices_lesser_upperblk_h[batch][i * blocksize * blocksize + j];
                    batch_self_energy_matrices_greater_upperblk_h[i][batch * blocksize * blocksize + j] = self_energy_matrices_greater_upperblk_h[batch][i * blocksize * blocksize + j];
                    batch_lesser_inv_matrices_upperblk_ref[i][batch * blocksize * blocksize + j] = lesser_inv_matrices_upperblk_ref[batch][i * blocksize * blocksize + j];
                    batch_greater_inv_matrices_upperblk_ref[i][batch * blocksize * blocksize + j] = greater_inv_matrices_upperblk_ref[batch][i * blocksize * blocksize + j];
                    batch_retarded_inv_matrices_upperblk_ref[i][batch * blocksize * blocksize + j] = retarded_inv_matrices_upperblk_ref[batch][i * blocksize * blocksize + j];
                    batch_retarded_inv_matrices_lowerblk_ref[i][batch * blocksize * blocksize + j] = retarded_inv_matrices_lowerblk_ref[batch][i * blocksize * blocksize + j];
                }
            }
        }

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


        double norm_lesser_diagblk = 0.0;
        double norm_lesser_upperblk = 0.0;
        double norm_greater_diagblk = 0.0;
        double norm_greater_upperblk = 0.0;
        double diff_lesser_diagblk = 0.0;
        double diff_lesser_upperblk = 0.0;
        double diff_greater_diagblk = 0.0;
        double diff_greater_upperblk = 0.0;
        double norm_retarded_diagblk = 0.0;
        double norm_retarded_upperblk = 0.0;
        double norm_retarded_lowerblk = 0.0;
        double diff_retarded_diagblk = 0.0;
        double diff_retarded_upperblk = 0.0;
        double diff_retarded_lowerblk = 0.0;

        for(unsigned int i = 0; i < n_blocks; i++){
            for(unsigned j =  0; j < 1 * blocksize * blocksize; j++){
                norm_lesser_diagblk += std::abs(batch_lesser_inv_matrices_diagblk_ref[i][j]) * std::abs(batch_lesser_inv_matrices_diagblk_ref[i][j]);
                diff_lesser_diagblk += std::abs(batch_lesser_inv_matrices_diagblk_h[i][j] - batch_lesser_inv_matrices_diagblk_ref[i][j]) *
                    std::abs(batch_lesser_inv_matrices_diagblk_h[i][j] - batch_lesser_inv_matrices_diagblk_ref[i][j]);
                norm_greater_diagblk += std::abs(batch_greater_inv_matrices_diagblk_ref[i][j]) * std::abs(batch_greater_inv_matrices_diagblk_ref[i][j]);
                diff_greater_diagblk += std::abs(batch_greater_inv_matrices_diagblk_h[i][j] - batch_greater_inv_matrices_diagblk_ref[i][j]) *
                    std::abs(batch_greater_inv_matrices_diagblk_h[i][j] - batch_greater_inv_matrices_diagblk_ref[i][j]);
                norm_retarded_diagblk += std::abs(batch_retarded_inv_matrices_diagblk_ref[i][j]) * std::abs(batch_retarded_inv_matrices_diagblk_ref[i][j]);
                diff_retarded_diagblk += std::abs(batch_retarded_inv_matrices_diagblk_h[i][j] - batch_retarded_inv_matrices_diagblk_ref[i][j]) *
                    std::abs(batch_retarded_inv_matrices_diagblk_h[i][j] - batch_retarded_inv_matrices_diagblk_ref[i][j]);
            }

        }
        for(unsigned int i = 0; i < n_blocks - 1; i++){
            for(unsigned int j =  0; j < 1 * blocksize * blocksize; j++){
                norm_lesser_upperblk += std::abs(batch_lesser_inv_matrices_upperblk_ref[i][j]) * std::abs(batch_lesser_inv_matrices_upperblk_ref[i][j]);
                diff_lesser_upperblk += std::abs(batch_lesser_inv_matrices_upperblk_h[i][j] - batch_lesser_inv_matrices_upperblk_ref[i][j]) *
                    std::abs(batch_lesser_inv_matrices_upperblk_h[i][j] - batch_lesser_inv_matrices_upperblk_ref[i][j]);
                norm_greater_upperblk += std::abs(batch_greater_inv_matrices_upperblk_ref[i][j]) * std::abs(batch_greater_inv_matrices_upperblk_ref[i][j]);
                diff_greater_upperblk += std::abs(batch_greater_inv_matrices_upperblk_h[i][j] - batch_greater_inv_matrices_upperblk_ref[i][j]) *
                    std::abs(batch_greater_inv_matrices_upperblk_h[i][j] - batch_greater_inv_matrices_upperblk_ref[i][j]);
                norm_retarded_upperblk += std::abs(batch_retarded_inv_matrices_upperblk_ref[i][j]) * std::abs(batch_retarded_inv_matrices_upperblk_ref[i][j]);
                diff_retarded_upperblk += std::abs(batch_retarded_inv_matrices_upperblk_h[i][j] - batch_retarded_inv_matrices_upperblk_ref[i][j]) *
                    std::abs(batch_retarded_inv_matrices_upperblk_h[i][j] - batch_retarded_inv_matrices_upperblk_ref[i][j]);
                norm_retarded_lowerblk += std::abs(batch_retarded_inv_matrices_lowerblk_ref[i][j]) * std::abs(batch_retarded_inv_matrices_lowerblk_ref[i][j]);
                diff_retarded_lowerblk += std::abs(batch_retarded_inv_matrices_lowerblk_h[i][j] - batch_retarded_inv_matrices_lowerblk_ref[i][j]) *
                    std::abs(batch_retarded_inv_matrices_lowerblk_h[i][j] - batch_retarded_inv_matrices_lowerblk_ref[i][j]);
            }
        }
        if(diff_lesser_diagblk/norm_lesser_diagblk > reltol || diff_lesser_upperblk/norm_lesser_upperblk > reltol || diff_greater_diagblk/norm_greater_diagblk > reltol || diff_greater_upperblk/norm_greater_upperblk > reltol || diff_retarded_diagblk/norm_retarded_diagblk > reltol || diff_retarded_upperblk/norm_retarded_upperblk > reltol || diff_retarded_lowerblk/norm_retarded_lowerblk > reltol){
            std::cout << diff_lesser_diagblk/norm_lesser_diagblk << std::endl;
            std::cout << diff_lesser_upperblk/norm_lesser_upperblk << std::endl;
            std::cout << diff_greater_diagblk/norm_greater_diagblk << std::endl;
            std::cout << diff_greater_upperblk/norm_greater_upperblk << std::endl;
            std::cout << diff_retarded_diagblk/norm_retarded_diagblk << std::endl;
            std::cout << diff_retarded_upperblk/norm_retarded_upperblk << std::endl;
            std::cout << diff_retarded_lowerblk/norm_retarded_lowerblk << std::endl;
            printf("FAILED FOR lesser greater retarded\n");
            not_failed = false;
        }

        // set outputs to zero
        for(unsigned int i = 0; i < n_blocks; i++){
            for(unsigned int batch = 0; batch < batch_size; batch++){
                for(unsigned int j = 0; j < blocksize * blocksize; j++){
                    batch_lesser_inv_matrices_diagblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                    batch_greater_inv_matrices_diagblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                    batch_retarded_inv_matrices_diagblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                }
            }
        }
        for(unsigned int i = 0; i < n_blocks-1; i++){
            for(unsigned int batch = 0; batch < batch_size; batch++){
                for(unsigned int j = 0; j < blocksize * blocksize; j++){
                    batch_lesser_inv_matrices_upperblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                    batch_greater_inv_matrices_upperblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                    batch_retarded_inv_matrices_upperblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                    batch_retarded_inv_matrices_lowerblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                }
            }
        }


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

        norm_lesser_diagblk = 0.0;
        norm_lesser_upperblk = 0.0;
        norm_greater_diagblk = 0.0;
        norm_greater_upperblk = 0.0;
        diff_lesser_diagblk = 0.0;
        diff_lesser_upperblk = 0.0;
        diff_greater_diagblk = 0.0;
        diff_greater_upperblk = 0.0;
        norm_retarded_diagblk = 0.0;
        norm_retarded_upperblk = 0.0;
        norm_retarded_lowerblk = 0.0;
        diff_retarded_diagblk = 0.0;
        diff_retarded_upperblk = 0.0;
        diff_retarded_lowerblk = 0.0;

        for(unsigned int i = 0; i < n_blocks; i++){
            for(unsigned j =  0; j < 1 * blocksize * blocksize; j++){
                norm_lesser_diagblk += std::abs(batch_lesser_inv_matrices_diagblk_ref[i][j]) * std::abs(batch_lesser_inv_matrices_diagblk_ref[i][j]);
                diff_lesser_diagblk += std::abs(batch_lesser_inv_matrices_diagblk_h[i][j] - batch_lesser_inv_matrices_diagblk_ref[i][j]) *
                    std::abs(batch_lesser_inv_matrices_diagblk_h[i][j] - batch_lesser_inv_matrices_diagblk_ref[i][j]);
                norm_greater_diagblk += std::abs(batch_greater_inv_matrices_diagblk_ref[i][j]) * std::abs(batch_greater_inv_matrices_diagblk_ref[i][j]);
                diff_greater_diagblk += std::abs(batch_greater_inv_matrices_diagblk_h[i][j] - batch_greater_inv_matrices_diagblk_ref[i][j]) *
                    std::abs(batch_greater_inv_matrices_diagblk_h[i][j] - batch_greater_inv_matrices_diagblk_ref[i][j]);
                norm_retarded_diagblk += std::abs(batch_retarded_inv_matrices_diagblk_ref[i][j]) * std::abs(batch_retarded_inv_matrices_diagblk_ref[i][j]);
                diff_retarded_diagblk += std::abs(batch_retarded_inv_matrices_diagblk_h[i][j] - batch_retarded_inv_matrices_diagblk_ref[i][j]) *
                    std::abs(batch_retarded_inv_matrices_diagblk_h[i][j] - batch_retarded_inv_matrices_diagblk_ref[i][j]);
            }

        }
        for(unsigned int i = 0; i < n_blocks - 1; i++){
            for(unsigned int j =  0; j < 1 * blocksize * blocksize; j++){
                norm_lesser_upperblk += std::abs(batch_lesser_inv_matrices_upperblk_ref[i][j]) * std::abs(batch_lesser_inv_matrices_upperblk_ref[i][j]);
                diff_lesser_upperblk += std::abs(batch_lesser_inv_matrices_upperblk_h[i][j] - batch_lesser_inv_matrices_upperblk_ref[i][j]) *
                    std::abs(batch_lesser_inv_matrices_upperblk_h[i][j] - batch_lesser_inv_matrices_upperblk_ref[i][j]);
                norm_greater_upperblk += std::abs(batch_greater_inv_matrices_upperblk_ref[i][j]) * std::abs(batch_greater_inv_matrices_upperblk_ref[i][j]);
                diff_greater_upperblk += std::abs(batch_greater_inv_matrices_upperblk_h[i][j] - batch_greater_inv_matrices_upperblk_ref[i][j]) *
                    std::abs(batch_greater_inv_matrices_upperblk_h[i][j] - batch_greater_inv_matrices_upperblk_ref[i][j]);
                norm_retarded_upperblk += std::abs(batch_retarded_inv_matrices_upperblk_ref[i][j]) * std::abs(batch_retarded_inv_matrices_upperblk_ref[i][j]);
                diff_retarded_upperblk += std::abs(batch_retarded_inv_matrices_upperblk_h[i][j] - batch_retarded_inv_matrices_upperblk_ref[i][j]) *
                    std::abs(batch_retarded_inv_matrices_upperblk_h[i][j] - batch_retarded_inv_matrices_upperblk_ref[i][j]);
                norm_retarded_lowerblk += std::abs(batch_retarded_inv_matrices_lowerblk_ref[i][j]) * std::abs(batch_retarded_inv_matrices_lowerblk_ref[i][j]);
                diff_retarded_lowerblk += std::abs(batch_retarded_inv_matrices_lowerblk_h[i][j] - batch_retarded_inv_matrices_lowerblk_ref[i][j]) *
                    std::abs(batch_retarded_inv_matrices_lowerblk_h[i][j] - batch_retarded_inv_matrices_lowerblk_ref[i][j]);
            }
        }
        if(diff_lesser_diagblk/norm_lesser_diagblk > reltol || diff_lesser_upperblk/norm_lesser_upperblk > reltol || diff_greater_diagblk/norm_greater_diagblk > reltol || diff_greater_upperblk/norm_greater_upperblk > reltol || diff_retarded_diagblk/norm_retarded_diagblk > reltol || diff_retarded_upperblk/norm_retarded_upperblk > reltol || diff_retarded_lowerblk/norm_retarded_lowerblk > reltol){
            std::cout << diff_lesser_diagblk/norm_lesser_diagblk << std::endl;
            std::cout << diff_lesser_upperblk/norm_lesser_upperblk << std::endl;
            std::cout << diff_greater_diagblk/norm_greater_diagblk << std::endl;
            std::cout << diff_greater_upperblk/norm_greater_upperblk << std::endl;
            std::cout << diff_retarded_diagblk/norm_retarded_diagblk << std::endl;
            std::cout << diff_retarded_upperblk/norm_retarded_upperblk << std::endl;
            std::cout << diff_retarded_lowerblk/norm_retarded_lowerblk << std::endl;
            printf("FAILED BATCHED lesser greater retarded\n");
            not_failed = false;
        }

        // set outputs to zero
        for(unsigned int i = 0; i < n_blocks; i++){
            for(unsigned int batch = 0; batch < batch_size; batch++){
                for(unsigned int j = 0; j < blocksize * blocksize; j++){
                    batch_lesser_inv_matrices_diagblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                    batch_greater_inv_matrices_diagblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                    batch_retarded_inv_matrices_diagblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                }
            }
        }
        for(unsigned int i = 0; i < n_blocks-1; i++){
            for(unsigned int batch = 0; batch < batch_size; batch++){
                for(unsigned int j = 0; j < blocksize * blocksize; j++){
                    batch_lesser_inv_matrices_upperblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                    batch_greater_inv_matrices_upperblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                    batch_retarded_inv_matrices_upperblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                    batch_retarded_inv_matrices_lowerblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                }
            }
        }

        rgf_lesser_greater_for(
            blocksize, matrix_size, batch_size,
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
            batch_greater_inv_matrices_upperblk_h);

        norm_lesser_diagblk = 0.0;
        norm_lesser_upperblk = 0.0;
        norm_greater_diagblk = 0.0;
        norm_greater_upperblk = 0.0;
        diff_lesser_diagblk = 0.0;
        diff_lesser_upperblk = 0.0;
        diff_greater_diagblk = 0.0;
        diff_greater_upperblk = 0.0;

        for(unsigned int i = 0; i < n_blocks; i++){
            for(unsigned j =  0; j < 1 * blocksize * blocksize; j++){
                norm_lesser_diagblk += std::abs(batch_lesser_inv_matrices_diagblk_ref[i][j]) * std::abs(batch_lesser_inv_matrices_diagblk_ref[i][j]);
                diff_lesser_diagblk += std::abs(batch_lesser_inv_matrices_diagblk_h[i][j] - batch_lesser_inv_matrices_diagblk_ref[i][j]) *
                    std::abs(batch_lesser_inv_matrices_diagblk_h[i][j] - batch_lesser_inv_matrices_diagblk_ref[i][j]);
                norm_greater_diagblk += std::abs(batch_greater_inv_matrices_diagblk_ref[i][j]) * std::abs(batch_greater_inv_matrices_diagblk_ref[i][j]);
                diff_greater_diagblk += std::abs(batch_greater_inv_matrices_diagblk_h[i][j] - batch_greater_inv_matrices_diagblk_ref[i][j]) *
                    std::abs(batch_greater_inv_matrices_diagblk_h[i][j] - batch_greater_inv_matrices_diagblk_ref[i][j]);
            }

        }
        for(unsigned int i = 0; i < n_blocks - 1; i++){
            for(unsigned int j =  0; j < 1 * blocksize * blocksize; j++){
                norm_lesser_upperblk += std::abs(batch_lesser_inv_matrices_upperblk_ref[i][j]) * std::abs(batch_lesser_inv_matrices_upperblk_ref[i][j]);
                diff_lesser_upperblk += std::abs(batch_lesser_inv_matrices_upperblk_h[i][j] - batch_lesser_inv_matrices_upperblk_ref[i][j]) *
                    std::abs(batch_lesser_inv_matrices_upperblk_h[i][j] - batch_lesser_inv_matrices_upperblk_ref[i][j]);
                norm_greater_upperblk += std::abs(batch_greater_inv_matrices_upperblk_ref[i][j]) * std::abs(batch_greater_inv_matrices_upperblk_ref[i][j]);
                diff_greater_upperblk += std::abs(batch_greater_inv_matrices_upperblk_h[i][j] - batch_greater_inv_matrices_upperblk_ref[i][j]) *
                    std::abs(batch_greater_inv_matrices_upperblk_h[i][j] - batch_greater_inv_matrices_upperblk_ref[i][j]);
            }
        }
        if(diff_lesser_diagblk/norm_lesser_diagblk > reltol || diff_lesser_upperblk/norm_lesser_upperblk > reltol || diff_greater_diagblk/norm_greater_diagblk > reltol || diff_greater_upperblk/norm_greater_upperblk > reltol){
            std::cout << diff_lesser_diagblk/norm_lesser_diagblk << std::endl;
            std::cout << diff_lesser_upperblk/norm_lesser_upperblk << std::endl;
            std::cout << diff_greater_diagblk/norm_greater_diagblk << std::endl;
            std::cout << diff_greater_upperblk/norm_greater_upperblk << std::endl;
            printf("FAILED FOR lesser greater\n");
            not_failed = false;
        }

        // set outputs to zero
        for(unsigned int i = 0; i < n_blocks; i++){
            for(unsigned int batch = 0; batch < batch_size; batch++){
                for(unsigned int j = 0; j < blocksize * blocksize; j++){
                    batch_lesser_inv_matrices_diagblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                    batch_greater_inv_matrices_diagblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                }
            }
        }
        for(unsigned int i = 0; i < n_blocks-1; i++){
            for(unsigned int batch = 0; batch < batch_size; batch++){
                for(unsigned int j = 0; j < blocksize * blocksize; j++){
                    batch_lesser_inv_matrices_upperblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                    batch_greater_inv_matrices_upperblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                }
            }
        }


        rgf_lesser_greater_batched(
            blocksize, matrix_size, batch_size,
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
            batch_greater_inv_matrices_upperblk_h);

        norm_lesser_diagblk = 0.0;
        norm_lesser_upperblk = 0.0;
        norm_greater_diagblk = 0.0;
        norm_greater_upperblk = 0.0;
        diff_lesser_diagblk = 0.0;
        diff_lesser_upperblk = 0.0;
        diff_greater_diagblk = 0.0;
        diff_greater_upperblk = 0.0;

        for(unsigned int i = 0; i < n_blocks; i++){
            for(unsigned j =  0; j < 1 * blocksize * blocksize; j++){
                norm_lesser_diagblk += std::abs(batch_lesser_inv_matrices_diagblk_ref[i][j]) * std::abs(batch_lesser_inv_matrices_diagblk_ref[i][j]);
                diff_lesser_diagblk += std::abs(batch_lesser_inv_matrices_diagblk_h[i][j] - batch_lesser_inv_matrices_diagblk_ref[i][j]) *
                    std::abs(batch_lesser_inv_matrices_diagblk_h[i][j] - batch_lesser_inv_matrices_diagblk_ref[i][j]);
                norm_greater_diagblk += std::abs(batch_greater_inv_matrices_diagblk_ref[i][j]) * std::abs(batch_greater_inv_matrices_diagblk_ref[i][j]);
                diff_greater_diagblk += std::abs(batch_greater_inv_matrices_diagblk_h[i][j] - batch_greater_inv_matrices_diagblk_ref[i][j]) *
                    std::abs(batch_greater_inv_matrices_diagblk_h[i][j] - batch_greater_inv_matrices_diagblk_ref[i][j]);
            }

        }
        for(unsigned int i = 0; i < n_blocks - 1; i++){
            for(unsigned int j =  0; j < 1 * blocksize * blocksize; j++){
                norm_lesser_upperblk += std::abs(batch_lesser_inv_matrices_upperblk_ref[i][j]) * std::abs(batch_lesser_inv_matrices_upperblk_ref[i][j]);
                diff_lesser_upperblk += std::abs(batch_lesser_inv_matrices_upperblk_h[i][j] - batch_lesser_inv_matrices_upperblk_ref[i][j]) *
                    std::abs(batch_lesser_inv_matrices_upperblk_h[i][j] - batch_lesser_inv_matrices_upperblk_ref[i][j]);
                norm_greater_upperblk += std::abs(batch_greater_inv_matrices_upperblk_ref[i][j]) * std::abs(batch_greater_inv_matrices_upperblk_ref[i][j]);
                diff_greater_upperblk += std::abs(batch_greater_inv_matrices_upperblk_h[i][j] - batch_greater_inv_matrices_upperblk_ref[i][j]) *
                    std::abs(batch_greater_inv_matrices_upperblk_h[i][j] - batch_greater_inv_matrices_upperblk_ref[i][j]);
            }
        }
        if(diff_lesser_diagblk/norm_lesser_diagblk > reltol || diff_lesser_upperblk/norm_lesser_upperblk > reltol || diff_greater_diagblk/norm_greater_diagblk > reltol || diff_greater_upperblk/norm_greater_upperblk > reltol){
            std::cout << diff_lesser_diagblk/norm_lesser_diagblk << std::endl;
            std::cout << diff_lesser_upperblk/norm_lesser_upperblk << std::endl;
            std::cout << diff_greater_diagblk/norm_greater_diagblk << std::endl;
            std::cout << diff_greater_upperblk/norm_greater_upperblk << std::endl;
            printf("FAILED BATCHED lesser greater\n");
            not_failed = false;
        }

        // set outputs to zero
        for(unsigned int i = 0; i < n_blocks; i++){
            for(unsigned int batch = 0; batch < batch_size; batch++){
                for(unsigned int j = 0; j < blocksize * blocksize; j++){
                    batch_lesser_inv_matrices_diagblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                    batch_greater_inv_matrices_diagblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                }
            }
        }
        for(unsigned int i = 0; i < n_blocks-1; i++){
            for(unsigned int batch = 0; batch < batch_size; batch++){
                for(unsigned int j = 0; j < blocksize * blocksize; j++){
                    batch_lesser_inv_matrices_upperblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                    batch_greater_inv_matrices_upperblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                }
            }
        }

        rgf_lesser_greater_batched_without_hostmalloc(
            blocksize, matrix_size, batch_size,
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
            batch_greater_inv_matrices_upperblk_h);

        norm_lesser_diagblk = 0.0;
        norm_lesser_upperblk = 0.0;
        norm_greater_diagblk = 0.0;
        norm_greater_upperblk = 0.0;
        diff_lesser_diagblk = 0.0;
        diff_lesser_upperblk = 0.0;
        diff_greater_diagblk = 0.0;
        diff_greater_upperblk = 0.0;

        for(unsigned int i = 0; i < n_blocks; i++){
            for(unsigned j =  0; j < 1 * blocksize * blocksize; j++){
                norm_lesser_diagblk += std::abs(batch_lesser_inv_matrices_diagblk_ref[i][j]) * std::abs(batch_lesser_inv_matrices_diagblk_ref[i][j]);
                diff_lesser_diagblk += std::abs(batch_lesser_inv_matrices_diagblk_h[i][j] - batch_lesser_inv_matrices_diagblk_ref[i][j]) *
                    std::abs(batch_lesser_inv_matrices_diagblk_h[i][j] - batch_lesser_inv_matrices_diagblk_ref[i][j]);
                norm_greater_diagblk += std::abs(batch_greater_inv_matrices_diagblk_ref[i][j]) * std::abs(batch_greater_inv_matrices_diagblk_ref[i][j]);
                diff_greater_diagblk += std::abs(batch_greater_inv_matrices_diagblk_h[i][j] - batch_greater_inv_matrices_diagblk_ref[i][j]) *
                    std::abs(batch_greater_inv_matrices_diagblk_h[i][j] - batch_greater_inv_matrices_diagblk_ref[i][j]);
            }

        }
        for(unsigned int i = 0; i < n_blocks - 1; i++){
            for(unsigned int j =  0; j < 1 * blocksize * blocksize; j++){
                norm_lesser_upperblk += std::abs(batch_lesser_inv_matrices_upperblk_ref[i][j]) * std::abs(batch_lesser_inv_matrices_upperblk_ref[i][j]);
                diff_lesser_upperblk += std::abs(batch_lesser_inv_matrices_upperblk_h[i][j] - batch_lesser_inv_matrices_upperblk_ref[i][j]) *
                    std::abs(batch_lesser_inv_matrices_upperblk_h[i][j] - batch_lesser_inv_matrices_upperblk_ref[i][j]);
                norm_greater_upperblk += std::abs(batch_greater_inv_matrices_upperblk_ref[i][j]) * std::abs(batch_greater_inv_matrices_upperblk_ref[i][j]);
                diff_greater_upperblk += std::abs(batch_greater_inv_matrices_upperblk_h[i][j] - batch_greater_inv_matrices_upperblk_ref[i][j]) *
                    std::abs(batch_greater_inv_matrices_upperblk_h[i][j] - batch_greater_inv_matrices_upperblk_ref[i][j]);
            }
        }
        if(diff_lesser_diagblk/norm_lesser_diagblk > reltol || diff_lesser_upperblk/norm_lesser_upperblk > reltol || diff_greater_diagblk/norm_greater_diagblk > reltol || diff_greater_upperblk/norm_greater_upperblk > reltol){
            std::cout << diff_lesser_diagblk/norm_lesser_diagblk << std::endl;
            std::cout << diff_lesser_upperblk/norm_lesser_upperblk << std::endl;
            std::cout << diff_greater_diagblk/norm_greater_diagblk << std::endl;
            std::cout << diff_greater_upperblk/norm_greater_upperblk << std::endl;
            printf("FAILED BATCHED w.o. host malloc lesser greater\n");
            not_failed = false;
        }

        // set outputs to zero
        for(unsigned int i = 0; i < n_blocks; i++){
            for(unsigned int batch = 0; batch < batch_size; batch++){
                for(unsigned int j = 0; j < blocksize * blocksize; j++){
                    batch_lesser_inv_matrices_diagblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                    batch_greater_inv_matrices_diagblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                }
            }
        }
        for(unsigned int i = 0; i < n_blocks-1; i++){
            for(unsigned int batch = 0; batch < batch_size; batch++){
                for(unsigned int j = 0; j < blocksize * blocksize; j++){
                    batch_lesser_inv_matrices_upperblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                    batch_greater_inv_matrices_upperblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                }
            }
        }


        rgf_retarded_for(blocksize, matrix_size, batch_size,
                                    batch_system_matrices_diagblk_h,
                                    batch_system_matrices_upperblk_h,
                                    batch_system_matrices_lowerblk_h,
                                    batch_retarded_inv_matrices_diagblk_h,
                                    batch_retarded_inv_matrices_upperblk_h,
                                    batch_retarded_inv_matrices_lowerblk_h);

        norm_retarded_diagblk = 0.0;
        norm_retarded_upperblk = 0.0;
        norm_retarded_lowerblk = 0.0;
        diff_retarded_diagblk = 0.0;
        diff_retarded_upperblk = 0.0;
        diff_retarded_lowerblk = 0.0;

        for(unsigned int i = 0; i < n_blocks; i++){
            for(unsigned j =  0; j < 1 * blocksize * blocksize; j++){
                norm_retarded_diagblk += std::abs(batch_retarded_inv_matrices_diagblk_ref[i][j]) * std::abs(batch_retarded_inv_matrices_diagblk_ref[i][j]);
                diff_retarded_diagblk += std::abs(batch_retarded_inv_matrices_diagblk_h[i][j] - batch_retarded_inv_matrices_diagblk_ref[i][j]) *
                    std::abs(batch_retarded_inv_matrices_diagblk_h[i][j] - batch_retarded_inv_matrices_diagblk_ref[i][j]);
            }

        }
        for(unsigned int i = 0; i < n_blocks - 1; i++){
            for(unsigned int j =  0; j < 1 * blocksize * blocksize; j++){
                norm_retarded_upperblk += std::abs(batch_retarded_inv_matrices_upperblk_ref[i][j]) * std::abs(batch_retarded_inv_matrices_upperblk_ref[i][j]);
                diff_retarded_upperblk += std::abs(batch_retarded_inv_matrices_upperblk_h[i][j] - batch_retarded_inv_matrices_upperblk_ref[i][j]) *
                    std::abs(batch_retarded_inv_matrices_upperblk_h[i][j] - batch_retarded_inv_matrices_upperblk_ref[i][j]);
                norm_retarded_lowerblk += std::abs(batch_retarded_inv_matrices_lowerblk_ref[i][j]) * std::abs(batch_retarded_inv_matrices_lowerblk_ref[i][j]);
                diff_retarded_lowerblk += std::abs(batch_retarded_inv_matrices_lowerblk_h[i][j] - batch_retarded_inv_matrices_lowerblk_ref[i][j]) *
                    std::abs(batch_retarded_inv_matrices_lowerblk_h[i][j] - batch_retarded_inv_matrices_lowerblk_ref[i][j]);
            }
        }
        if(diff_retarded_diagblk/norm_retarded_diagblk > reltol || diff_retarded_upperblk/norm_retarded_upperblk > reltol || diff_retarded_lowerblk/norm_retarded_lowerblk > reltol){
            std::cout << diff_retarded_diagblk/norm_retarded_diagblk << std::endl;
            std::cout << diff_retarded_upperblk/norm_retarded_upperblk << std::endl;
            std::cout << diff_retarded_lowerblk/norm_retarded_lowerblk << std::endl;
            printf("FAILED FOR retarded\n");
            not_failed = false;
        }

        // set outputs to zero
        for(unsigned int i = 0; i < n_blocks; i++){
            for(unsigned int batch = 0; batch < batch_size; batch++){
                for(unsigned int j = 0; j < blocksize * blocksize; j++){
                    batch_retarded_inv_matrices_diagblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                }
            }
        }
        for(unsigned int i = 0; i < n_blocks-1; i++){
            for(unsigned int batch = 0; batch < batch_size; batch++){
                for(unsigned int j = 0; j < blocksize * blocksize; j++){
                    batch_retarded_inv_matrices_upperblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                    batch_retarded_inv_matrices_lowerblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                }
            }
        }

        rgf_retarded_batched(blocksize, matrix_size, batch_size,
                                    batch_system_matrices_diagblk_h,
                                    batch_system_matrices_upperblk_h,
                                    batch_system_matrices_lowerblk_h,
                                    batch_retarded_inv_matrices_diagblk_h,
                                    batch_retarded_inv_matrices_upperblk_h,
                                    batch_retarded_inv_matrices_lowerblk_h);


        norm_retarded_diagblk = 0.0;
        norm_retarded_upperblk = 0.0;
        norm_retarded_lowerblk = 0.0;
        diff_retarded_diagblk = 0.0;
        diff_retarded_upperblk = 0.0;
        diff_retarded_lowerblk = 0.0;

        for(unsigned int i = 0; i < n_blocks; i++){
            for(unsigned j =  0; j < 1 * blocksize * blocksize; j++){
                norm_retarded_diagblk += std::abs(batch_retarded_inv_matrices_diagblk_ref[i][j]) * std::abs(batch_retarded_inv_matrices_diagblk_ref[i][j]);
                diff_retarded_diagblk += std::abs(batch_retarded_inv_matrices_diagblk_h[i][j] - batch_retarded_inv_matrices_diagblk_ref[i][j]) *
                    std::abs(batch_retarded_inv_matrices_diagblk_h[i][j] - batch_retarded_inv_matrices_diagblk_ref[i][j]);
            }

        }
        for(unsigned int i = 0; i < n_blocks - 1; i++){
            for(unsigned int j =  0; j < 1 * blocksize * blocksize; j++){
                norm_retarded_upperblk += std::abs(batch_retarded_inv_matrices_upperblk_ref[i][j]) * std::abs(batch_retarded_inv_matrices_upperblk_ref[i][j]);
                diff_retarded_upperblk += std::abs(batch_retarded_inv_matrices_upperblk_h[i][j] - batch_retarded_inv_matrices_upperblk_ref[i][j]) *
                    std::abs(batch_retarded_inv_matrices_upperblk_h[i][j] - batch_retarded_inv_matrices_upperblk_ref[i][j]);
                norm_retarded_lowerblk += std::abs(batch_retarded_inv_matrices_lowerblk_ref[i][j]) * std::abs(batch_retarded_inv_matrices_lowerblk_ref[i][j]);
                diff_retarded_lowerblk += std::abs(batch_retarded_inv_matrices_lowerblk_h[i][j] - batch_retarded_inv_matrices_lowerblk_ref[i][j]) *
                    std::abs(batch_retarded_inv_matrices_lowerblk_h[i][j] - batch_retarded_inv_matrices_lowerblk_ref[i][j]);
            }
        }
        if(diff_retarded_diagblk/norm_retarded_diagblk > reltol || diff_retarded_upperblk/norm_retarded_upperblk > reltol || diff_retarded_lowerblk/norm_retarded_lowerblk > reltol){
            std::cout << diff_retarded_diagblk/norm_retarded_diagblk << std::endl;
            std::cout << diff_retarded_upperblk/norm_retarded_upperblk << std::endl;
            std::cout << diff_retarded_lowerblk/norm_retarded_lowerblk << std::endl;
            printf("FAILED BATCHED retarded\n");
            not_failed = false;
        }

        // set outputs to zero
        for(unsigned int i = 0; i < n_blocks; i++){
            for(unsigned int batch = 0; batch < batch_size; batch++){
                for(unsigned int j = 0; j < blocksize * blocksize; j++){
                    batch_retarded_inv_matrices_diagblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                }
            }
        }
        for(unsigned int i = 0; i < n_blocks-1; i++){
            for(unsigned int batch = 0; batch < batch_size; batch++){
                for(unsigned int j = 0; j < blocksize * blocksize; j++){
                    batch_retarded_inv_matrices_upperblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                    batch_retarded_inv_matrices_lowerblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                }
            }
        }

        rgf_retarded_batched_strided(blocksize, matrix_size, batch_size,
                                    batch_system_matrices_diagblk_h,
                                    batch_system_matrices_upperblk_h,
                                    batch_system_matrices_lowerblk_h,
                                    batch_retarded_inv_matrices_diagblk_h,
                                    batch_retarded_inv_matrices_upperblk_h,
                                    batch_retarded_inv_matrices_lowerblk_h);


        norm_retarded_diagblk = 0.0;
        norm_retarded_upperblk = 0.0;
        norm_retarded_lowerblk = 0.0;
        diff_retarded_diagblk = 0.0;
        diff_retarded_upperblk = 0.0;
        diff_retarded_lowerblk = 0.0;

        for(unsigned int i = 0; i < n_blocks; i++){
            for(unsigned j =  0; j < 1 * blocksize * blocksize; j++){
                norm_retarded_diagblk += std::abs(batch_retarded_inv_matrices_diagblk_ref[i][j]) * std::abs(batch_retarded_inv_matrices_diagblk_ref[i][j]);
                diff_retarded_diagblk += std::abs(batch_retarded_inv_matrices_diagblk_h[i][j] - batch_retarded_inv_matrices_diagblk_ref[i][j]) *
                    std::abs(batch_retarded_inv_matrices_diagblk_h[i][j] - batch_retarded_inv_matrices_diagblk_ref[i][j]);
            }

        }
        for(unsigned int i = 0; i < n_blocks - 1; i++){
            for(unsigned int j =  0; j < 1 * blocksize * blocksize; j++){
                norm_retarded_upperblk += std::abs(batch_retarded_inv_matrices_upperblk_ref[i][j]) * std::abs(batch_retarded_inv_matrices_upperblk_ref[i][j]);
                diff_retarded_upperblk += std::abs(batch_retarded_inv_matrices_upperblk_h[i][j] - batch_retarded_inv_matrices_upperblk_ref[i][j]) *
                    std::abs(batch_retarded_inv_matrices_upperblk_h[i][j] - batch_retarded_inv_matrices_upperblk_ref[i][j]);
                norm_retarded_lowerblk += std::abs(batch_retarded_inv_matrices_lowerblk_ref[i][j]) * std::abs(batch_retarded_inv_matrices_lowerblk_ref[i][j]);
                diff_retarded_lowerblk += std::abs(batch_retarded_inv_matrices_lowerblk_h[i][j] - batch_retarded_inv_matrices_lowerblk_ref[i][j]) *
                    std::abs(batch_retarded_inv_matrices_lowerblk_h[i][j] - batch_retarded_inv_matrices_lowerblk_ref[i][j]);
            }
        }
        if(diff_retarded_diagblk/norm_retarded_diagblk > reltol || diff_retarded_upperblk/norm_retarded_upperblk > reltol || diff_retarded_lowerblk/norm_retarded_lowerblk > reltol){
            std::cout << diff_retarded_diagblk/norm_retarded_diagblk << std::endl;
            std::cout << diff_retarded_upperblk/norm_retarded_upperblk << std::endl;
            std::cout << diff_retarded_lowerblk/norm_retarded_lowerblk << std::endl;
            printf("FAILED BATCHED retarded strided\n");
            not_failed = false;
        }

        // set outputs to zero
        for(unsigned int i = 0; i < n_blocks; i++){
            for(unsigned int batch = 0; batch < batch_size; batch++){
                for(unsigned int j = 0; j < blocksize * blocksize; j++){
                    batch_retarded_inv_matrices_diagblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                }
            }
        }
        for(unsigned int i = 0; i < n_blocks-1; i++){
            for(unsigned int batch = 0; batch < batch_size; batch++){
                for(unsigned int j = 0; j < blocksize * blocksize; j++){
                    batch_retarded_inv_matrices_upperblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                    batch_retarded_inv_matrices_lowerblk_h[i][batch * blocksize * blocksize + j] = 0.0;
                }
            }
        }

        //free batched memory
        for(unsigned int i = 0; i < n_blocks; i++){
            cudaFreeHost(batch_system_matrices_diagblk_h[i]);
            cudaFreeHost(batch_self_energy_matrices_lesser_diagblk_h[i]);
            cudaFreeHost(batch_self_energy_matrices_greater_diagblk_h[i]);
            cudaFreeHost(batch_lesser_inv_matrices_diagblk_ref[i]);
            cudaFreeHost(batch_greater_inv_matrices_diagblk_ref[i]);
            cudaFreeHost(batch_lesser_inv_matrices_diagblk_h[i]);
            cudaFreeHost(batch_greater_inv_matrices_diagblk_h[i]);
            cudaFreeHost(batch_retarded_inv_matrices_diagblk_ref[i]);
            cudaFreeHost(batch_retarded_inv_matrices_diagblk_h[i]);
        }
        for(unsigned int i = 0; i < n_blocks-1; i++){
            cudaFreeHost(batch_system_matrices_upperblk_h[i]);
            cudaFreeHost(batch_system_matrices_lowerblk_h[i]);
            cudaFreeHost(batch_self_energy_matrices_lesser_upperblk_h[i]);
            cudaFreeHost(batch_self_energy_matrices_greater_upperblk_h[i]);
            cudaFreeHost(batch_lesser_inv_matrices_upperblk_ref[i]);
            cudaFreeHost(batch_greater_inv_matrices_upperblk_ref[i]);
            cudaFreeHost(batch_lesser_inv_matrices_upperblk_h[i]);
            cudaFreeHost(batch_greater_inv_matrices_upperblk_h[i]);
            cudaFreeHost(batch_retarded_inv_matrices_upperblk_ref[i]);
            cudaFreeHost(batch_retarded_inv_matrices_lowerblk_ref[i]);
            cudaFreeHost(batch_retarded_inv_matrices_upperblk_h[i]);
            cudaFreeHost(batch_retarded_inv_matrices_lowerblk_h[i]);
        }
    }
    if(not_failed){
        printf("PASSED\n");
    }
    else{
        printf("FAILED\n");
    }

    // free contiguous memory
    for(unsigned int batch = 0; batch < batch_size; batch++){
        cudaFreeHost(system_matrices_diagblk_h[batch]);
        cudaFreeHost(system_matrices_upperblk_h[batch]);
        cudaFreeHost(system_matrices_lowerblk_h[batch]);
        cudaFreeHost(self_energy_matrices_lesser_diagblk_h[batch]);
        cudaFreeHost(self_energy_matrices_lesser_upperblk_h[batch]);
        cudaFreeHost(self_energy_matrices_greater_diagblk_h[batch]);
        cudaFreeHost(self_energy_matrices_greater_upperblk_h[batch]);
        cudaFreeHost(lesser_inv_matrices_diagblk_ref[batch]);
        cudaFreeHost(lesser_inv_matrices_upperblk_ref[batch]);
        cudaFreeHost(greater_inv_matrices_diagblk_ref[batch]);
        cudaFreeHost(greater_inv_matrices_upperblk_ref[batch]);
        cudaFreeHost(retarded_inv_matrices_diagblk_ref[batch]);
        cudaFreeHost(retarded_inv_matrices_upperblk_ref[batch]);
        cudaFreeHost(retarded_inv_matrices_lowerblk_ref[batch]);
    }
    return 0;
}








