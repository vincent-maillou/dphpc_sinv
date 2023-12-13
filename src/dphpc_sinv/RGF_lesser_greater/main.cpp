#include "lesser_greater.h"


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

    complex_h *system_matrices_diagblk_h[batch_size];
    complex_h *system_matrices_upperblk_h[batch_size];
    complex_h *system_matrices_lowerblk_h[batch_size];
    complex_h *self_energy_matrices_lesser_diagblk_h[batch_size];
    complex_h *self_energy_matrices_lesser_upperblk_h[batch_size];
    complex_h *self_energy_matrices_lesser_lowerblk_h[batch_size];
    complex_h *self_energy_matrices_greater_diagblk_h[batch_size];
    complex_h *self_energy_matrices_greater_upperblk_h[batch_size];
    complex_h *self_energy_matrices_greater_lowerblk_h[batch_size];

    complex_h *lesser_inv_matrices_diagblk_ref[batch_size];
    complex_h *lesser_inv_matrices_upperblk_ref[batch_size];
    complex_h *lesser_inv_matrices_lowerblk_ref[batch_size];
    complex_h *greater_inv_matrices_diagblk_ref[batch_size];
    complex_h *greater_inv_matrices_upperblk_ref[batch_size];
    complex_h *greater_inv_matrices_lowerblk_ref[batch_size];

    for(unsigned int batch = 0; batch < batch_size; batch++){

        // Load matrix to invert
        complex_h* system_matrix_diagblk = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
        std::string diagblk_path = base_path + "system_matrix_"+ std::to_string(batch) +"_diagblk.bin";
        load_binary_matrix(diagblk_path.c_str(), system_matrix_diagblk, blocksize, matrix_size);

        complex_h* system_matrix_upperblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
        std::string upperblk_path = base_path + "system_matrix_"+ std::to_string(batch) +"_upperblk.bin";
        load_binary_matrix(upperblk_path.c_str(), system_matrix_upperblk, blocksize, off_diag_size);

        complex_h* system_matrix_lowerblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
        std::string lowerblk_path = base_path + "system_matrix_"+ std::to_string(batch) +"_lowerblk.bin";
        load_binary_matrix(lowerblk_path.c_str(), system_matrix_lowerblk, blocksize, off_diag_size);

        // load the self energy
        complex_h* self_energy_lesser_diagblk = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
        std::string self_energy_lesser_diagblk_path = base_path + "self_energy_lesser_"+ std::to_string(batch) +"_diagblk.bin";
        load_binary_matrix(self_energy_lesser_diagblk_path.c_str(), self_energy_lesser_diagblk, blocksize, matrix_size);

        complex_h* self_energy_lesser_upperblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
        std::string self_energy_lesser_upperblk_path = base_path + "self_energy_lesser_"+ std::to_string(batch) +"_upperblk.bin";
        load_binary_matrix(self_energy_lesser_upperblk_path.c_str(), self_energy_lesser_upperblk, blocksize, off_diag_size);

        complex_h* self_energy_lesser_lowerblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
        std::string self_energy_lesser_lowerblk_path = base_path + "self_energy_lesser_"+ std::to_string(batch) +"_lowerblk.bin";
        load_binary_matrix(self_energy_lesser_lowerblk_path.c_str(), self_energy_lesser_lowerblk, blocksize, off_diag_size);

        complex_h* self_energy_greater_diagblk = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
        std::string self_energy_greater_diagblk_path = base_path + "self_energy_greater_"+ std::to_string(batch) +"_diagblk.bin";
        load_binary_matrix(self_energy_greater_diagblk_path.c_str(), self_energy_greater_diagblk, blocksize, matrix_size);

        complex_h* self_energy_greater_upperblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
        std::string self_energy_greater_upperblk_path = base_path + "self_energy_greater_"+ std::to_string(batch) +"_upperblk.bin";
        load_binary_matrix(self_energy_greater_upperblk_path.c_str(), self_energy_greater_upperblk, blocksize, off_diag_size);

        complex_h* self_energy_greater_lowerblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
        std::string self_energy_greater_lowerblk_path = base_path + "self_energy_greater_"+ std::to_string(batch) +"_lowerblk.bin";
        load_binary_matrix(self_energy_greater_lowerblk_path.c_str(), self_energy_greater_lowerblk, blocksize, off_diag_size);

        complex_h* lesser_inv_diagblk = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
        std::string lesser_diagblk_path = base_path + "lesser_"+ std::to_string(batch) +"_inv_diagblk.bin";
        load_binary_matrix(lesser_diagblk_path.c_str(), lesser_inv_diagblk, blocksize, matrix_size);

        complex_h* lesser_inv_upperblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
        std::string lesser_upperblk_path = base_path + "lesser_"+ std::to_string(batch) +"_inv_upperblk.bin";
        load_binary_matrix(lesser_upperblk_path.c_str(), lesser_inv_upperblk, blocksize, off_diag_size);

        complex_h* lesser_inv_lowerblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
        std::string lesser_lowerblk_path = base_path + "lesser_"+ std::to_string(batch) +"_inv_lowerblk.bin";
        load_binary_matrix(lesser_lowerblk_path.c_str(), lesser_inv_lowerblk, blocksize, off_diag_size);

        complex_h* greater_inv_diagblk = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
        std::string greater_diagblk_path = base_path + "greater_"+ std::to_string(batch) +"_inv_diagblk.bin";
        load_binary_matrix(greater_diagblk_path.c_str(), greater_inv_diagblk, blocksize, matrix_size);

        complex_h* greater_inv_upperblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
        std::string greater_upperblk_path = base_path + "greater_"+ std::to_string(batch) +"_inv_upperblk.bin";
        load_binary_matrix(greater_upperblk_path.c_str(), greater_inv_upperblk, blocksize, off_diag_size);

        complex_h* greater_inv_lowerblk = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
        std::string greater_lowerblk_path = base_path + "greater_"+ std::to_string(batch) +"_inv_lowerblk.bin";
        load_binary_matrix(greater_lowerblk_path.c_str(), greater_inv_lowerblk, blocksize, off_diag_size);


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
        complex_h* self_energy_lesser_lowerblk_h = NULL;
        complex_h* self_energy_greater_diagblk_h = NULL;
        complex_h* self_energy_greater_upperblk_h = NULL;
        complex_h* self_energy_greater_lowerblk_h = NULL;
        complex_h* lesser_inv_diagblk_ref = NULL;
        complex_h* lesser_inv_upperblk_ref = NULL;
        complex_h* lesser_inv_lowerblk_ref = NULL;
        complex_h* greater_inv_diagblk_ref = NULL;
        complex_h* greater_inv_upperblk_ref = NULL;
        complex_h* greater_inv_lowerblk_ref = NULL;
        cudaMallocHost((void**)&system_matrix_diagblk_h, blocksize * matrix_size * sizeof(complex_h));
        cudaMallocHost((void**)&system_matrix_upperblk_h, blocksize * off_diag_size * sizeof(complex_h));
        cudaMallocHost((void**)&system_matrix_lowerblk_h, blocksize * off_diag_size * sizeof(complex_h));
        cudaMallocHost((void**)&self_energy_lesser_diagblk_h, blocksize * matrix_size * sizeof(complex_h));
        cudaMallocHost((void**)&self_energy_lesser_upperblk_h, blocksize * off_diag_size * sizeof(complex_h));
        cudaMallocHost((void**)&self_energy_lesser_lowerblk_h, blocksize * off_diag_size * sizeof(complex_h));
        cudaMallocHost((void**)&self_energy_greater_diagblk_h, blocksize * matrix_size * sizeof(complex_h));
        cudaMallocHost((void**)&self_energy_greater_upperblk_h, blocksize * off_diag_size * sizeof(complex_h));
        cudaMallocHost((void**)&self_energy_greater_lowerblk_h, blocksize * off_diag_size * sizeof(complex_h));
        cudaMallocHost((void**)&lesser_inv_diagblk_ref, blocksize * matrix_size * sizeof(complex_h));
        cudaMallocHost((void**)&lesser_inv_upperblk_ref, blocksize * off_diag_size * sizeof(complex_h));
        cudaMallocHost((void**)&lesser_inv_lowerblk_ref, blocksize * off_diag_size * sizeof(complex_h));
        cudaMallocHost((void**)&greater_inv_diagblk_ref, blocksize * matrix_size * sizeof(complex_h));
        cudaMallocHost((void**)&greater_inv_upperblk_ref, blocksize * off_diag_size * sizeof(complex_h));
        cudaMallocHost((void**)&greater_inv_lowerblk_ref, blocksize * off_diag_size * sizeof(complex_h));

        for(unsigned int i = 0; i < blocksize * matrix_size; i++){
            // block index
            int k = i / (blocksize * blocksize);
            // index inside block
            int h = i % (blocksize * blocksize);
            // row inside block
            int m = h % blocksize;
            // col inside block
            int n = h / blocksize;
            system_matrix_diagblk_h[i] = system_matrix_diagblk[m*matrix_size + k*blocksize + n];
            self_energy_lesser_diagblk_h[i] = self_energy_lesser_diagblk[m*matrix_size + k*blocksize + n];
            self_energy_greater_diagblk_h[i] = self_energy_greater_diagblk[m*matrix_size + k*blocksize + n];
            lesser_inv_diagblk_ref[i] = lesser_inv_diagblk[m*matrix_size + k*blocksize + n];
            greater_inv_diagblk_ref[i] = greater_inv_diagblk[m*matrix_size + k*blocksize + n];
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
            system_matrix_upperblk_h[i] = system_matrix_upperblk[m*off_diag_size + k*blocksize + n];
            system_matrix_lowerblk_h[i] = system_matrix_lowerblk[m*off_diag_size + k*blocksize + n];
            self_energy_lesser_upperblk_h[i] = self_energy_lesser_upperblk[m*off_diag_size + k*blocksize + n];
            self_energy_lesser_lowerblk_h[i] = self_energy_lesser_lowerblk[m*off_diag_size + k*blocksize + n];
            self_energy_greater_upperblk_h[i] = self_energy_greater_upperblk[m*off_diag_size + k*blocksize + n];
            self_energy_greater_lowerblk_h[i] = self_energy_greater_lowerblk[m*off_diag_size + k*blocksize + n];
            lesser_inv_upperblk_ref[i] = lesser_inv_upperblk[m*off_diag_size + k*blocksize + n];
            lesser_inv_lowerblk_ref[i] = lesser_inv_lowerblk[m*off_diag_size + k*blocksize + n];
            greater_inv_upperblk_ref[i] = greater_inv_upperblk[m*off_diag_size + k*blocksize + n];
            greater_inv_lowerblk_ref[i] = greater_inv_lowerblk[m*off_diag_size + k*blocksize + n];
        }
        system_matrices_diagblk_h[batch] = system_matrix_diagblk_h;
        system_matrices_upperblk_h[batch] = system_matrix_upperblk_h;
        system_matrices_lowerblk_h[batch] = system_matrix_lowerblk_h;
        self_energy_matrices_lesser_diagblk_h[batch] = self_energy_lesser_diagblk_h;
        self_energy_matrices_lesser_upperblk_h[batch] = self_energy_lesser_upperblk_h;
        self_energy_matrices_lesser_lowerblk_h[batch] = self_energy_lesser_lowerblk_h;
        self_energy_matrices_greater_diagblk_h[batch] = self_energy_greater_diagblk_h;
        self_energy_matrices_greater_upperblk_h[batch] = self_energy_greater_upperblk_h;
        self_energy_matrices_greater_lowerblk_h[batch] = self_energy_greater_lowerblk_h;
        
        lesser_inv_matrices_diagblk_ref[batch] = lesser_inv_diagblk_ref;
        lesser_inv_matrices_upperblk_ref[batch] = lesser_inv_upperblk_ref;
        lesser_inv_matrices_lowerblk_ref[batch] = lesser_inv_lowerblk_ref;
        greater_inv_matrices_diagblk_ref[batch] = greater_inv_diagblk_ref;
        greater_inv_matrices_upperblk_ref[batch] = greater_inv_upperblk_ref;
        greater_inv_matrices_lowerblk_ref[batch] = greater_inv_lowerblk_ref;


        // allocate memory for the inv
        complex_h* lesser_inv_diagblk_h = NULL;
        complex_h* lesser_inv_upperblk_h = NULL;
        complex_h* lesser_inv_lowerblk_h = NULL;
        complex_h* greater_inv_diagblk_h = NULL;
        complex_h* greater_inv_upperblk_h = NULL;
        complex_h* greater_inv_lowerblk_h = NULL;

        cudaMallocHost((void**)&lesser_inv_diagblk_h, blocksize * matrix_size * sizeof(complex_h));
        cudaMallocHost((void**)&lesser_inv_upperblk_h, blocksize * off_diag_size * sizeof(complex_h));
        cudaMallocHost((void**)&lesser_inv_lowerblk_h, blocksize * off_diag_size * sizeof(complex_h));
        cudaMallocHost((void**)&greater_inv_diagblk_h, blocksize * matrix_size * sizeof(complex_h));
        cudaMallocHost((void**)&greater_inv_upperblk_h, blocksize * off_diag_size * sizeof(complex_h));
        cudaMallocHost((void**)&greater_inv_lowerblk_h, blocksize * off_diag_size * sizeof(complex_h));

        rgf_lesser_greater(blocksize, matrix_size,
                                system_matrix_diagblk_h,
                                system_matrix_upperblk_h,
                                system_matrix_lowerblk_h,
                                self_energy_lesser_diagblk_h,
                                self_energy_lesser_upperblk_h,
                                self_energy_lesser_lowerblk_h,
                                self_energy_greater_diagblk_h,
                                self_energy_greater_upperblk_h,
                                self_energy_greater_lowerblk_h,
                                lesser_inv_diagblk_h,
                                lesser_inv_upperblk_h,
                                lesser_inv_lowerblk_h,
                                greater_inv_diagblk_h,
                                greater_inv_upperblk_h,
                                greater_inv_lowerblk_h);

        std::cout << "batch " << batch << std::endl;

        // ----- RESULT CHECKING SECTION -----


        // // print last block of inverted matrix
        // for(unsigned int i = blocksize *(matrix_size-blocksize); i < blocksize * matrix_size; i++){
        //     std::cout << "batch_lesser_inv_diagblk_h[" << i << "] = " << batch_lesser_inv_diagblk_h[i] << std::endl;
        //     std::cout << "inv_diagblk_ref[" << i << "] = " << inv_diagblk_ref[i] << std::endl;
        // }

        double lesser_norm_diagblk = 0.0;
        double lesser_norm_upperblk = 0.0;
        double lesser_norm_lowerblk = 0.0;
        double lesser_diff_diagblk = 0.0;
        double lesser_diff_upperblk = 0.0;
        double lesser_diff_lowerblk = 0.0;

        double greater_norm_diagblk = 0.0;
        double greater_norm_upperblk = 0.0;
        double greater_norm_lowerblk = 0.0;
        double greater_diff_diagblk = 0.0;
        double greater_diff_upperblk = 0.0;
        double greater_diff_lowerblk = 0.0;

        for(unsigned int i = 0; i < blocksize * matrix_size; i++){
            lesser_norm_diagblk += std::abs(lesser_inv_diagblk_ref[i]);
            lesser_diff_diagblk += std::abs(lesser_inv_diagblk_h[i] - lesser_inv_diagblk_ref[i]);
            greater_norm_diagblk += std::abs(greater_inv_diagblk_ref[i]);
            greater_diff_diagblk += std::abs(greater_inv_diagblk_h[i] - greater_inv_diagblk_ref[i]);
        }
        for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
            lesser_norm_upperblk += std::abs(lesser_inv_upperblk_ref[i]);
            lesser_norm_lowerblk += std::abs(lesser_inv_lowerblk_ref[i]);
            lesser_diff_upperblk += std::abs(lesser_inv_upperblk_h[i] - lesser_inv_upperblk_ref[i]);
            lesser_diff_lowerblk += std::abs(lesser_inv_lowerblk_h[i] - lesser_inv_lowerblk_ref[i]);
            greater_norm_upperblk += std::abs(greater_inv_upperblk_ref[i]);
            greater_norm_lowerblk += std::abs(greater_inv_lowerblk_ref[i]);
            greater_diff_upperblk += std::abs(greater_inv_upperblk_h[i] - greater_inv_upperblk_ref[i]);
            greater_diff_lowerblk += std::abs(greater_inv_lowerblk_h[i] - greater_inv_lowerblk_ref[i]);
        }
        double eps = 1e-9;
        if(lesser_diff_diagblk/lesser_norm_diagblk > eps){
            printf("Error: lesser_diff_diagblk/lesser_norm_diagblk = %e\n", lesser_diff_diagblk/lesser_norm_diagblk);
        }
        else{
            printf("lesser_diff_diagblk/lesser_norm_diagblk = %e\n", lesser_diff_diagblk/lesser_norm_diagblk);
        }
        if(lesser_diff_upperblk/lesser_norm_upperblk > eps){
            printf("Error: lesser_diff_upperblk/lesser_norm_upperblk = %e\n", lesser_diff_upperblk/lesser_norm_upperblk);
        }
        else{
            printf("lesser_diff_upperblk/lesser_norm_upperblk = %e\n", lesser_diff_upperblk/lesser_norm_upperblk);
        }
        if(lesser_diff_lowerblk/lesser_norm_lowerblk > eps){
            printf("Error: lesser_diff_lowerblk/lesser_norm_lowerblk = %e\n", lesser_diff_lowerblk/lesser_norm_lowerblk);
        }
        else{
            printf("lesser_diff_lowerblk/lesser_norm_lowerblk = %e\n", lesser_diff_lowerblk/lesser_norm_lowerblk);
        }
        if(greater_diff_diagblk/greater_norm_diagblk > eps){
            printf("Error: greater_diff_diagblk/greater_norm_diagblk = %e\n", greater_diff_diagblk/greater_norm_diagblk);
        }
        else{
            printf("greater_diff_diagblk/greater_norm_diagblk = %e\n", greater_diff_diagblk/greater_norm_diagblk);
        }
        if(greater_diff_upperblk/greater_norm_upperblk > eps){
            printf("Error: greater_diff_upperblk/greater_norm_upperblk = %e\n", greater_diff_upperblk/greater_norm_upperblk);
        }
        else{
            printf("greater_diff_upperblk/greater_norm_upperblk = %e\n", greater_diff_upperblk/greater_norm_upperblk);
        }
        if(greater_diff_lowerblk/greater_norm_lowerblk > eps){
            printf("Error: greater_diff_lowerblk/greater_norm_lowerblk = %e\n", greater_diff_lowerblk/greater_norm_lowerblk);
        }
        else{
            printf("greater_diff_lowerblk/greater_norm_lowerblk = %e\n", greater_diff_lowerblk/greater_norm_lowerblk);
        }



        cudaFreeHost(lesser_inv_diagblk_h);
        cudaFreeHost(lesser_inv_upperblk_h);
        cudaFreeHost(lesser_inv_lowerblk_h);
        cudaFreeHost(greater_inv_diagblk_h);
        cudaFreeHost(greater_inv_upperblk_h);
        cudaFreeHost(greater_inv_lowerblk_h);

        // free non contiguous memory
        free(system_matrix_diagblk);
        free(system_matrix_upperblk);
        free(system_matrix_lowerblk);
        free(self_energy_lesser_diagblk);
        free(self_energy_lesser_upperblk);
        free(self_energy_lesser_lowerblk);
        free(self_energy_greater_diagblk);
        free(self_energy_greater_upperblk);
        free(self_energy_greater_lowerblk);
        free(lesser_inv_diagblk);
        free(lesser_inv_upperblk);
        free(lesser_inv_lowerblk);
        free(greater_inv_diagblk);
        free(greater_inv_upperblk);
        free(greater_inv_lowerblk);
    }

    // free contiguous memory
    for(unsigned int batch = 0; batch < batch_size; batch++){
        cudaFreeHost(system_matrices_diagblk_h[batch]);
        cudaFreeHost(system_matrices_upperblk_h[batch]);
        cudaFreeHost(system_matrices_lowerblk_h[batch]);
        cudaFreeHost(lesser_inv_matrices_diagblk_ref[batch]);
        cudaFreeHost(lesser_inv_matrices_upperblk_ref[batch]);
        cudaFreeHost(lesser_inv_matrices_lowerblk_ref[batch]);
        cudaFreeHost(greater_inv_matrices_diagblk_ref[batch]);
        cudaFreeHost(greater_inv_matrices_upperblk_ref[batch]);
        cudaFreeHost(greater_inv_matrices_lowerblk_ref[batch]);
    }
    return 0;
}








