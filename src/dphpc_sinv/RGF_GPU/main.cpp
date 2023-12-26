#include "lesser_greater_retarded.h"
#include "batched_lesser_greater_retarded.h"
#include "for_lesser_greater_retarded.h"

#include <omp.h>
#include <fstream>

#define cudaErrchk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      std::printf("CUDAassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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
    double eps = 1e-7;
    int n_tests = n_tests;
    for(int k = 0; k < 1; k++){
        for(unsigned int batch = 0; batch < batch_size; batch++){

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
                retarded_inv_diagblk_ref[i] = retarded_inv_diagblk[m*matrix_size + k*blocksize + n];
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
                self_energy_greater_upperblk_h[i] = self_energy_greater_upperblk[m*off_diag_size + k*blocksize + n];
                lesser_inv_upperblk_ref[i] = lesser_inv_upperblk[m*off_diag_size + k*blocksize + n];
                greater_inv_upperblk_ref[i] = greater_inv_upperblk[m*off_diag_size + k*blocksize + n];
                retarded_inv_upperblk_ref[i] = retarded_inv_upperblk[m*off_diag_size + k*blocksize + n];
                retarded_inv_lowerblk_ref[i] = retarded_inv_lowerblk[m*off_diag_size + k*blocksize + n];
            }
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

            std::cout << "batch " << batch << std::endl;


            double lesser_norm_diagblk = 0.0;
            double lesser_norm_upperblk = 0.0;
            double lesser_diff_diagblk = 0.0;
            double lesser_diff_upperblk = 0.0;

            double greater_norm_diagblk = 0.0;
            double greater_norm_upperblk = 0.0;
            double greater_diff_diagblk = 0.0;
            double greater_diff_upperblk = 0.0;

            double retarded_norm_diagblk = 0.0;
            double retarded_norm_upperblk = 0.0;
            double retarded_norm_lowerblk = 0.0;
            double retarded_diff_diagblk = 0.0;
            double retarded_diff_upperblk = 0.0;
            double retarded_diff_lowerblk = 0.0;


            for(unsigned int i = 0; i < blocksize * matrix_size; i++){
                lesser_norm_diagblk += std::abs(lesser_inv_diagblk_ref[i])*std::abs(lesser_inv_diagblk_ref[i]);
                lesser_diff_diagblk += std::abs(lesser_inv_diagblk_h[i] - lesser_inv_diagblk_ref[i]) * std::abs(lesser_inv_diagblk_h[i] - lesser_inv_diagblk_ref[i]);
                greater_norm_diagblk += std::abs(greater_inv_diagblk_ref[i]) * std::abs(greater_inv_diagblk_ref[i]);
                greater_diff_diagblk += std::abs(greater_inv_diagblk_h[i] - greater_inv_diagblk_ref[i]) * std::abs(greater_inv_diagblk_h[i] - greater_inv_diagblk_ref[i]);
                retarded_norm_diagblk += std::abs(retarded_inv_diagblk_ref[i]) * std::abs(retarded_inv_diagblk_ref[i]);
                retarded_diff_diagblk += std::abs(retarded_inv_diagblk_h[i] - retarded_inv_diagblk_ref[i]) * std::abs(retarded_inv_diagblk_h[i] - retarded_inv_diagblk_ref[i]);
            }
            lesser_norm_diagblk = std::sqrt(lesser_norm_diagblk);
            lesser_diff_diagblk = std::sqrt(lesser_diff_diagblk);
            greater_norm_diagblk = std::sqrt(greater_norm_diagblk);
            greater_diff_diagblk = std::sqrt(greater_diff_diagblk);
            retarded_norm_diagblk = std::sqrt(retarded_norm_diagblk);
            retarded_diff_diagblk = std::sqrt(retarded_diff_diagblk);
            for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
                lesser_norm_upperblk += std::abs(lesser_inv_upperblk_ref[i]) * std::abs(lesser_inv_upperblk_ref[i]);
                lesser_diff_upperblk += std::abs(lesser_inv_upperblk_h[i] - lesser_inv_upperblk_ref[i]) * std::abs(lesser_inv_upperblk_h[i] - lesser_inv_upperblk_ref[i]);
                greater_norm_upperblk += std::abs(greater_inv_upperblk_ref[i]) * std::abs(greater_inv_upperblk_ref[i]);
                greater_diff_upperblk += std::abs(greater_inv_upperblk_h[i] - greater_inv_upperblk_ref[i]) * std::abs(greater_inv_upperblk_h[i] - greater_inv_upperblk_ref[i]);
                retarded_norm_upperblk += std::abs(retarded_inv_upperblk_ref[i]) * std::abs(retarded_inv_upperblk_ref[i]);
                retarded_diff_upperblk += std::abs(retarded_inv_upperblk_h[i] - retarded_inv_upperblk_ref[i]) * std::abs(retarded_inv_upperblk_h[i] - retarded_inv_upperblk_ref[i]);
                retarded_norm_lowerblk += std::abs(retarded_inv_lowerblk_ref[i]) * std::abs(retarded_inv_lowerblk_ref[i]);
                retarded_diff_lowerblk += std::abs(retarded_inv_lowerblk_h[i] - retarded_inv_lowerblk_ref[i]) * std::abs(retarded_inv_lowerblk_h[i] - retarded_inv_lowerblk_ref[i]);
            }
            lesser_norm_upperblk = std::sqrt(lesser_norm_upperblk);
            lesser_diff_upperblk = std::sqrt(lesser_diff_upperblk);
            greater_norm_upperblk = std::sqrt(greater_norm_upperblk);
            greater_diff_upperblk = std::sqrt(greater_diff_upperblk);
            retarded_norm_upperblk = std::sqrt(retarded_norm_upperblk);
            retarded_diff_upperblk = std::sqrt(retarded_diff_upperblk);
            retarded_norm_lowerblk = std::sqrt(retarded_norm_lowerblk);
            retarded_diff_lowerblk = std::sqrt(retarded_diff_lowerblk);
            if(lesser_diff_diagblk/lesser_norm_diagblk > eps || lesser_diff_upperblk/lesser_norm_upperblk > eps || greater_diff_diagblk/greater_norm_diagblk > eps || greater_diff_upperblk/greater_norm_upperblk > eps || retarded_diff_diagblk/retarded_norm_diagblk > eps || retarded_diff_upperblk/retarded_norm_upperblk > eps || retarded_diff_lowerblk/retarded_norm_lowerblk > eps){
                printf("FAILED\n");
                not_failed = false;
            }
            

            // printf("lesser_diff_diagblk/lesser_norm_diagblk = %e\n", lesser_diff_diagblk/lesser_norm_diagblk);
            // printf("lesser_diff_upperblk/lesser_norm_upperblk = %e\n", lesser_diff_upperblk/lesser_norm_upperblk);
            // printf("greater_diff_diagblk/greater_norm_diagblk = %e\n", greater_diff_diagblk/greater_norm_diagblk);
            // printf("greater_diff_upperblk/greater_norm_upperblk = %e\n", greater_diff_upperblk/greater_norm_upperblk);
            // printf("retarded_diff_diagblk/retarded_norm_diagblk = %e\n", retarded_diff_diagblk/retarded_norm_diagblk);
            // printf("retarded_diff_upperblk/retarded_norm_upperblk = %e\n", retarded_diff_upperblk/retarded_norm_upperblk);
            // printf("retarded_diff_lowerblk/retarded_norm_lowerblk = %e\n", retarded_diff_lowerblk/retarded_norm_lowerblk);
        

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

        int number_of_measurements = 1;
        double times_for[number_of_measurements];
        double times_batched[number_of_measurements];
        double time;

        for(int i = 0; i < number_of_measurements; i++){
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
            times_for[i] = time;
        }

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
        if(diff_lesser_diagblk/norm_lesser_diagblk > eps || diff_lesser_upperblk/norm_lesser_upperblk > eps || diff_greater_diagblk/norm_greater_diagblk > eps || diff_greater_upperblk/norm_greater_upperblk > eps || diff_retarded_diagblk/norm_retarded_diagblk > eps || diff_retarded_upperblk/norm_retarded_upperblk > eps || diff_retarded_lowerblk/norm_retarded_lowerblk > eps){
            printf("FAILED BATCHED\n");
            not_failed = false;
        }


        for(int i = 0; i < number_of_measurements; i++){
            time = -omp_get_wtime();
            rgf_lesser_greater_batched(blocksize, matrix_size, batch_size,
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
            times_batched[i] = time;
            
        }

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
        if(diff_lesser_diagblk/norm_lesser_diagblk > eps || diff_lesser_upperblk/norm_lesser_upperblk > eps || diff_greater_diagblk/norm_greater_diagblk > eps || diff_greater_upperblk/norm_greater_upperblk > eps || diff_retarded_diagblk/norm_retarded_diagblk > eps || diff_retarded_upperblk/norm_retarded_upperblk > eps || diff_retarded_lowerblk/norm_retarded_lowerblk > eps){
            printf("FAILED BATCHED\n");
            not_failed = false;
        }

        std::ofstream outputFile_times_for;
        std::string filename = "times_for_" + std::to_string(matrix_size) + "_"   + std::to_string(n_blocks) + "_"  + std::to_string(batch_size) + "_"+ ".txt";
        outputFile_times_for.open(filename);
        if(outputFile_times_for.is_open()){
            for(int i = 0; i < number_of_measurements; i++){
                outputFile_times_for << times_for[i] << std::endl;
            }
        }
        else{
            std::cout << "Unable to open file" << std::endl;
        }
        outputFile_times_for.close();

        std::ofstream outputFile_times_batched;
        filename = "times_batched_" + std::to_string(matrix_size) + "_"   + std::to_string(n_blocks) + "_"  + std::to_string(batch_size) + "_"+ ".txt";
        outputFile_times_batched.open(filename);
        if(outputFile_times_batched.is_open()){
            for(int i = 0; i < number_of_measurements; i++){
                outputFile_times_batched << times_batched[i] << std::endl;
            }
        }
        else{
            std::cout << "Unable to open file" << std::endl;
        }
        outputFile_times_batched.close();



        // printf("diff_lesser_diagblk/norm_lesser_diagblk = %e\n", std::sqrt(diff_lesser_diagblk/norm_lesser_diagblk));
        // printf("diff_lesser_upperblk/norm_lesser_upperblk = %e\n", std::sqrt(diff_lesser_upperblk/norm_lesser_upperblk));
        // printf("diff_greater_diagblk/norm_greater_diagblk = %e\n", std::sqrt(diff_greater_diagblk/norm_greater_diagblk));
        // printf("diff_greater_upperblk/norm_greater_upperblk = %e\n", std::sqrt(diff_greater_upperblk/norm_greater_upperblk));
        // printf("diff_retarded_diagblk/norm_retarded_diagblk = %e\n", std::sqrt(diff_retarded_diagblk/norm_retarded_diagblk));
        // printf("diff_retarded_upperblk/norm_retarded_upperblk = %e\n", std::sqrt(diff_retarded_upperblk/norm_retarded_upperblk));
        // printf("diff_retarded_lowerblk/norm_retarded_lowerblk = %e\n", std::sqrt(diff_retarded_lowerblk/norm_retarded_lowerblk));

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








