#include <iostream>
#include <fstream>
#include <array>
#include <complex>
#include <string>
#define MKL_Complex16 std::complex<double>
#include <mkl.h>
#include <mkl_lapacke.h>

#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include <Eigen/PardisoSupport>

#include <omp.h>
#include <mpi.h>
#include <cuda.h>
#include "openacc.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
// Additional cuda include's
#include <cuComplex.h>
#include <cuda/std/complex>
#include <cublas_v2.h>
#include <cusolverDn.h>

using complex_h = std::complex<double>;
using complex_d = cuDoubleComplex;

// Start of additions for cuda impl
#define cudaErrchk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      std::printf("CUDAassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define cusolverErrchk(ans) { cusolverAssert((ans), __FILE__, __LINE__); }
inline void cusolverAssert(cusolverStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUSOLVER_STATUS_SUCCESS) 
   {
        //Did not find a counter part to cudaGetErrorString in cusolver
        std::printf("CUSOLVERassert: %s %d\n", file, line);
        if (abort) exit(code);
   }
}


#define cublasErrchk(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUBLAS_STATUS_SUCCESS) 
   {
        //Did not find a counter part to cudaGetErrorString in cublas
        std::printf("CUBLASassert: %s %d\n", file, line);
        if (abort) exit(code);
   }
}

void reduce_schur_sequentially(Eigen::MatrixXcd** eigenA,
                             Eigen::MatrixXcd** G_matrices,
                             Eigen::MatrixXcd** L_matrices,
                             Eigen::MatrixXcd** U_matrices,
                             int partitions,
                             int partition_blocksize,
                             int blocksize,
                             int rank
);

void aggregate_Gblocks_tofinalinverse_sequentially(int partitions,
                                       int partition_blocksize,
                                       int blocksize,
                                       Eigen::MatrixXcd** G_matrices,
                                       Eigen::MatrixXcd& G_final
);


void produce_schur_sequentially(Eigen::MatrixXcd** eigenA,
                             Eigen::MatrixXcd** G_matrices,
                             Eigen::MatrixXcd** L_matrices,
                             Eigen::MatrixXcd** U_matrices,
                             int partitions,
                             int partition_blocksize,
                             int blocksize,
                             int rank
);

std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd> reduce_schur_topleftcorner(
    Eigen::MatrixXcd& A,
    int start_blockrow,
    int partition_blocksize,
    int blocksize
);

std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd> reduce_schur_bottomrightcorner(
    Eigen::MatrixXcd& A,
    int start_blockrow,
    int partition_blocksize,
    int blocksize
);

std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd> reduce_schur_bottomrightcorner_2(
    Eigen::MatrixXcd& A,
    int partition_blocksize,
    int blocksize
);

std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd> reduce_schur_central(
    Eigen::MatrixXcd& A,
    int start_blockrow,
    int partition_blocksize,
    int blocksize
);

std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd> reduce_schur_central_2(
    Eigen::MatrixXcd& A,
    int partition_blocksize,
    int blocksize
);

void aggregate_reduced_system_locally(
    Eigen::MatrixXcd& A,
    Eigen::MatrixXcd** A_schur_processes,
    int nblocks_schur_system,
    int partition_blocksize,
    int blocksize,
    int partitions
);

void writeback_inverted_system_locally(
    Eigen::MatrixXcd G,
    Eigen::MatrixXcd** G_schur_processes,
    int nblocks_schur_system,
    int partition_blocksize,
    int blocksize,
    int partitions
);

void produceSchurTopLeftCorner(Eigen::MatrixXcd A,
                               Eigen::MatrixXcd L,
                               Eigen::MatrixXcd U,
                               Eigen::MatrixXcd& G,
                               int start_blockrow,
                               int partition_blocksize,
                               int blocksize
                               
);

void produceSchurBottomRightCorner(Eigen::MatrixXcd A,
                                   Eigen::MatrixXcd L,
                                   Eigen::MatrixXcd U,
                                   Eigen::MatrixXcd& G,
                                   int start_blockrow,
                                   int partition_blocksize,
                                   int blocksize
);   

void produceSchurBottomRightCorner_2(Eigen::MatrixXcd A,
                                   Eigen::MatrixXcd L,
                                   Eigen::MatrixXcd U,
                                   Eigen::MatrixXcd& G,
                                   int partition_blocksize,
                                   int blocksize
);   


void produceSchurCentral(Eigen::MatrixXcd A,
                                   Eigen::MatrixXcd L,
                                   Eigen::MatrixXcd U,
                                   Eigen::MatrixXcd& G,
                                   int start_blockrow,
                                   int partition_blocksize,
                                   int blocksize
);   

void produceSchurCentral_2(Eigen::MatrixXcd A,
                                   Eigen::MatrixXcd L,
                                   Eigen::MatrixXcd U,
                                   Eigen::MatrixXcd& G,
                                   int partition_blocksize,
                                   int blocksize
);   


void fill_buffer(Eigen::MatrixXcd& inMatrix,
		 Eigen::MatrixXcd** eigenA, 
		 int partition_blocksize, 
		 int blocksize, 
		 int rank, 
		 int partitions);

void fill_buffer_2(Eigen::MatrixXcd& inMatrix,
		 Eigen::MatrixXcd eigenA, 
		 int partition_blocksize, 
		 int blocksize, 
		 int rank, 
		 int partitions);


void fill_reduced_schur_matrix(Eigen::MatrixXcd& A_schur, 
			       double* comm_buf, 
			       int in_buf_size, 
			       int blocksize, 
			       int partitions);

void fill_reduced_schur_matrix_cd(Eigen::MatrixXcd& A_schur, 
                   std::complex<double>* comm_buf_cd,
			       int in_buf_size, 
			       int blocksize, 
			       int partitions,
                   int rank);

                   

void myFunction(Eigen::MatrixXcd& A);

void load_matrix(
    std::string filename, 
    std::complex<double> *matrix, 
    int rows, 
    int cols);


void read_central_testblock(Eigen::MatrixXcd& A,
                             int N,
                             int rank,
                             std::string filename
);

void read_central_testblocks(Eigen::MatrixXcd** A,
                             int N,
                             int num_central_partitions,
                             std::string filename
);


void compareSINV_referenceInverse_byblock(int n_blocks,
                                     int blocksize,
                                     Eigen::MatrixXcd G_final,
                                     Eigen::MatrixXcd full_inverse,
                                     int rank
);

void compareSINV_referenceInverse_localprodG_byblock(int partitions,
                                     int blocksize,
                                     int partition_blocksize,
                                     Eigen::MatrixXcd G_local,
                                     Eigen::MatrixXcd full_inverse,
                                     int rank
);


Eigen::MatrixXcd psr_seqsolve_fulltest(const std::string test_folder,
                             int N,
                             int num_central_partitions,
                             int blocksize,
                             int n_blocks,
                             int partitions,
                             int partition_blocksize,
                             int rank,
                             int n_blocks_schursystem,
                             Eigen::MatrixXcd& eigenA_read_in
); 

Eigen::MatrixXcd psr_solve_fulltest(const std::string test_folder,
                             int N,
                             int num_central_partitions,
                             int blocksize,
                             int n_blocks,
                             int partitions,
                             int partition_blocksize,
                             int rank,
                             int n_blocks_schursystem,
                             Eigen::MatrixXcd& eigenA_read_in
); 

Eigen::MatrixXcd psr_seqsolve(int N,
                             int blocksize,
                             int n_blocks,
                             int partitions,
                             int partition_blocksize,
                             int rank,
                             int n_blocks_schursystem,
                             Eigen::MatrixXcd& eigenA_read_in,
                             bool compare_reference
);

Eigen::MatrixXcd psr_solve(int N,
                             int blocksize,
                             int n_blocks,
                             int partitions,
                             int partition_blocksize,
                             int rank,
                             int n_blocks_schursystem,
                             Eigen::MatrixXcd& eigenA_read_in,
                             bool compare_reference
);

Eigen::MatrixXcd psr_solve_customMPI(int N,
                             int blocksize,
                             int n_blocks,
                             int partitions,
                             int partition_blocksize,
                             int rank,
                             int n_blocks_schursystem,
                             Eigen::MatrixXcd& eigenA_read_in,
                             bool compare_reference
);

Eigen::MatrixXcd psr_solve_customMPI_gpu(int N,
                             int blocksize,
                             int n_blocks,
                             int partitions,
                             int partition_blocksize,
                             int rank,
                             int n_blocks_schursystem,
                             Eigen::MatrixXcd& eigenA_read_in,
                             bool compare_reference
);

void createblockMatrixType(MPI_Datatype* blockMatrixType, int stride, int blocksize);

void create_subblock_Type(MPI_Datatype* subblockType, int stride, int blocksize, int rowblocks);
void create_resized_subblock_Type(MPI_Datatype* subblockType_resized, MPI_Datatype subblockType, int stride, int blocksize, int rowBlocks);

void create_ul2_redschur_blockpattern_Type(MPI_Datatype* blockPatternType, MPI_Datatype subblockType, int blocksize, int stride, int partition_blocksize);

void create_br2_redschur_blockpattern_Type(MPI_Datatype* blockPatternType, MPI_Datatype subblockType, int blocksize, int stride, int partition_blocksize);

void create_central_redschur_blockpattern_Type(MPI_Datatype* blockPatternType, MPI_Datatype subblockType, MPI_Datatype subblockType_2, int blocksize, int stride, int partition_blocksize);


void create_identity_GPU(cuDoubleComplex* I, int matrix_size);

void extract_subblock_from_GPU(cuDoubleComplex* subblock, cuDoubleComplex* GPU_matrix, int blocksize, int stride, int rowBlock, int colBlock);
void copy_rowblocks_buffer2GPU(cuDoubleComplex* GPU_matrix, cuDoubleComplex* CPU_buffer, int blocksize, int stride, int rowBlocks, int rowBlock, int colBlock, int buffBlock);
void copy_rowblocks_GPU2buffer(cuDoubleComplex* GPU_matrix, cuDoubleComplex* CPU_buffer, int blocksize, int stride, int rowBlocks, int rowBlock, int colBlock, int buffBlock);
void copy_rowblocks_GPU2GPU(cuDoubleComplex* GPU_matrix1, cuDoubleComplex* GPU_matrix2, int blocksize, int stride1, int stride2, int rowBlocks, int rowBlock1, int colBlock1, int rowBlock2, int colBlock2);
void invert_GPU_matrix(cuDoubleComplex* GPU_matrix, cuDoubleComplex* I, int blocksize, cusolverDnHandle_t cusolverH, cuDoubleComplex* d_work, int* info_d, int info_h, int* ipiv_d);
void invert_GPU_matrix_complete(cuDoubleComplex* GPU_matrix, cuDoubleComplex* A_schur_gpu, int blocksize, cusolverDnHandle_t cusolverH);
