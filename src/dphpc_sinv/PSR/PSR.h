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

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

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

void createblockMatrixType(MPI_Datatype* blockMatrixType, int stride, int blocksize);

void create_subblock_Type(MPI_Datatype* subblockType, int stride, int blocksize, int rowblocks);
void create_resized_subblock_Type(MPI_Datatype* subblockType_resized, MPI_Datatype subblockType, int stride, int blocksize, int rowBlocks);

void create_ul2_redschur_blockpattern_Type(MPI_Datatype* blockPatternType, MPI_Datatype subblockType, int blocksize, int stride, int partition_blocksize);

void create_br2_redschur_blockpattern_Type(MPI_Datatype* blockPatternType, MPI_Datatype subblockType, int blocksize, int stride, int partition_blocksize);

void create_central_redschur_blockpattern_Type(MPI_Datatype* blockPatternType, MPI_Datatype subblockType, MPI_Datatype subblockType_2, int blocksize, int stride, int partition_blocksize);