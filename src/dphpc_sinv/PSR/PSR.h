#include <iostream>
#include <fstream>
#include <array>
#include <complex>
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

std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd> reduce_schur_central(
    Eigen::MatrixXcd& A,
    int start_blockrow,
    int partition_blocksize,
    int blocksize
);

void aggregate_reduced_system_locally(
    Eigen::MatrixXcd& A,
    Eigen::MatrixXcd** A_schur_processes,
    int partition_blocksize,
    int blocksize
);


void myFunction(Eigen::MatrixXcd& A);