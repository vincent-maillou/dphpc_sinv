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

std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd, Eigen::MatrixXcd> reduce_schur_topleftcorner(
    Eigen::MatrixXd& A,
    int start_blockrow,
    int partition_blocksize,
    int blocksize
);
