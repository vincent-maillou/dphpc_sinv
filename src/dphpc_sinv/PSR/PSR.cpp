#include <iostream>
#include <fstream>
#include <complex>
#define MKL_Complex16 std::complex<double>
#include <mkl.h>
#include <mkl_lapacke.h>

#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>

#include <omp.h>
#include <mpi.h>
#include <cuda.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>



void load_matrix(
    std::string filename, 
    std::complex<double> *matrix, 
    int rows, 
    int cols)
{
    // Open the binary file for reading
    std::ifstream input(filename, std::ios::binary);
    if (input.is_open()) {
        // Read the binary data into the std::complex<double> array
        input.read(reinterpret_cast<char*>(matrix), sizeof(std::complex<double>) * rows * cols);

        // Check if the read operation was successful
        if (!input) {
            std::cerr << "Read operation failed or reached the end of the file." << std::endl;
        } 
        // Close the input file
        input.close();
    } else {
        std::cerr << "Failed to open the binary file for reading." << std::endl;
    }
}

int main(int argc, char *argv[]) {

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current MPI-process ID. O, 1, ...
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get the total number of processes

    std::cout << "Hello from process " << rank << " of " << size << std::endl;

    const int N = 64; // Change this to the desired size of your NxN matrix
    std::complex<double>* A = new std::complex<double>[N * N];
    std::complex<double>* A_inv = new std::complex<double>[N * N];
    std::string filename = "matrix_0_diagblk.bin";
    std::string filename_inv = "matrix_0_inverse_diagblk.bin";

    load_matrix("matrix_0_diagblk.bin", A, N, N);
    load_matrix("matrix_0_inverse_diagblk.bin", A_inv, N, N);

    // // Fill A with your matrix data
    // for (int i = 0; i < N * N; i++) {
    //     A[i] = i;
    // }
    int* ipiv = new int[N];
    
    int info;


    // Perform the LU factorization
    info = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, N, N, A, N, ipiv);

    if (info == 0) {
        // If factorization was successful, invert the matrix
        info = LAPACKE_zgetri(LAPACK_ROW_MAJOR, N, A, N, ipiv);
        if (info){
            std::cerr << "Matrix inversion failed." << std::endl;
        }
    } else {
        std::cerr << "LU factorization failed." << std::endl;
    }

    double error = 0.0;
    for (int i = 0; i < N * N; i++) {
        error += std::abs(A[i] - A_inv[i]);
    }
    std::cout << "Error: " << error << std::endl;

    load_matrix("matrix_0_diagblk.bin", A, N, N);
    Eigen::Map<Eigen::Matrix<std::complex<double>, N, N, Eigen::RowMajor>> eigenMatrix(A, N, N);
    auto invertedMatrix = eigenMatrix.inverse();
    auto invertedMatrix2 = eigenMatrix.inverse();

    double error_eigen = 0.0;
    for (int i = 0; i < N * N; i++) {
        error_eigen += std::abs(invertedMatrix(i) - A_inv[i]);
    }

    std::cout << "Error Eigen: " << error_eigen << std::endl;

    double error_eigen2 = 0.0;
    #pragma omp parallel for reduction(+:error_eigen2)
    for (int i = 0; i < N * N; i++) {
        error_eigen2 += std::abs(invertedMatrix2(i) - A_inv[i]);
    }

    std::cout << "Error Eigen 2: " << error_eigen << std::endl;

    delete[] A;
    delete[] A_inv;
    delete[] ipiv;

    MPI_Finalize();

    return 0;
}