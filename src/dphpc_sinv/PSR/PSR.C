#include <iostream>
#include <mkl.h>

int main() {
    int N = 2; // Change this to the desired size of your NxN matrix
    double* A = new double[N * N]; // Replace with your own matrix data

    // Fill A with your matrix data
    for (int i = 0; i < N * N; i++) {
        A[i] = i;
    }
    int* ipiv = new int[N];
    
    int info;

    // Perform the LU factorization
    info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, N, N, A, N, ipiv);

    if (info == 0) {
        // If factorization was successful, invert the matrix
        info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, N, A, N, ipiv);
        if (info == 0) {
            // Inversion was successful, and the inverted matrix is now in A
            std::cout << "Inverted Matrix:\n";
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    std::cout << A[i * N + j] << "\t";
                }
                std::cout << "\n";
            }
        } else {
            std::cerr << "Matrix inversion failed." << std::endl;
        }
    } else {
        std::cerr << "LU factorization failed." << std::endl;
    }

    delete[] A;
    delete[] ipiv;

    return 0;
}