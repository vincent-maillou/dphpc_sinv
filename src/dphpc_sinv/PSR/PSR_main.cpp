#include "PSR.h"

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

    const int N = 120; // Change this to the desired size of your NxN matrix
    const int blocksize = 8; // Change this to the desired blocksize
    // Memory allocation for each "process"
    std::complex<double>* A = new std::complex<double>[N * N];
    // std::complex<double>* A0 = new std::complex<double>[N * N];
    // std::complex<double>* A1 = new std::complex<double>[N * N];
    // std::complex<double>* A2 = new std::complex<double>[N * N];

    std::complex<double>* A_inv = new std::complex<double>[N * N];
    std::complex<double>* A_ref = new std::complex<double>[N * N];
    std::string filename = "matrix_0_diagblk.bin";
    std::string reference_filename = "/home/dleonard/Documents/forked_SINV/SINV/tests/psr_tests/saved_matrices/A_full.bin";
    std::string filename_inv = "matrix_0_inverse_diagblk.bin";

    load_matrix("matrix_0_diagblk.bin", A, N, N);
    // load_matrix("matrix_0_diagblk.bin", A0, N, N);
    // load_matrix("matrix_0_diagblk.bin", A1, N, N);
    // load_matrix("matrix_0_diagblk.bin", A2, N, N);
    load_matrix("/home/dleonard/Documents/forked_SINV/SINV/tests/psr_tests/saved_matrices/A_full.bin", A_ref, N, N);
    load_matrix("matrix_0_inverse_diagblk.bin", A_inv, N, N);

    // Test cases schur reduction

    // // full matrix from different path
    Eigen::MatrixXcd eigenA_read_in = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(A, N, N);
    Eigen::MatrixXcd eigenA_ref = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(A_ref, N, N);

    // top left corner schur reduction
    std::complex<double>* A_red_top = new std::complex<double>[N * N];
    load_matrix("/home/dleonard/Documents/forked_SINV/SINV/tests/psr_tests/saved_matrices/A_red_s_top_full.bin", A_red_top, N, N);
    Eigen::MatrixXcd eigenArt = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(A_red_top, N, N);

    // center schur reduction
    std::complex<double>* A_red_center = new std::complex<double>[N * N];
    load_matrix("/home/dleonard/Documents/forked_SINV/SINV/tests/psr_tests/saved_matrices/A_red_s_centre_full.bin", A_red_center, N, N);
    Eigen::MatrixXcd eigenArc = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(A_red_center, N, N);

    // bottom right corner schur reduction
    std::complex<double>* A_red_bot = new std::complex<double>[N * N];
    load_matrix("/home/dleonard/Documents/forked_SINV/SINV/tests/psr_tests/saved_matrices/A_red_s_bottom_full.bin", A_red_bot, N, N);
    Eigen::MatrixXcd eigenArb = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(A_red_bot, N, N);

    /* End of Test case schur reduction read-in*/

    int n_blocks = N / blocksize;
    int partitions = 3; // Change to number of MPI processes
    int partition_blocksize = n_blocks / partitions;

    Eigen::MatrixXcd** eigenA = new Eigen::MatrixXcd*[partitions];
    for(int i = 0; i < partitions; ++i) {
        eigenA[i] = new Eigen::MatrixXcd(N, N);
        *eigenA[i] = eigenA_read_in;
    }

    // Check if the matrices are the same
    if (*eigenA[0] == eigenA_ref && *eigenA[1] == eigenA_ref && *eigenA[2] == eigenA_ref) {
        std::cout << "A with orig. A are the same." << std::endl;
    } else {
        std::cout << "A with orig. A are different." << std::endl;
    }

    // Begin reduce_schur
    Eigen::MatrixXcd** G_matrices = new Eigen::MatrixXcd*[partitions];
    Eigen::MatrixXcd** L_matrices = new Eigen::MatrixXcd*[partitions];
    Eigen::MatrixXcd** U_matrices = new Eigen::MatrixXcd*[partitions];

    for (int i = 0; i < partitions; ++i) {
        int start_blockrow = i * partition_blocksize;
        
        G_matrices[i] = new Eigen::MatrixXcd(eigenA[i]->rows(), eigenA[i]->cols());
        L_matrices[i] = new Eigen::MatrixXcd(eigenA[i]->rows(), eigenA[i]->cols());
        U_matrices[i] = new Eigen::MatrixXcd(eigenA[i]->rows(), eigenA[i]->cols());

        G_matrices[i]->setZero();
        L_matrices[i]->setZero();
        U_matrices[i]->setZero();

        std::cout << "Process " << rank << " is reducing blockrows " << start_blockrow << " to " << start_blockrow + partition_blocksize - 1 << std::endl;
        

        if (i == 0){
            auto result = reduce_schur_topleftcorner(*eigenA[i], start_blockrow, partition_blocksize, blocksize);
            *L_matrices[i] = std::get<0>(result);
            *U_matrices[i] = std::get<1>(result);
        }

        if (i == 1){
            auto result = reduce_schur_central(*eigenA[i], start_blockrow, partition_blocksize, blocksize);
            *L_matrices[i] = std::get<0>(result);
            *U_matrices[i] = std::get<1>(result);
        }

        if (i == 2){
            auto result = reduce_schur_bottomrightcorner(*eigenA[i], start_blockrow, partition_blocksize, blocksize);
            *L_matrices[i] = std::get<0>(result);
            *U_matrices[i] = std::get<1>(result);
        }

    }
    // End reduce_schur

    //Define aggregated schur matrix on "process 0"
    Eigen::MatrixXcd A_schur = Eigen::MatrixXcd(blocksize*(partitions - 1) * 2, blocksize*(partitions - 1) * 2);
    A_schur.setZero();

    // Test case for aggregate schur
    std::complex<double>* A_schur_test = new std::complex<double>[2 * blocksize * (partitions - 1) * 2 * blocksize  * (partitions - 1)];
    load_matrix("/home/dleonard/Documents/forked_SINV/SINV/tests/psr_tests/saved_matrices/A_schur.bin", A_schur_test, 2 *  blocksize * (partitions - 1), 2 * blocksize * (partitions - 1));
    Eigen::MatrixXcd eigenA_schur_test = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(A_schur_test, 2 * blocksize * (partitions - 1), 2 * blocksize * (partitions - 1));

    aggregate_reduced_system_locally(A_schur, eigenA, (partitions - 1) * 2, partition_blocksize, blocksize, partitions);

    // Test case for aggregate schur
    std::complex<double>* G_schur_test = new std::complex<double>[2 * blocksize * (partitions - 1) * 2 * blocksize  * (partitions - 1)];
    load_matrix("/home/dleonard/Documents/forked_SINV/SINV/tests/psr_tests/saved_matrices/G_schur.bin", G_schur_test, 2 *  blocksize * (partitions - 1), 2 * blocksize * (partitions - 1));
    Eigen::MatrixXcd eigenG_schur_test = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(G_schur_test, 2 * blocksize * (partitions - 1), 2 * blocksize * (partitions - 1));

    auto G_schur = A_schur.inverse();

        /* To-Do: Implement write_back of the inverted Schur Blocks in a serial code by 
    filling the function "writeback_inverted_system_locally" defined in the header file. In the end G_matrices should contain the correct inverted blocks.

    Reference Python Implementation: sendback_inverted_reduced_system and receiveback_inverted_reduced_system
    
    To-Do: Generate Testcase for G_matrices from the reference Implementation.*/

    // Check if the matrices are the same
    if (eigenA[0]->isApprox(eigenArt)) {
        std::cout << "Top Left Schur Matrix are the same." << std::endl;
    } else {
        std::cout << "Top Left Schur Matrix are different." << std::endl;
    }

    // Check if the matrices are the same
    if (eigenA[1]->isApprox(eigenArc)) {
        std::cout << "Center Schur Matrix are the same." << std::endl;
    } else {
        std::cout << "Center Schur Matrix are different." << std::endl;
    }

    // Check if the matrices are the same
    if (eigenA[2]->isApprox(eigenArb)) {
        std::cout << "Bottom Right Schur Matrix are the same." << std::endl;
    } else {
        std::cout << "Bottom Right Schur Matrix are different." << std::endl;
    }

    //Check if reduced schur matrices are the same
    if (A_schur.isApprox(eigenA_schur_test)) {
        std::cout << "A_schur are the same." << std::endl;
    } else {
        std::cout << "A_schur are different." << std::endl;
    }

    //Check if reduced schur matrices are the same
    if (G_schur.isApprox(eigenG_schur_test)) {
        std::cout << "G_schur are the same." << std::endl;
    } else {
        std::cout << "G_schur are different." << std::endl;
    }


    delete[] A;
    delete[] A_inv;
    delete[] A_ref;
    delete[] A_red_top;
    delete[] A_red_center;
    delete[] A_red_bot;

    for(int i = 0; i < partitions; ++i) {
        delete eigenA[i];
        delete G_matrices[i];
    }

    MPI_Finalize();

    return 0;
}