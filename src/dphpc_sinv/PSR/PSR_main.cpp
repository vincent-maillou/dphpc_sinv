#include "PSR.h"

int main(int argc, char *argv[]) {

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current MPI-process ID. O, 1, ...
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get the total number of processes

    std::cout << "Hello from process " << rank << " of " << size << std::endl;

    std::string test_folder = "/home/dleonard/Documents/dphpc_sinv/src/dphpc_sinv/PSR/test_matrices/120_8_3/";

    const int N = 120; // Change this to the desired size of your NxN matrix
    const int blocksize = 8; // Change this to the desired blocksize
    int partitions = size; // Change to number of MPI processes
    int num_central_partitions = partitions - 2;

    // Partition Parameters
    int n_blocks = N / blocksize;
    int partition_blocksize = n_blocks / partitions;

    int n_blocks_schursystem = (partitions - 1) * 2;
    // End of Partition Parameters

    bool FullSeqTest = false;

    // Memory allocation for each "process"
    std::complex<double>* A = new std::complex<double>[N * N];

    load_matrix(test_folder + "A_full.bin", A, N, N);


    Eigen::MatrixXcd eigenA_read_in = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(A, N, N);

    if (FullSeqTest) {
        auto G_final = psr_seqsolve_fulltest(test_folder,
                                 N,
                                 num_central_partitions,
                                 blocksize,
                                 n_blocks,
                                 partitions,
                                 partition_blocksize,
                                 rank,
                                 n_blocks_schursystem,
                                 eigenA_read_in
        );
    }
    else {
        auto G_final = psr_seqsolve(N, blocksize, n_blocks, partitions, partition_blocksize, rank, n_blocks_schursystem, eigenA_read_in, true);
    }   

    delete[] A;

    MPI_Finalize();

    return 0;
}