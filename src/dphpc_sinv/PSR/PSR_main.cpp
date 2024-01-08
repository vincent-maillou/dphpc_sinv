#include "PSR.h"

int main(int argc, char *argv[]) {

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current MPI-process ID. O, 1, ...
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get the total number of processes

    std::cout << "Hello from process " << rank << " of " << size << std::endl;

    //std::string test_folder = "/home/dleonard/Documents/dphpc_sinv/src/dphpc_sinv/PSR/test_matrices/120_8_3/";
    std::string test_folder = "/home/dleonard/Documents/dphpc_sinv/src/dphpc_sinv/PSR/";

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
    bool check_result = true;

    // Memory allocation for each "process"
    std::complex<double>* A = new std::complex<double>[N * N];
    std::complex<double>* A_diagblk = new std::complex<double>[n_blocks * blocksize * blocksize];
    std::complex<double>* A_upperblk = new std::complex<double>[(n_blocks-1) * blocksize * blocksize];
    std::complex<double>* A_lowerblk = new std::complex<double>[(n_blocks-1) * blocksize * blocksize];

    load_matrix(test_folder + "A_full.bin", A, N, N);
    load_matrix(test_folder + "matrix_0_diagblk.bin", A_diagblk, blocksize, n_blocks * blocksize);
    load_matrix(test_folder + "matrix_0_upperblk.bin", A_upperblk, blocksize, (n_blocks-1) * blocksize);
    load_matrix(test_folder + "matrix_0_lowerblk.bin", A_lowerblk, blocksize, (n_blocks-1) * blocksize);

    Eigen::MatrixXcd eigenA_read_in = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(A, N, N);
    Eigen::MatrixXcd eigenA_diagblk = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(A_diagblk, blocksize, n_blocks * blocksize);
    Eigen::MatrixXcd eigenA_upperblk = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(A_upperblk, blocksize, (n_blocks-1) * blocksize);
    Eigen::MatrixXcd eigenA_lowerblk = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(A_lowerblk, blocksize, (n_blocks-1) * blocksize);

    // if (rank == 0){
    //     if (eigenA_read_in.block(0, 0, blocksize, blocksize).isApprox(eigenA_diagblk.block(0, 0,blocksize,blocksize))){
    //         std::cout << "Block 0 0 is equal" << std::endl;
    //     }
    //     else {
    //         std::cout << "Block 0 0 is not equal" << std::endl;
    //     }
    //     if (eigenA_read_in.block(0, blocksize, blocksize, blocksize).isApprox(eigenA_upperblk.block(0, 0,blocksize,blocksize))){
    //         std::cout << "Block 0 1 is equal" << std::endl;
    //     }
    //     else {
    //         std::cout << "Block 0 1 is not equal" << std::endl;
    //     }
    //     if (eigenA_read_in.block(blocksize, 0, blocksize, blocksize).isApprox(eigenA_lowerblk.block(0, 0,blocksize,blocksize))){
    //         std::cout << "Block 1 0 is equal" << std::endl;
    //     }
    //     else {
    //         std::cout << "Block 1 0 is not equal" << std::endl;
    //     }
    // }

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
        auto G_final = psr_solve_customMPI_gpu(N, blocksize, n_blocks, partitions, partition_blocksize, rank, n_blocks_schursystem, eigenA_read_in,
                                                 eigenA_diagblk, eigenA_upperblk, eigenA_lowerblk, check_result);
        // auto G_final = psr_solve_customMPI(N, blocksize, n_blocks, partitions, partition_blocksize, rank, n_blocks_schursystem, eigenA_read_in, \
        //                                    eigenA_diagblk, eigenA_upperblk, eigenA_lowerblk, check_result);
    }   

    // Synchonization
    MPI_Barrier(MPI_COMM_WORLD);

    check_result = false;
    int nruns = 3;

    for (int i = 0; i < nruns; i++) {
        if(rank == 0){
            std::cout << "Run: " << i << " ..starting";
        }
        auto G_final = psr_solve_customMPI_gpu(N, blocksize, n_blocks, partitions, partition_blocksize, rank, n_blocks_schursystem, eigenA_diagblk,
                                                 eigenA_diagblk, eigenA_upperblk, eigenA_lowerblk, check_result);
        // auto G_final = psr_solve_customMPI(N, blocksize, n_blocks, partitions, partition_blocksize, rank, n_blocks_schursystem, eigenA_diagblk, \
        //                                    eigenA_diagblk, eigenA_upperblk, eigenA_lowerblk, check_result);
        MPI_Barrier(MPI_COMM_WORLD);
    }


    // if (rank == 0) {
    //     std::cout << "Block: " << "0  0" << std::endl;
    //     std::cout << eigenA_read_in.block(0, 0, blocksize, blocksize) << std::endl;
    //     std::cout << "Block: " << "0  1" << std::endl;
    //     std::cout << eigenA_read_in.block(0, blocksize, blocksize, blocksize) << std::endl;
    //     std::cout << "Block: " << "1  1" << std::endl;
    //     std::cout << eigenA_read_in.block(blocksize, blocksize, blocksize, blocksize) << std::endl;
    //     std::cout << "Block: " << "2  1" << std::endl;
    //     std::cout << eigenA_read_in.block(2 * blocksize, blocksize, blocksize, blocksize) << std::endl;
    //     std::cout << "Block: " << "2  2" << std::endl;
    //     std::cout << eigenA_read_in.block(2 * blocksize, 2 * blocksize, blocksize, blocksize) << std::endl;
    //     std::cout << "Block: " << "2  3" << std::endl;
    //     std::cout << eigenA_read_in.block(2 * blocksize, 3 * blocksize, blocksize, blocksize) << std::endl;

    // }
    // MPI_Datatype subblockType;
    // create_subblock_Type(&subblockType, N, blocksize, 1);

    // MPI_Datatype subblockType_resized;
    // create_resized_subblock_Type(&subblockType_resized, subblockType, N , blocksize, 1);

    // MPI_Datatype subblockType_2;
    // create_subblock_Type(&subblockType_2, N, blocksize, 2);

    // MPI_Datatype subblockType_3;
    // create_subblock_Type(&subblockType_3, N, blocksize, 3);

    // MPI_Datatype blockPatternType;
    // create_ur2_block_pattern_Type(&blockPatternType, subblockType_2, subblockType, blocksize, N);

    // MPI_Datatype exampleType;
    // create_receive_example_block_pattern_Type(&exampleType, blockPatternType, subblockType_2, blocksize, N);

    // int *receivecounts = new int[2];
    // receivecounts[0] = 3;
    // receivecounts[1] = 3;

    // int *displacements = new int[2];
    // displacements[0] = 0;
    // displacements[1] = n_blocks * blocksize;

    // if (rank == 0) {
    //     Eigen::MatrixXcd A_recv = Eigen::MatrixXcd::Zero(N, N);
    //     MPI_Allgatherv(eigenA_read_in.data(), 1, blockPatternType, A_recv.data(), receivecounts, displacements, subblockType_resized, MPI_COMM_WORLD);
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     //MPI_Send(eigenA_read_in.data(), 1, exampleType, 1, 0, MPI_COMM_WORLD);
    //     // std::cout << "Block: " << "3  3" << std::endl;
    //     // std::cout << A_recv.block(3 * blocksize, 3 * blocksize, blocksize, blocksize) << std::endl;
    //     // std::cout << "Block: " << "3  2" << std::endl;
    //     // std::cout << A_recv.block(3 * blocksize, 2 * blocksize, blocksize, blocksize) << std::endl;
    // }

    // else if (rank == 1) {
    //     Eigen::MatrixXcd A_recv = Eigen::MatrixXcd::Zero(N, N);
    //     Eigen::MatrixXcd A_0 = Eigen::MatrixXcd::Zero(N, N);
    //     //MPI_Recv(A_recv.data(), 1, exampleType, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //     MPI_Allgatherv(eigenA_read_in.data(), 1, blockPatternType, A_recv.data(), receivecounts, displacements, subblockType_resized, MPI_COMM_WORLD);
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     std::cout << "Block: " << "0  0" << std::endl;
    //     std::cout << A_recv.block(0, 0, blocksize, blocksize) << std::endl;
    //     std::cout << "Block: " << "1  0" << std::endl;
    //     std::cout << A_recv.block(blocksize, 0, blocksize, blocksize) << std::endl;
    //     std::cout << "Block: " << "2 0 " << std::endl;
    //     std::cout << A_recv.block(2 * blocksize, 0, blocksize, blocksize) << std::endl;
    //     std::cout << "Block: " << "1  1" << std::endl;
    //     std::cout << A_recv.block(blocksize, blocksize, blocksize, blocksize) << std::endl;
    //     // std::cout << "Block: " << "2  1" << std::endl;
    //     // std::cout << A_recv.block(2 * blocksize, blocksize, blocksize, blocksize) << std::endl;
    //     // std::cout << "Block: " << "2  2" << std::endl;
    //     // std::cout << A_recv.block(2 * blocksize, 2 * blocksize, blocksize, blocksize) << std::endl;
    //     // std::cout << "Block: " << "2  3" << std::endl;
    //     // std::cout << A_recv.block(2 * blocksize, 3 * blocksize, blocksize, blocksize) << std::endl;
    //     // std::cout << "Block: " << "0  2" << std::endl;
    //     // std::cout << A_recv.block(0, 2 * blocksize, blocksize, blocksize) << std::endl;
    //     // std::cout << "Block: " << "4  4" << std::endl;
    //     // std::cout << A_recv.block(4 * blocksize, 4 * blocksize, blocksize, blocksize) << std::endl;
    //     std::cout << "Block: " << "2 2" << std::endl;
    //     std::cout << A_recv.block(2 * blocksize, 2 * blocksize, blocksize, blocksize) << std::endl;
    //     std::cout << "Block: " << "2  1" << std::endl;
    //     std::cout << A_recv.block(2 * blocksize, 1 * blocksize, blocksize, blocksize) << std::endl;
    //     std::cout << "Block: " << "2  3" << std::endl;
    //     std::cout << A_recv.block(2 * blocksize, 3 * blocksize, blocksize, blocksize) << std::endl;
    //     std::cout << "Block: " << "3  3" << std::endl;
    //     std::cout << A_recv.block(3 * blocksize, 3 * blocksize, blocksize, blocksize) << std::endl;
    //     std::cout << "Block: " << "3  2" << std::endl;
    //     std::cout << A_recv.block(3 * blocksize, 2 * blocksize, blocksize, blocksize) << std::endl;
    //     std::cout << "Block: " << "2 4 " << std::endl;
    //     std::cout << A_recv.block(2 * blocksize, 4 * blocksize, blocksize, blocksize) << std::endl;

    //     std::cout << "matrix norm: " << A_recv.block(0,0, 5*blocksize, 5*blocksize).norm() << std::endl;
    //     std::cout << "matrix norm full: " << A_recv.norm() << std::endl;
    // }

    delete[] A;
    // MPI_Type_free(&subblockType);
    // MPI_Type_free(&subblockType_resized);
    // MPI_Type_free(&subblockType_2);
    // MPI_Type_free(&subblockType_3);
    // MPI_Type_free(&blockPatternType);
    // MPI_Type_free(&exampleType);
    MPI_Finalize();

    return 0;
}
