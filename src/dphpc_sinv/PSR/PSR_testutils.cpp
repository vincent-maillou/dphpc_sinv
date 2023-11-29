#include "PSR.h"


void read_central_testblock(Eigen::MatrixXcd& A,
                             int N,
                             int rank,
                             std::string filename
){

    //std::string filename_i = filename + std::to_string(rank);
    std::string filename_i = filename + std::to_string(rank) + ".bin";
    std::complex<double>* temp = new std::complex<double>[N * N];
    load_matrix(filename_i, temp, N, N);
    A = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(temp, N, N);
    delete[] temp;


}

void read_central_testblocks(Eigen::MatrixXcd** A,
                             int N,
                             int num_central_partitions,
                             std::string filename
){

    for (int i = 0; i < num_central_partitions; i++) {
        std::string filename_i = filename + std::to_string(i + 1) + ".bin";
        std::complex<double>* temp = new std::complex<double>[N * N];
        A[i] = new Eigen::MatrixXcd(N, N);
        load_matrix(filename_i, temp, N, N);
        *A[i] = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(temp, N, N);
        delete[] temp;
    }

}


void compareSINV_referenceInverse_byblock(int n_blocks,
                                     int blocksize,
                                     Eigen::MatrixXcd G_final,
                                     Eigen::MatrixXcd full_inverse,
                                     int rank
){
    if (rank == 0) {
        for (int i = 0; i < n_blocks; ++i) {
        if(G_final.block(i * blocksize, i * blocksize, blocksize, blocksize).isApprox(full_inverse.block(i * blocksize, i * blocksize, blocksize, blocksize))){
            std::cout << "Diagonal Block " << i << " is the same." << std::endl;
        } else {
            std::cout << "Diagonal Block " << i << " is different." << std::endl;
        }

        if(i < n_blocks - 1){
                if(G_final.block(i * blocksize, (i + 1) * blocksize, blocksize, blocksize).isApprox(full_inverse.block(i * blocksize, (i + 1) * blocksize, blocksize, blocksize))){
                    std::cout << "Off-Diagonal Block " << i << " is the same." << std::endl;
                } else {
                    std::cout << "Off-Diagonal Block " << i << " is different." << std::endl;
                }

                if(G_final.block((i + 1) * blocksize, i * blocksize, blocksize, blocksize).isApprox(full_inverse.block((i + 1) * blocksize, i * blocksize, blocksize, blocksize))){
                    std::cout << "Off-Diagonal Block " << i << " is the same." << std::endl;
                } else {
                    std::cout << "Off-Diagonal Block " << i << " is different." << std::endl;
                }
        }
        }
    }
}






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
){

    // top left corner schur reduction
    std::complex<double>* A_red_top = new std::complex<double>[N * N];
    load_matrix(test_folder + "A_red_s_top_full.bin", A_red_top, N, N);
    Eigen::MatrixXcd eigenArt = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(A_red_top, N, N);

    // center schur reduction
    Eigen::MatrixXcd** eigenArc = new Eigen::MatrixXcd*[partitions-2];
    read_central_testblocks(eigenArc, N, num_central_partitions, test_folder + "A_red_s_central_full");

    // bottom right corner schur reduction
    std::complex<double>* A_red_bot = new std::complex<double>[N * N];
    load_matrix(test_folder + "A_red_s_bottom_full.bin", A_red_bot, N, N);
    Eigen::MatrixXcd eigenArb = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(A_red_bot, N, N);

    /* End of Test case schur reduction read-in*/

    Eigen::MatrixXcd** eigenA = new Eigen::MatrixXcd*[partitions];
    for(int i = 0; i < partitions; ++i) {
        eigenA[i] = new Eigen::MatrixXcd(N, N);
        *eigenA[i] = eigenA_read_in;
    }

    // Referece inverse
    auto full_inverse = eigenA_read_in.inverse();

    // Begin reduce_schur
    Eigen::MatrixXcd** G_matrices = new Eigen::MatrixXcd*[partitions];
    Eigen::MatrixXcd** L_matrices = new Eigen::MatrixXcd*[partitions];
    Eigen::MatrixXcd** U_matrices = new Eigen::MatrixXcd*[partitions];

    reduce_schur_sequentially(eigenA, G_matrices, L_matrices, U_matrices,
                             partitions,
                             partition_blocksize,
                             blocksize,
                             rank);

    // End reduce_schur

    //Define aggregated schur matrix on "process 0"
    Eigen::MatrixXcd A_schur = Eigen::MatrixXcd(blocksize*n_blocks_schursystem, blocksize*n_blocks_schursystem);
    A_schur.setZero();

    // Test case for aggregate schur
    std::complex<double>* A_schur_test = new std::complex<double>[blocksize * n_blocks_schursystem * blocksize * n_blocks_schursystem];
    load_matrix(test_folder + "A_schur.bin", A_schur_test,  blocksize * n_blocks_schursystem, blocksize * n_blocks_schursystem);
    Eigen::MatrixXcd eigenA_schur_test = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(A_schur_test, blocksize * n_blocks_schursystem, blocksize * n_blocks_schursystem);

    //Start of changes for MPIALLGATHER
    unsigned long comm_buf_size = (blocksize * blocksize * partitions * 6) << 1; 
    double* comm_buf = new double[comm_buf_size];

    unsigned long in_buf_size = (blocksize * blocksize * 6) << 1;
    Eigen::MatrixXcd inMatrix = Eigen::MatrixXcd::Zero(blocksize, 6*blocksize);
    fill_buffer(inMatrix, eigenA, partition_blocksize, blocksize, rank, partitions);

    double* in_buf = (double*) inMatrix.data();
    MPI_Allgather(in_buf, in_buf_size, MPI_DOUBLE, comm_buf, in_buf_size, MPI_DOUBLE, MPI_COMM_WORLD);

    fill_reduced_schur_matrix(A_schur, comm_buf, in_buf_size, blocksize, partitions);

    delete[] comm_buf;
    //aggregate_reduced_system_locally(A_schur, eigenA, n_blocks_schursystem, partition_blocksize, blocksize, partitions);
    //End of changes

    // Test case for aggregate schur
    std::complex<double>* G_schur_test = new std::complex<double>[blocksize * n_blocks_schursystem * blocksize * n_blocks_schursystem];
    load_matrix(test_folder + "G_schur.bin", G_schur_test, blocksize * n_blocks_schursystem , blocksize * n_blocks_schursystem);
    Eigen::MatrixXcd eigenG_schur_test = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(G_schur_test, blocksize * n_blocks_schursystem, blocksize * n_blocks_schursystem);

    auto G_schur = A_schur.inverse();

    // top left corner before schur production
    std::complex<double>* G_red_top = new std::complex<double>[N * N];
    load_matrix(test_folder + "G_red_s_top_full.bin", G_red_top, N, N);
    Eigen::MatrixXcd eigenGrt = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(G_red_top, N, N);

    // center before schur production
    Eigen::MatrixXcd** eigenGrc = new Eigen::MatrixXcd*[partitions-2];
    read_central_testblocks(eigenGrc, N, num_central_partitions, test_folder + "G_red_s_central_full");

    // bottom right corner before schur production
    std::complex<double>* G_red_bot = new std::complex<double>[N * N];
    load_matrix(test_folder + "G_red_s_bottom_full.bin", G_red_bot, N, N);
    Eigen::MatrixXcd eigenGrb = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(G_red_bot, N, N);

    writeback_inverted_system_locally(G_schur, G_matrices, n_blocks_schursystem, partition_blocksize, blocksize, partitions);

    // top left corner after schur production
    std::complex<double>* G_prod_top = new std::complex<double>[N * N];
    load_matrix(test_folder + "G_prod_s_top_full.bin", G_prod_top, N, N);
    Eigen::MatrixXcd eigenGpt = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(G_prod_top, N, N);

    // center after schur production
    Eigen::MatrixXcd** eigenGpc = new Eigen::MatrixXcd*[partitions-2];
    read_central_testblocks(eigenGpc, N, num_central_partitions, test_folder + "G_prod_s_central_full");

    // bottom right cornerafter schur production
    std::complex<double>* G_prod_bot = new std::complex<double>[N * N];
    load_matrix(test_folder + "G_prod_s_bottom_full.bin", G_prod_bot, N, N);
    Eigen::MatrixXcd eigenGpb = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(G_prod_bot, N, N);

    produce_schur_sequentially(eigenA, G_matrices, L_matrices, U_matrices,
                             partitions,
                             partition_blocksize,
                             blocksize,
                             rank);


    Eigen::MatrixXcd G_final = Eigen::MatrixXcd(N, N);
    G_final.setZero();

    aggregate_Gblocks_tofinalinverse_sequentially(partitions,
                                       partition_blocksize,
                                       blocksize,
                                       G_matrices,
                                       G_final
    );


    compareSINV_referenceInverse_byblock(n_blocks,
                                     blocksize,
                                     G_final,
                                     full_inverse,
                                     0
    );

    // Check if the matrices are the same
    if (eigenA[0]->isApprox(eigenArt)) {
        std::cout << "Top Left Schur Matrix are the same." << std::endl;
    } else {
        std::cout << "Top Left Schur Matrix are different." << std::endl;
    }

    for(int i = 0; i < num_central_partitions; ++i) {
        if (eigenA[i+1]->isApprox(*eigenArc[i])) {
            std::cout << "Center Schur Matrix " + std::to_string(i + 1) + " are the same." << std::endl;
        } else {
            std::cout << "Center Schur Matrix " + std::to_string(i + 1) + " are different." << std::endl;
        }
    }

    // Check if the matrices are the same
    if (eigenA[partitions - 1]->isApprox(eigenArb)) {
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


    //Check if the produced inverted matrices are the same
    if (G_matrices[0]->block(0, 0, (partition_blocksize + 1) * blocksize, (partition_blocksize + 1) * blocksize)\
    .isApprox(eigenGpt.block(0, 0, (partition_blocksize + 1) * blocksize, (partition_blocksize + 1) * blocksize))){
        std::cout << "Top Left prod. G Matrix are the same." << std::endl;
    } else {
        std::cout << "Top Left prod. G Matrix are different." << std::endl;
    }

    //Check if the write-back inverted matrices are the same
    for(int i = 0; i < num_central_partitions; ++i) {
        if (G_matrices[i+1]->block((partition_blocksize + 1) * blocksize, (partition_blocksize + 1) * blocksize, blocksize, blocksize)\
        .isApprox((*eigenGpc[i]).block((partition_blocksize + 1) * blocksize, (partition_blocksize + 1) * blocksize, blocksize, blocksize))) {
            std::cout << "Center prod. G Matrix " + std::to_string(i + 1) + " (second diag) are the same." << std::endl;
        } else {
            std::cout << "Center prod. G Matrix " + std::to_string(i + 1) + " (second diag) are different." << std::endl;
        }
    }

    //Check if the write-back inverted matrices are the same
    if (G_matrices[partitions - 1]->block((partitions - 1 ) * (partition_blocksize) * blocksize, (partitions - 1 ) * (partition_blocksize) * blocksize - blocksize, (partition_blocksize) * blocksize, (partition_blocksize + 1) * blocksize)\
    .isApprox(eigenGpb.block((partitions - 1 ) * (partition_blocksize) * blocksize, (partitions - 1 ) * (partition_blocksize) * blocksize - blocksize, (partition_blocksize) * blocksize, (partition_blocksize + 1) * blocksize))) {
        std::cout << "Bottom Right prod. G Matrix are the same." << std::endl;
    } else {
        std::cout << "Bottom Right prod. G Matrix are different." << std::endl;
    }

    delete[] A_red_top;
    delete[] A_red_bot;

    delete[] G_red_top;
    delete[] G_red_bot;

    for(int i = 0; i < partitions; ++i) {
        delete eigenA[i];
        delete G_matrices[i];
        delete L_matrices[i];
        delete U_matrices[i];
        if (i < num_central_partitions){
            delete eigenArc[i];
            delete eigenGrc[i];
            delete eigenGpc[i];
        }
    }

    delete[] eigenA;
    delete[] G_matrices;
    delete[] L_matrices;
    delete[] U_matrices;
    delete[] eigenArc;
    delete[] eigenGrc;
    delete[] eigenGpc;

    return G_final;


}
