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

    aggregate_reduced_system_locally(A_schur, eigenA, n_blocks_schursystem, partition_blocksize, blocksize, partitions);

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
){  

    // Load in all relevant Matrices for Testing

    // Load in relevant A partitions after reduce_schur
    std::complex<double>* A_red = new std::complex<double>[N * N];
    Eigen::MatrixXcd eigenAr;
    if(rank == 0) {
	load_matrix(test_folder + "A_red_s_top_full.bin", A_red, N, N);
	eigenAr = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(A_red, N, N);
    }
    if(rank > 0 && rank < partitions - 1) {
	read_central_testblock(eigenAr, N, rank, test_folder + "A_red_s_central_full");
    }
    if(rank == partitions - 1) {
	load_matrix(test_folder + "A_red_s_bottom_full.bin", A_red, N, N);
	eigenAr = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(A_red, N, N);
    }

    // Load in the reduced A_schur and it's inverse G_schur
    std::complex<double>* A_schur_test = new std::complex<double>[blocksize * n_blocks_schursystem * blocksize * n_blocks_schursystem];
    load_matrix(test_folder + "A_schur.bin", A_schur_test,  blocksize * n_blocks_schursystem, blocksize * n_blocks_schursystem);
    Eigen::MatrixXcd eigenA_schur_test = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(A_schur_test, blocksize * n_blocks_schursystem, blocksize * n_blocks_schursystem);

    std::complex<double>* G_schur_test = new std::complex<double>[blocksize * n_blocks_schursystem * blocksize * n_blocks_schursystem];
    load_matrix(test_folder + "G_schur.bin", G_schur_test, blocksize * n_blocks_schursystem , blocksize * n_blocks_schursystem);
    Eigen::MatrixXcd eigenG_schur_test = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(G_schur_test, blocksize * n_blocks_schursystem, blocksize * n_blocks_schursystem);

    // Load in all relevant partitions of G after writing back from G_schur
    std::complex<double>* G_red = new std::complex<double>[N*N];
    Eigen::MatrixXcd eigenGr;
    if(rank == 0) {
	load_matrix(test_folder + "G_red_s_top_full.bin", G_red, N, N);
	eigenGr = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(G_red, N, N);
    }
    if(rank > 0 && rank < partitions - 1) {
	read_central_testblock(eigenGr, N, rank, test_folder + "G_red_s_central_full");
    }
    if(rank == partitions - 1) {
	load_matrix(test_folder + "G_red_s_bottom_full.bin", G_red, N, N);
	eigenGr = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(G_red, N, N);
    }

    // Load in all relevant partitions of G after produce_schur
    std::complex<double>* G_prod = new std::complex<double>[N*N];
    Eigen::MatrixXcd eigenGp;
    if(rank == 0) {
	load_matrix(test_folder + "G_prod_s_top_full.bin", G_prod, N, N);
	eigenGp = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(G_prod, N, N);
    }
    if(rank > 0 && rank < partitions - 1) {
	read_central_testblock(eigenGp, N, rank, test_folder + "G_prod_s_central_full");
    }
    if(rank == partitions - 1) {
	load_matrix(test_folder + "G_prod_s_bottom_full.bin", G_prod, N, N);
	eigenGp = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(G_prod, N, N);
    }




    Eigen::MatrixXcd eigenA2 = eigenA_read_in;

    // Referece inverse
    Eigen::MatrixXcd full_inverse = eigenA_read_in.inverse();
	
    //Limit it to the processes partition of A
    int start_blockrow = rank * partition_blocksize;
    int rowSizePartition = partition_blocksize * blocksize;
    int colSizePartition = (partition_blocksize + 2) * blocksize;
    Eigen::MatrixXcd processA;
    Eigen::MatrixXcd G,L,U;

    if (rank == 0) {
	processA = eigenA2.block(0, 0, rowSizePartition, colSizePartition-blocksize);
	G = Eigen::MatrixXcd::Zero(rowSizePartition, colSizePartition-blocksize);
    }
    if (rank > 0 && rank < partitions - 1) {
	int startRowIndex = start_blockrow*blocksize;
	int startColIndex = (start_blockrow-1)*blocksize;
	processA = eigenA2.block(startRowIndex, startColIndex, rowSizePartition, colSizePartition);
	G = Eigen::MatrixXcd::Zero(rowSizePartition, colSizePartition);
    }
    if (rank == partitions - 1) {
	int startRowIndex = start_blockrow*blocksize;
	int startColIndex = (start_blockrow-1)*blocksize;
	processA = eigenA2.block(startRowIndex, startColIndex, rowSizePartition, colSizePartition-blocksize);
	G = Eigen::MatrixXcd::Zero(rowSizePartition, colSizePartition-blocksize);
    }


    // Start reduce_schur
    std::cout << "Process " << rank << " is reducing blockrows " << start_blockrow << " to " << start_blockrow + partition_blocksize - 1 << std::endl;


    if (rank == 0){
        auto result = reduce_schur_topleftcorner(processA, 0, partition_blocksize, blocksize);
        L = std::get<0>(result);
        U = std::get<1>(result);
    }

    if (rank > 0 && rank < partitions - 1){
        auto result = reduce_schur_central_2(processA, partition_blocksize, blocksize);
        L = std::get<0>(result);
        U = std::get<1>(result);
    }
    
    if (rank == partitions - 1){
        auto result = reduce_schur_bottomrightcorner_2(processA, partition_blocksize, blocksize);
        L = std::get<0>(result);
        U = std::get<1>(result);
    }
    // End reduce_schur

    // Start of MPIALLGATHER for reduced_schur_system and inverse of said system
    Eigen::MatrixXcd A_schur = Eigen::MatrixXcd(blocksize*n_blocks_schursystem, blocksize*n_blocks_schursystem);
    A_schur.setZero();

    unsigned long comm_buf_size = (blocksize * blocksize * partitions * 6) << 1; 
    double* comm_buf = new double[comm_buf_size];

    unsigned long in_buf_size = (blocksize * blocksize * 6) << 1;
    Eigen::MatrixXcd inMatrix = Eigen::MatrixXcd::Zero(blocksize, 6*blocksize);
    fill_buffer_2(inMatrix, processA, partition_blocksize, blocksize, rank, partitions);


    double* in_buf = (double*) inMatrix.data();
    MPI_Allgather(in_buf, in_buf_size, MPI_DOUBLE, comm_buf, in_buf_size, MPI_DOUBLE, MPI_COMM_WORLD);

    fill_reduced_schur_matrix(A_schur, comm_buf, in_buf_size, blocksize, partitions);

    delete[] comm_buf;

    auto G_schur = A_schur.inverse();
    // End of MPIALLGATHER for reduced_schur_system and inverse of said system

    // Start of writeback of reduced inverse to full G partitions
    if(rank == 0) {
	int start_rowindice = (partition_blocksize - 1) * blocksize;
	int start_colindice = (partition_blocksize - 1) * blocksize;
	G.block(start_rowindice, start_colindice, blocksize, (blocksize << 1)) = G_schur.block(0, 0, blocksize, (blocksize << 1));	
    }
    if(rank > 0 && rank < partitions - 1) {
	// Upper left double block of process-local G
	int start_rowindice_remote = (1 + ((rank - 1) << 1)) * blocksize; // (rank - 1) * 2 + 1
	int start_colindice_remote = ((rank - 1) << 1) * blocksize; // (rank - 1) * 2
	G.block(0, 0, blocksize, (blocksize << 1)) = G_schur.block(start_rowindice_remote, start_colindice_remote, blocksize, (blocksize << 1));

	// Upper right single block of process-local G
	int start_colindice = partition_blocksize * blocksize;
	start_colindice_remote += (blocksize << 1);
	G.block(0, start_colindice, blocksize, blocksize) = G_schur.block(start_rowindice_remote, start_colindice_remote, blocksize, blocksize);

	// Lower left single block of process-local G
	int start_rowindice = (partition_blocksize - 1) * blocksize;
	start_colindice = blocksize;
	start_rowindice_remote += blocksize;
	start_colindice_remote -= blocksize;
	G.block(start_rowindice, start_colindice, blocksize, blocksize) = G_schur.block(start_rowindice_remote, start_colindice_remote, blocksize, blocksize);

	// Lower right double block of process-local G
	start_colindice = partition_blocksize * blocksize;
	start_colindice_remote += blocksize;
	G.block(start_rowindice, start_colindice, blocksize, (blocksize << 1)) = G_schur.block(start_rowindice_remote, start_colindice_remote, blocksize, (blocksize << 1));
    }
    if(rank == partitions - 1) {
	int start_rowindice_remote = (1 + ((partitions - 2) << 1)) * blocksize;
	int start_colindice_remote = start_rowindice_remote - blocksize;
	G.block(0, 0, blocksize, (blocksize << 1)) = G_schur.block(start_rowindice_remote, start_colindice_remote, blocksize, (blocksize << 1));
    }
    // End of writeback of reduced inverse to full G partitions


    // Start of produce_schur
    std::cout << "Process " << rank << " is producing blockrows " << start_blockrow << " to " << start_blockrow + partition_blocksize - 1 << std::endl;

    if(rank == 0) {
	produceSchurTopLeftCorner(processA, L, U, G, 0, partition_blocksize, blocksize);
    }
    if(rank > 0 && rank < partitions - 1) {
	produceSchurCentral_2(processA, L, U, G, partition_blocksize, blocksize);
    }
    if(rank == partitions - 1) {
	produceSchurBottomRightCorner_2(processA, L, U, G, partition_blocksize, blocksize);
    }
    // End of produce_schur

    // Start of reconstructing Tridiagonal system of the full inverse via MPIALLGATHER    
    Eigen::MatrixXcd G_final = Eigen::MatrixXcd(N, N);
    G_final.setZero();

    comm_buf_size = (rowSizePartition * colSizePartition * partitions) << 1;
    comm_buf = new double[comm_buf_size];
    in_buf_size = (rowSizePartition * colSizePartition) << 1;
    // !!!!!!
    // Attention I am currently purposefully overshooting the boundaries of G.data() for processes 0 and partitions - 1 
    // in the MPI_Allgather i.e. for those processes G.data() has only size ((rowSizePartition * (colSizePartition - blocksize)) << 1) insted of in_buf_size.
    // But since I am not writing anything back from that overshoot area it doesn't impact correctness currently
    // !!!!!
    in_buf = (double*) G.data();
    MPI_Allgather(in_buf, in_buf_size, MPI_DOUBLE, comm_buf, in_buf_size, MPI_DOUBLE, MPI_COMM_WORLD);

    
    for(int i = 0; i < partitions; ++i) {
	int start_rowindice = i * partition_blocksize * blocksize;
	int start_colindice = 0;
	if(i > 0) {
		start_colindice = start_rowindice - blocksize;
	}
	int rowSize = rowSizePartition;
	int colSize = colSizePartition;
	if(i == 0 || i == partitions - 1) {
		colSize -= blocksize;
	}
	G_final.block(start_rowindice, start_colindice, rowSize, colSize) =
			Eigen::Map<Eigen::MatrixXcd> ( (std::complex<double>*) (comm_buf + (i * in_buf_size)), rowSize, colSize);

	// Setting the off Tridiagonal blocks used in the produceSchurCentral step to 0 
	if(i > 0 && i < partitions - 1) {
		(G_final.block(start_rowindice, start_colindice + 3 * blocksize, blocksize, (partition_blocksize - 1) * blocksize)).setZero();
		(G_final.block(start_rowindice + 2 * blocksize, start_colindice + blocksize, (partition_blocksize - 1) * blocksize, blocksize)).setZero();
	}

    }

    delete[] comm_buf;
    // End of reconstructing Tridiagonal system of the full inverse via MPIALLGATHER    
    
    compareSINV_referenceInverse_byblock(n_blocks,
    			     blocksize,
    			     G_final,
    			     full_inverse
    );

    // done to avoid more if statements for every check
    if(rank == 0 || rank == partitions - 1) {
	colSizePartition -= blocksize;
    }

    //Check if matrices after reduce_schur are the same
    int rowStart = start_blockrow * blocksize;
    int colStart = std::max(0, rowStart - blocksize);
    if(processA.isApprox(eigenAr.block(rowStart, colStart, rowSizePartition, colSizePartition))) {
        std::cout << "Schur Matrices for process " + std::to_string(rank) + " are the same." << std::endl;
    } else {
        std::cout << "Schur Matrices for process " + std::to_string(rank) + " are different." << std::endl;
    }


    //Check if reduced A_schur matrices are the same
    if (A_schur.isApprox(eigenA_schur_test)) {
        std::cout << "A_schur are the same." << std::endl;
    } else {
        std::cout << "A_schur are different." << std::endl;
    }

    //Check if the inverse of the reduced A_schur matrices are the same
    if (G_schur.isApprox(eigenG_schur_test)) {
        std::cout << "G_schur are the same." << std::endl;
    } else {
        std::cout << "G_schur are different." << std::endl;
    }

    //Check if matrices after produce_schur are the same
    if(rank == 0 || rank == partitions - 1) {
	if (G.isApprox(eigenGp.block(rowStart, colStart, rowSizePartition, colSizePartition))) {
		std::cout << "G Matrices for process " + std::to_string(rank) + " are the same." << std::endl;
	} else {
		std::cout << "G Matrices for process " + std::to_string(rank) + " are different." << std::endl;
	}
    } else {
	bool test = true;
	for(int i = 0; i < partition_blocksize; ++i) {
		if(!G.block(i * blocksize, i*blocksize, blocksize, 3*blocksize).isApprox(eigenGp.block( rowStart + i*blocksize, colStart + i*blocksize, blocksize, 3*blocksize))) {
			test = false;	
		}
	}
        if(test) {
		std::cout << "G Matrices for process " + std::to_string(rank) + " are the same." << std::endl;
	} else {
		std::cout << "G Matrices for process " + std::to_string(rank) + " are different." << std::endl;
	}
    }


    return G_final;
}


