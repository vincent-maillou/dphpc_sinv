#include "PSR.h"

void myFunction(Eigen::MatrixXcd& A) {
    // Your function logic here
    std::cout << A << std::endl;
}

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

void reduce_schur_sequentially(Eigen::MatrixXcd** eigenA,
                             Eigen::MatrixXcd** G_matrices,
                             Eigen::MatrixXcd** L_matrices,
                             Eigen::MatrixXcd** U_matrices,
                             int partitions,
                             int partition_blocksize,
                             int blocksize,
                             int rank) {
   // Generate the G, L and U matrices for each process
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

        if (i > 0 && i < partitions - 1){
            auto result = reduce_schur_central(*eigenA[i], start_blockrow, partition_blocksize, blocksize);
            *L_matrices[i] = std::get<0>(result);
            *U_matrices[i] = std::get<1>(result);
        }

        if (i == partitions - 1){
            auto result = reduce_schur_bottomrightcorner(*eigenA[i], start_blockrow, partition_blocksize, blocksize);
            *L_matrices[i] = std::get<0>(result);
            *U_matrices[i] = std::get<1>(result);
        }

    }
}


std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd> reduce_schur_topleftcorner(
    Eigen::MatrixXcd& A,
    int start_blockrow,
    int partition_blocksize,
    int blocksize
) {
    Eigen::MatrixXcd L = Eigen::MatrixXcd::Zero(A.rows(), A.cols());
    Eigen::MatrixXcd U = Eigen::MatrixXcd::Zero(A.rows(), A.cols());

    // Corner elimination downward
    for (int i_blockrow = start_blockrow + 1; i_blockrow < start_blockrow + partition_blocksize; ++i_blockrow) {
        int im1_rowindice = (i_blockrow - 1) * blocksize;
        int i_rowindice = i_blockrow * blocksize;
        

        Eigen::MatrixXcd A_inv_im1_im1 = A.block(im1_rowindice, im1_rowindice, blocksize, blocksize).inverse();

        L.block(i_rowindice, im1_rowindice, blocksize, blocksize) =
            A.block(i_rowindice, im1_rowindice, blocksize, blocksize) * A_inv_im1_im1;

        U.block(im1_rowindice, i_rowindice, blocksize, blocksize) =
            A_inv_im1_im1 * A.block(im1_rowindice, i_rowindice, blocksize, blocksize);

        A.block(i_rowindice, i_rowindice, blocksize, blocksize) -=
            L.block(i_rowindice, im1_rowindice, blocksize, blocksize) *
            A.block(im1_rowindice, i_rowindice, blocksize, blocksize);
    }

    return std::make_tuple(L, U);
}


std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd> reduce_schur_bottomrightcorner(
    Eigen::MatrixXcd& A,
    int start_blockrow,
    int partition_blocksize,
    int blocksize
) {
    Eigen::MatrixXcd L = Eigen::MatrixXcd::Zero(A.rows(), A.cols());
    Eigen::MatrixXcd U = Eigen::MatrixXcd::Zero(A.rows(), A.cols());

    // Corner elimination upward
    for (int i_blockrow = start_blockrow + partition_blocksize - 2; i_blockrow >= start_blockrow; --i_blockrow) {
        int i_rowindice = i_blockrow * blocksize;
        int ip1_rowindice = (i_blockrow + 1) * blocksize;

        Eigen::MatrixXcd A_inv_ip1_ip1 = A.block(ip1_rowindice, ip1_rowindice, blocksize, blocksize).inverse();

        L.block(i_rowindice, ip1_rowindice, blocksize, blocksize) =
            A.block(i_rowindice, ip1_rowindice, blocksize, blocksize) * A_inv_ip1_ip1;

        U.block(ip1_rowindice, i_rowindice, blocksize, blocksize) =
            A_inv_ip1_ip1 * A.block(ip1_rowindice, i_rowindice, blocksize, blocksize);

        A.block(i_rowindice, i_rowindice, blocksize, blocksize) -=
            L.block(i_rowindice, ip1_rowindice, blocksize, blocksize) *
            A.block(ip1_rowindice, i_rowindice, blocksize, blocksize);
    }

    return std::make_tuple(L, U);
}

std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd> reduce_schur_bottomrightcorner_2(
    Eigen::MatrixXcd& A,
    int partition_blocksize,
    int blocksize
) {
    Eigen::MatrixXcd L = Eigen::MatrixXcd::Zero(A.rows(), A.cols());
    Eigen::MatrixXcd U = Eigen::MatrixXcd::Zero(A.rows(), A.cols());

    // Corner elimination upward
    for (int i_blockrow = partition_blocksize - 1; i_blockrow >= 1; --i_blockrow) {
        int i_rowindice = i_blockrow * blocksize;
	int il1_rowindice = i_rowindice - blocksize;    
        int ip1_rowindice = i_rowindice + blocksize;

        Eigen::MatrixXcd A_inv_ip1_ip1 = A.block(i_rowindice, ip1_rowindice, blocksize, blocksize).inverse();

        L.block(il1_rowindice, ip1_rowindice, blocksize, blocksize) =
            A.block(il1_rowindice, ip1_rowindice, blocksize, blocksize) * A_inv_ip1_ip1;

        U.block(i_rowindice, i_rowindice, blocksize, blocksize) =
            A_inv_ip1_ip1 * A.block(i_rowindice, i_rowindice, blocksize, blocksize);

        A.block(il1_rowindice, i_rowindice, blocksize, blocksize) -=
            L.block(il1_rowindice, ip1_rowindice, blocksize, blocksize) *
            A.block(i_rowindice, i_rowindice, blocksize, blocksize);
    }

    return std::make_tuple(L, U);
}

std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd> reduce_schur_central(
    Eigen::MatrixXcd& A,
    int start_blockrow,
    int partition_blocksize,
    int blocksize
) {
    Eigen::MatrixXcd L = Eigen::MatrixXcd::Zero(A.rows(), A.cols());
    Eigen::MatrixXcd U = Eigen::MatrixXcd::Zero(A.rows(), A.cols());
    
    // Center elimination downward
    for (int i_blockrow = start_blockrow + 2; i_blockrow < start_blockrow + partition_blocksize; ++i_blockrow) {
        int im1_rowindice = (i_blockrow - 1) * blocksize;
        int i_rowindice = i_blockrow * blocksize;

        int top_rowindice = start_blockrow * blocksize;

        Eigen::MatrixXcd A_inv_im1_im1 = A.block(im1_rowindice, im1_rowindice, blocksize, blocksize).inverse();

        L.block(i_rowindice, im1_rowindice, blocksize, blocksize) =
            A.block(i_rowindice, im1_rowindice, blocksize, blocksize) * A_inv_im1_im1;

        L.block(top_rowindice, im1_rowindice, blocksize, blocksize) =
            A.block(top_rowindice, im1_rowindice, blocksize, blocksize) * A_inv_im1_im1;

        U.block(im1_rowindice, i_rowindice, blocksize, blocksize) =
            A_inv_im1_im1 * A.block(im1_rowindice, i_rowindice, blocksize, blocksize);

        U.block(im1_rowindice, top_rowindice, blocksize, blocksize) =
            A_inv_im1_im1 * A.block(im1_rowindice, top_rowindice, blocksize, blocksize);

        A.block(i_rowindice, i_rowindice, blocksize, blocksize) -=
            L.block(i_rowindice, im1_rowindice, blocksize, blocksize) *
            A.block(im1_rowindice, i_rowindice, blocksize, blocksize);

        A.block(top_rowindice, top_rowindice, blocksize, blocksize) -=
            L.block(top_rowindice, im1_rowindice, blocksize, blocksize) *
            A.block(im1_rowindice, top_rowindice, blocksize, blocksize);

        A.block(i_rowindice, top_rowindice, blocksize, blocksize) -=
            L.block(i_rowindice, im1_rowindice, blocksize, blocksize) *
            A.block(im1_rowindice, top_rowindice, blocksize, blocksize);

        A.block(top_rowindice, i_rowindice, blocksize, blocksize) -=
            L.block(top_rowindice, im1_rowindice, blocksize, blocksize) *
            A.block(im1_rowindice, i_rowindice, blocksize, blocksize);
    }

    return std::make_tuple(L, U);
}


std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd> reduce_schur_central_2(
    Eigen::MatrixXcd& A,
    int partition_blocksize,
    int blocksize
) {
    Eigen::MatrixXcd L = Eigen::MatrixXcd::Zero(A.rows(), A.cols());
    Eigen::MatrixXcd U = Eigen::MatrixXcd::Zero(A.rows(), A.cols());

    // Center elimination downward
    for (int i_blockrow = 2; i_blockrow < partition_blocksize; ++i_blockrow) {
        int i_rowindice = i_blockrow * blocksize;
        int im1_rowindice = i_rowindice - blocksize;
	int ip1_rowindice = i_rowindice + blocksize;

        int top_rowindice = 0;
	int top_rowindice_col = blocksize;

        Eigen::MatrixXcd A_inv_im1_im1 = A.block(im1_rowindice, i_rowindice, blocksize, blocksize).inverse();

        L.block(i_rowindice, i_rowindice, blocksize, blocksize) =
            A.block(i_rowindice, i_rowindice, blocksize, blocksize) * A_inv_im1_im1;

        L.block(top_rowindice, i_rowindice, blocksize, blocksize) =
            A.block(top_rowindice, i_rowindice, blocksize, blocksize) * A_inv_im1_im1;

        U.block(im1_rowindice, ip1_rowindice, blocksize, blocksize) =
            A_inv_im1_im1 * A.block(im1_rowindice, ip1_rowindice, blocksize, blocksize);

        U.block(im1_rowindice, top_rowindice_col, blocksize, blocksize) =
            A_inv_im1_im1 * A.block(im1_rowindice, top_rowindice_col, blocksize, blocksize);

        A.block(i_rowindice, ip1_rowindice, blocksize, blocksize) -=
            L.block(i_rowindice, i_rowindice, blocksize, blocksize) *
            A.block(im1_rowindice, ip1_rowindice, blocksize, blocksize);

        A.block(top_rowindice, top_rowindice_col, blocksize, blocksize) -=
            L.block(top_rowindice, i_rowindice, blocksize, blocksize) *
            A.block(im1_rowindice, top_rowindice_col, blocksize, blocksize);

        A.block(i_rowindice, top_rowindice_col, blocksize, blocksize) -=
            L.block(i_rowindice, i_rowindice, blocksize, blocksize) *
            A.block(im1_rowindice, top_rowindice_col, blocksize, blocksize);

        A.block(top_rowindice, ip1_rowindice, blocksize, blocksize) -=
            L.block(top_rowindice, i_rowindice, blocksize, blocksize) *
            A.block(im1_rowindice, ip1_rowindice, blocksize, blocksize);
    }

    return std::make_tuple(L, U);
}


void aggregate_reduced_system_locally(
    Eigen::MatrixXcd& A_schur,
    Eigen::MatrixXcd** A_schur_processes,
    int nblocks_schur_system,
    int partition_blocksize,
    int blocksize,
    int partitions
)
{
    // A_schur will first take as the first row the (local) reduced row of the root process.
    int start_rowindice = 0;

    int start_rowindice_remote = (0 + partition_blocksize - 1) * blocksize;

    int start_colindice = 0;

    int start_colindice_remote = (partition_blocksize - 1) * blocksize;
    
    A_schur.block(0, 0, blocksize, 2 * blocksize) =
        A_schur_processes[0]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize);

    
    // Then, A_schur will aggregate the Schur complement rows of the central processes.
    // Each central process sends 2 rows (4 distinct blocks that have been locally aggregated
    // by the sending process) to the root.
    for (int process_i = 1; process_i < partitions - 1; ++process_i) {
        // Assuming comm.recv is equivalent to direct assignment
        // Upper left double block of process-local A_schur
        start_rowindice = blocksize + (process_i - 1) * 2 * blocksize;

        start_rowindice_remote = (process_i  * partition_blocksize) * blocksize;
        
        start_colindice = 2 * (process_i - 1) * blocksize;

        start_colindice_remote = (process_i * partition_blocksize - 1) * blocksize;
        
        A_schur.block(start_rowindice, start_colindice, blocksize, 2 * blocksize) =
            A_schur_processes[process_i]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize);

        // Upper right single block of process-local A_schur
        start_colindice += 2 * blocksize;

        start_colindice_remote += partition_blocksize * blocksize;

        A_schur.block(start_rowindice, start_colindice, blocksize, blocksize) =
            A_schur_processes[process_i]->block(start_rowindice_remote, start_colindice_remote, blocksize, blocksize);

        // Lower left single block of process-local A_schur
        start_rowindice += blocksize;
        start_colindice -= blocksize;

        start_rowindice_remote = ((process_i + 1) * partition_blocksize - 1) * blocksize;
        start_colindice_remote = (process_i * partition_blocksize) * blocksize;

        A_schur.block(start_rowindice, start_colindice, blocksize, blocksize) =
            A_schur_processes[process_i]->block(start_rowindice_remote, start_colindice_remote, blocksize, blocksize);

        // Lower right double block of process-local A_schur
        start_colindice += blocksize;
        start_colindice_remote += (partition_blocksize -1 ) * blocksize;

        A_schur.block(start_rowindice, start_colindice, blocksize, 2 * blocksize) =
            A_schur_processes[process_i]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize);


    }
    
    // Finally, A_schur will aggregate the Schur complement row of the last process.
    //start_rowindice_remote = 80;

    start_rowindice_remote = (partitions - 1) * partition_blocksize * blocksize;

    start_rowindice = (nblocks_schur_system - 1) * blocksize;


    start_colindice_remote = (partitions - 1) * partition_blocksize * blocksize - blocksize;

    start_colindice = (nblocks_schur_system - 2) * blocksize;
    
    // Assuming comm.recv is equivalent to direct assignment
    A_schur.block(start_rowindice, start_colindice, blocksize, 2 * blocksize) =
        A_schur_processes[partitions-1]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize);
}


void writeback_inverted_system_locally(
    Eigen::MatrixXcd G,
    Eigen::MatrixXcd** G_schur_processes,
    int nblocks_schur_system,
    int partition_blocksize,
    int blocksize,
    int partitions
) 
{
    // full G_BCR will be spread across all processes, the first process will take the upper left double block of G.
    int start_rowindice = 0;

    int start_rowindice_remote = (0 + partition_blocksize - 1) * blocksize;

    int start_colindice = 0;

    int start_colindice_remote = (partition_blocksize - 1) * blocksize;
    
    G_schur_processes[0]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize) =
        G.block(0, 0, blocksize, 2 * blocksize);

    // Then, G will be scattered to the schur complement rows of the central processes.
    // Each central process receives 2 rows (4 distinct blocks that have been locally aggregated
    // by the sending process) from the root.
    for (int process_i = 1; process_i < partitions - 1; ++process_i) {
        // Assuming comm.recv is equivalent to direct assignment
        // Upper left double block of process-local A_schur
        start_rowindice = blocksize + (process_i - 1) * 2 * blocksize;

        start_rowindice_remote = (process_i  * partition_blocksize) * blocksize;
        
        start_colindice = 2 * (process_i - 1) * blocksize;

        start_colindice_remote = (process_i * partition_blocksize - 1) * blocksize;
        
        G_schur_processes[process_i]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize) =
            G.block(start_rowindice, start_colindice, blocksize, 2 * blocksize);
        
        // Upper right single block of process-local G
        start_colindice += 2 * blocksize;

        start_colindice_remote += partition_blocksize * blocksize;

        G_schur_processes[process_i]->block(start_rowindice_remote, start_colindice_remote, blocksize, blocksize) =
            G.block(start_rowindice, start_colindice, blocksize, blocksize);

        // Lower left single block of process-local G
        start_rowindice += blocksize;
        start_colindice -= blocksize;

        start_rowindice_remote = ((process_i + 1) * partition_blocksize - 1) * blocksize;
        start_colindice_remote = (process_i * partition_blocksize) * blocksize;

        G_schur_processes[process_i]->block(start_rowindice_remote, start_colindice_remote, blocksize, blocksize) =
            G.block(start_rowindice, start_colindice, blocksize, blocksize);

        // Lower right double block of process-local G
        start_colindice += blocksize;
        start_colindice_remote += (partition_blocksize -1 ) * blocksize;

        G_schur_processes[process_i]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize) =
           G.block(start_rowindice, start_colindice, blocksize, 2 * blocksize);

    }
    // Finally, G_BCR will scatter the Schur complement row to the last process.

    start_rowindice_remote = (partitions - 1) * partition_blocksize * blocksize;

    start_rowindice = (nblocks_schur_system - 1) * blocksize;


    start_colindice_remote = (partitions - 1) * partition_blocksize * blocksize - blocksize;

    start_colindice = (nblocks_schur_system - 2) * blocksize;
    
    // Assuming comm.recv is equivalent to direct assignment
    G_schur_processes[partitions - 1]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize) =
        G.block(start_rowindice, start_colindice, blocksize, 2 * blocksize);

}

void produce_schur_sequentially(Eigen::MatrixXcd** eigenA,
                             Eigen::MatrixXcd** G_matrices,
                             Eigen::MatrixXcd** L_matrices,
                             Eigen::MatrixXcd** U_matrices,
                             int partitions,
                             int partition_blocksize,
                             int blocksize,
                             int rank) {
   
    for (int i = 0; i < partitions; ++i) {
        int start_blockrow = i * partition_blocksize;

        std::cout << "Process " << rank << " is producing blockrows " << start_blockrow << " to " << start_blockrow + partition_blocksize - 1 << std::endl;


        if (i == 0){
            produceSchurTopLeftCorner(*eigenA[i], *L_matrices[i], *U_matrices[i], *G_matrices[i], start_blockrow, partition_blocksize, blocksize);
        }

        if (i > 0 && i < partitions -1){
            produceSchurCentral(*eigenA[i], *L_matrices[i], *U_matrices[i], *G_matrices[i], start_blockrow, partition_blocksize, blocksize);
        }

        if (i == partitions - 1){
            produceSchurBottomRightCorner(*eigenA[i], *L_matrices[i], *U_matrices[i], *G_matrices[i], start_blockrow, partition_blocksize, blocksize);
        }

    }
}

void produceSchurTopLeftCorner(Eigen::MatrixXcd A,
                               Eigen::MatrixXcd L,
                               Eigen::MatrixXcd U,
                               Eigen::MatrixXcd& G,
                               int start_blockrow,
                               int partition_blocksize,
                               int blocksize) {
    int top_blockrow = start_blockrow;
    int bottom_blockrow = start_blockrow + partition_blocksize;
    
    // Upper left corner produced upwards
    for (int i = bottom_blockrow - 1; i > top_blockrow; --i) {

        int im1_rowindice = (i - 1) * blocksize;
        int i_rowindice = i * blocksize;

        G.block(i_rowindice, im1_rowindice, blocksize, blocksize) =
            -G.block(i_rowindice, i_rowindice, blocksize, blocksize) *
            L.block(i_rowindice, im1_rowindice, blocksize, blocksize);
        
        G.block(im1_rowindice, i_rowindice, blocksize, blocksize) =
            -U.block(im1_rowindice, i_rowindice, blocksize, blocksize) *
            G.block(i_rowindice, i_rowindice, blocksize, blocksize);
        
        G.block(im1_rowindice, im1_rowindice, blocksize, blocksize) =
            (A.block(im1_rowindice, im1_rowindice, blocksize, blocksize).inverse()) -
            U.block(im1_rowindice, i_rowindice, blocksize, blocksize) *
            G.block(i_rowindice, im1_rowindice, blocksize, blocksize);
    }
}


void produceSchurBottomRightCorner(Eigen::MatrixXcd A,
                                   Eigen::MatrixXcd L,
                                   Eigen::MatrixXcd U,
                                   Eigen::MatrixXcd& G,
                                   int start_blockrow,
                                   int partition_blocksize,
                                   int blocksize) {
    int top_blockrow = start_blockrow;
    int bottom_blockrow = start_blockrow + partition_blocksize;
    
    // Lower right corner produced downwards
    for (int i = top_blockrow; i < bottom_blockrow - 1; ++i) {
        int i_rowindice = i * blocksize;
        int ip1_rowindice = (i + 1) * blocksize;

        G.block(i_rowindice, ip1_rowindice, blocksize, blocksize) =
            -G.block(i_rowindice, i_rowindice, blocksize, blocksize) *
            L.block(i_rowindice, ip1_rowindice, blocksize, blocksize);
        
        G.block(ip1_rowindice, i_rowindice, blocksize, blocksize) =
            -U.block(ip1_rowindice, i_rowindice, blocksize, blocksize) *
            G.block(i_rowindice, i_rowindice, blocksize, blocksize);
        
        G.block(ip1_rowindice, ip1_rowindice, blocksize, blocksize) =
            (A.block(ip1_rowindice, ip1_rowindice, blocksize, blocksize).inverse()) -
            U.block(ip1_rowindice, i_rowindice, blocksize, blocksize) *
            G.block(i_rowindice, ip1_rowindice, blocksize, blocksize);
    }
}

void produceSchurBottomRightCorner_2(Eigen::MatrixXcd A,
                                   Eigen::MatrixXcd L,
                                   Eigen::MatrixXcd U,
                                   Eigen::MatrixXcd& G,
                                   int partition_blocksize,
                                   int blocksize) {
    int top_blockrow = 0;
    int bottom_blockrow = partition_blocksize;
    
    // Lower right corner produced downwards
    for (int i = top_blockrow; i < bottom_blockrow - 1; ++i) {
        int i_rowindice = i * blocksize;
        int ip1_rowindice = (i + 1) * blocksize;
        int i_rowindiceCol = i_rowindice + blocksize;
        int ip1_rowindiceCol = ip1_rowindice + blocksize;

        G.block(i_rowindice, ip1_rowindiceCol, blocksize, blocksize) =
            -G.block(i_rowindice, i_rowindiceCol, blocksize, blocksize) *
            L.block(i_rowindice, ip1_rowindiceCol, blocksize, blocksize);
        
        G.block(ip1_rowindice, i_rowindiceCol, blocksize, blocksize) =
            -U.block(ip1_rowindice, i_rowindiceCol, blocksize, blocksize) *
            G.block(i_rowindice, i_rowindiceCol, blocksize, blocksize);
        
        G.block(ip1_rowindice, ip1_rowindiceCol, blocksize, blocksize) =
            (A.block(ip1_rowindice, ip1_rowindiceCol, blocksize, blocksize).inverse()) -
            U.block(ip1_rowindice, i_rowindiceCol, blocksize, blocksize) *
            G.block(i_rowindice, ip1_rowindiceCol, blocksize, blocksize);
    }
}

void produceSchurCentral(Eigen::MatrixXcd A,
                         Eigen::MatrixXcd L,
                         Eigen::MatrixXcd U,
                         Eigen::MatrixXcd& G,
                         int start_blockrow,
                         int partition_blocksize,
                         int blocksize) {
    int top_blockrow = start_blockrow;
    int bottom_blockrow = start_blockrow + partition_blocksize;

    int top_rowindice = top_blockrow * blocksize;
    int topp1_rowindice = (top_blockrow + 1) * blocksize;
    int topp2_rowindice = (top_blockrow + 2) * blocksize;

    int botm1_rowindice = (bottom_blockrow - 2) * blocksize;
    int bot_rowindice = (bottom_blockrow - 1) * blocksize;

    G.block(bot_rowindice, botm1_rowindice, blocksize, blocksize) =
        -1 * (G.block(bot_rowindice, top_rowindice, blocksize, blocksize) *
              L.block(top_rowindice, botm1_rowindice, blocksize, blocksize) +
              G.block(bot_rowindice, bot_rowindice, blocksize, blocksize) *
              L.block(bot_rowindice, botm1_rowindice, blocksize, blocksize));

    G.block(botm1_rowindice, bot_rowindice, blocksize, blocksize) =
        -1 * (U.block(botm1_rowindice, bot_rowindice, blocksize, blocksize) *
              G.block(bot_rowindice, bot_rowindice, blocksize, blocksize) +
              U.block(botm1_rowindice, top_rowindice, blocksize, blocksize) *
              G.block(top_rowindice, bot_rowindice, blocksize, blocksize));

    for (int i = bottom_blockrow - 2; i > top_blockrow; --i) {
        int i_rowindice = i * blocksize;
        int ip1_rowindice = (i + 1) * blocksize;

        G.block(top_rowindice, i_rowindice, blocksize, blocksize) =
            -1 * (G.block(top_rowindice, top_rowindice, blocksize, blocksize) *
                  L.block(top_rowindice, i_rowindice, blocksize, blocksize) +
                  G.block(top_rowindice, ip1_rowindice, blocksize, blocksize) *
                  L.block(ip1_rowindice, i_rowindice, blocksize, blocksize));

        G.block(i_rowindice, top_rowindice, blocksize, blocksize) =
            -1 * (U.block(i_rowindice, ip1_rowindice, blocksize, blocksize) *
                  G.block(ip1_rowindice, top_rowindice, blocksize, blocksize) +
                  U.block(i_rowindice, top_rowindice, blocksize, blocksize) *
                  G.block(top_rowindice, top_rowindice, blocksize, blocksize));
    }

    for (int i = bottom_blockrow - 2; i > top_blockrow + 1; --i) {
        int im1_rowindice = (i - 1) * blocksize;
        int i_rowindice = i * blocksize;
        int ip1_rowindice = (i + 1) * blocksize;

        // Compute the inverse block
        Eigen::MatrixXcd invBlock = A.block(i_rowindice, i_rowindice, blocksize, blocksize).inverse();
        G.block(i_rowindice, i_rowindice, blocksize, blocksize) = invBlock -
            U.block(i_rowindice, top_rowindice, blocksize, blocksize) *
            G.block(top_rowindice, i_rowindice, blocksize, blocksize) -
            U.block(i_rowindice, ip1_rowindice, blocksize, blocksize) *
            G.block(ip1_rowindice, i_rowindice, blocksize, blocksize);

        G.block(im1_rowindice, i_rowindice, blocksize, blocksize) =
            -1 * (U.block(im1_rowindice, top_rowindice, blocksize, blocksize) *
                  G.block(top_rowindice, i_rowindice, blocksize, blocksize) +
                  U.block(im1_rowindice, i_rowindice, blocksize, blocksize) *
                  G.block(i_rowindice, i_rowindice, blocksize, blocksize));

        G.block(i_rowindice, im1_rowindice, blocksize, blocksize) =
            -1 * (G.block(i_rowindice, top_rowindice, blocksize, blocksize) *
                  L.block(top_rowindice, im1_rowindice, blocksize, blocksize) +
                  G.block(i_rowindice, i_rowindice, blocksize, blocksize) *
                  L.block(i_rowindice, im1_rowindice, blocksize, blocksize));
    }

    G.block(topp1_rowindice, topp1_rowindice, blocksize, blocksize) =
        (A.block(topp1_rowindice, topp1_rowindice, blocksize, blocksize).inverse()) -
        U.block(topp1_rowindice, top_rowindice, blocksize, blocksize) *
        G.block(top_rowindice, topp1_rowindice, blocksize, blocksize) -
        U.block(topp1_rowindice, topp2_rowindice, blocksize, blocksize) *
        G.block(topp2_rowindice, topp1_rowindice, blocksize, blocksize);
}


void produceSchurCentral_2(Eigen::MatrixXcd A,
                         Eigen::MatrixXcd L,
                         Eigen::MatrixXcd U,
                         Eigen::MatrixXcd& G,
                         int partition_blocksize,
                         int blocksize) {
    int top_blockrow = 0;
    int bottom_blockrow = partition_blocksize;

    int top_rowindice = 0;
    int topp1_rowindice = blocksize;
    int topp2_rowindice = blocksize << 1;

    int botm1_rowindice = (bottom_blockrow - 2) * blocksize;
    int bot_rowindice = (bottom_blockrow - 1) * blocksize;

    int top_rowindiceCol = blocksize;
    int topp1_rowindiceCol = blocksize << 1;
    int topp2_rowindiceCol = (blocksize << 1) + blocksize;

    int botm1_rowindiceCol = (bottom_blockrow - 1) * blocksize;
    int bot_rowindiceCol = bottom_blockrow * blocksize;

    G.block(bot_rowindice, botm1_rowindiceCol, blocksize, blocksize) =
        -1 * (G.block(bot_rowindice, top_rowindiceCol, blocksize, blocksize) *
              L.block(top_rowindice, botm1_rowindiceCol, blocksize, blocksize) +
              G.block(bot_rowindice, bot_rowindiceCol, blocksize, blocksize) *
              L.block(bot_rowindice, botm1_rowindiceCol, blocksize, blocksize));

    G.block(botm1_rowindice, bot_rowindiceCol, blocksize, blocksize) =
        -1 * (U.block(botm1_rowindice, bot_rowindiceCol, blocksize, blocksize) *
              G.block(bot_rowindice, bot_rowindiceCol, blocksize, blocksize) +
              U.block(botm1_rowindice, top_rowindiceCol, blocksize, blocksize) *
              G.block(top_rowindice, bot_rowindiceCol, blocksize, blocksize));

    for (int i = bottom_blockrow - 2; i > top_blockrow; --i) {
        int i_rowindice = i * blocksize;
        int ip1_rowindice = (i + 1) * blocksize;
	int i_rowindiceCol = i_rowindice + blocksize;
	int ip1_rowindiceCol = ip1_rowindice + blocksize;

        G.block(top_rowindice, i_rowindiceCol, blocksize, blocksize) =
            -1 * (G.block(top_rowindice, top_rowindiceCol, blocksize, blocksize) *
                  L.block(top_rowindice, i_rowindiceCol, blocksize, blocksize) +
                  G.block(top_rowindice, ip1_rowindiceCol, blocksize, blocksize) *
                  L.block(ip1_rowindice, i_rowindiceCol, blocksize, blocksize));

        G.block(i_rowindice, top_rowindiceCol, blocksize, blocksize) =
            -1 * (U.block(i_rowindice, ip1_rowindiceCol, blocksize, blocksize) *
                  G.block(ip1_rowindice, top_rowindiceCol, blocksize, blocksize) +
                  U.block(i_rowindice, top_rowindiceCol, blocksize, blocksize) *
                  G.block(top_rowindice, top_rowindiceCol, blocksize, blocksize));
    }

    for (int i = bottom_blockrow - 2; i > top_blockrow + 1; --i) {
        int im1_rowindice = (i - 1) * blocksize;
        int i_rowindice = i * blocksize;
        int ip1_rowindice = (i + 1) * blocksize;
	int im1_rowindiceCol = im1_rowindice + blocksize;
	int i_rowindiceCol = i_rowindice + blocksize;
	int ip1_rowindiceCol = ip1_rowindice + blocksize;

        // Compute the inverse block
        Eigen::MatrixXcd invBlock = A.block(i_rowindice, i_rowindiceCol, blocksize, blocksize).inverse();
        G.block(i_rowindice, i_rowindiceCol, blocksize, blocksize) = invBlock -
            U.block(i_rowindice, top_rowindiceCol, blocksize, blocksize) *
            G.block(top_rowindice, i_rowindiceCol, blocksize, blocksize) -
            U.block(i_rowindice, ip1_rowindiceCol, blocksize, blocksize) *
            G.block(ip1_rowindice, i_rowindiceCol, blocksize, blocksize);

        G.block(im1_rowindice, i_rowindiceCol, blocksize, blocksize) =
            -1 * (U.block(im1_rowindice, top_rowindiceCol, blocksize, blocksize) *
                  G.block(top_rowindice, i_rowindiceCol, blocksize, blocksize) +
                  U.block(im1_rowindice, i_rowindiceCol, blocksize, blocksize) *
                  G.block(i_rowindice, i_rowindiceCol, blocksize, blocksize));

        G.block(i_rowindice, im1_rowindiceCol, blocksize, blocksize) =
            -1 * (G.block(i_rowindice, top_rowindiceCol, blocksize, blocksize) *
                  L.block(top_rowindice, im1_rowindiceCol, blocksize, blocksize) +
                  G.block(i_rowindice, i_rowindiceCol, blocksize, blocksize) *
                  L.block(i_rowindice, im1_rowindiceCol, blocksize, blocksize));
    }

    G.block(topp1_rowindice, topp1_rowindiceCol, blocksize, blocksize) =
        (A.block(topp1_rowindice, topp1_rowindiceCol, blocksize, blocksize).inverse()) -
        U.block(topp1_rowindice, top_rowindiceCol, blocksize, blocksize) *
        G.block(top_rowindice, topp1_rowindiceCol, blocksize, blocksize) -
        U.block(topp1_rowindice, topp2_rowindiceCol, blocksize, blocksize) *
        G.block(topp2_rowindice, topp1_rowindiceCol, blocksize, blocksize);
}

void aggregate_Gblocks_tofinalinverse_sequentially(int partitions,
                                       int partition_blocksize,
                                       int blocksize,
                                       Eigen::MatrixXcd** G_matrices,
                                       Eigen::MatrixXcd& G_final
)
{       
    for (int i = 0; i < partitions; ++i) {
        int start_blockrow = i * partition_blocksize;

        if (i == 0){
            for (int j = 0; j < partition_blocksize; ++j){
                G_final.block(start_blockrow * blocksize + j * blocksize, start_blockrow * blocksize + j * blocksize, blocksize, blocksize) =\
                    (*G_matrices[i]).block(start_blockrow * blocksize + j * blocksize, start_blockrow * blocksize + j * blocksize, blocksize, blocksize);
                G_final.block(start_blockrow * blocksize + j * blocksize, (start_blockrow + 1) * blocksize + j * blocksize, blocksize, blocksize) =\
                    (*G_matrices[i]).block(start_blockrow * blocksize + j * blocksize, (start_blockrow + 1) * blocksize + j * blocksize, blocksize, blocksize);
                if (j < partition_blocksize - 1){
                    G_final.block((start_blockrow + 1) * blocksize + j * blocksize, start_blockrow * blocksize + j * blocksize, blocksize, blocksize) =\
                        (*G_matrices[i]).block((start_blockrow + 1) * blocksize + j * blocksize, start_blockrow * blocksize + j * blocksize, blocksize, blocksize);
                }
            }
            
        }

        if (i > 0 && i < partitions - 1){
            for (int j = 0; j < partition_blocksize; ++j){
                G_final.block(start_blockrow * blocksize + j * blocksize, start_blockrow * blocksize + j * blocksize, blocksize, blocksize) =\
                (*G_matrices[i]).block(start_blockrow * blocksize + j * blocksize, start_blockrow * blocksize + j * blocksize, blocksize, blocksize);

                G_final.block(start_blockrow * blocksize + j * blocksize, (start_blockrow + 1) * blocksize + j * blocksize, blocksize, blocksize) =\
                (*G_matrices[i]).block(start_blockrow * blocksize + j * blocksize, (start_blockrow + 1) * blocksize + j * blocksize, blocksize, blocksize);

                G_final.block(start_blockrow * blocksize + j * blocksize, (start_blockrow - 1) * blocksize + j * blocksize, blocksize, blocksize) =\
                (*G_matrices[i]).block(start_blockrow * blocksize + j * blocksize, (start_blockrow - 1) * blocksize + j * blocksize, blocksize, blocksize);
            }
        }

        if (i == partitions - 1){
            for (int j = 0; j < partition_blocksize; ++j){
                G_final.block(start_blockrow * blocksize + j * blocksize, start_blockrow * blocksize + j * blocksize, blocksize, blocksize) =\
                (*G_matrices[i]).block(start_blockrow * blocksize + j * blocksize, start_blockrow * blocksize + j * blocksize, blocksize, blocksize);

                G_final.block(start_blockrow * blocksize + j * blocksize, (start_blockrow - 1) * blocksize + j * blocksize, blocksize, blocksize) =\
                (*G_matrices[i]).block(start_blockrow * blocksize + j * blocksize, (start_blockrow - 1) * blocksize + j * blocksize, blocksize, blocksize);

                if(j < partition_blocksize -1){
                    G_final.block(start_blockrow * blocksize + j * blocksize, (start_blockrow + 1) * blocksize + j * blocksize, blocksize, blocksize) =\
                    (*G_matrices[i]).block(start_blockrow * blocksize + j * blocksize, (start_blockrow + 1) * blocksize + j * blocksize, blocksize, blocksize);
                }
            }
        }

    }

}


void fill_buffer(Eigen::MatrixXcd& inMatrix, Eigen::MatrixXcd** eigenA, int partition_blocksize, int blocksize, int rank, int partitions) {

    if(rank == 0) {
	int start_rowindice_remote = (0 + partition_blocksize - 1) * blocksize;

	int start_colindice_remote = (partition_blocksize - 1) * blocksize;

	inMatrix.block(0, 0, blocksize, 2*blocksize) = (eigenA[0]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize));
    } else if (rank == partitions-1) {
	int start_rowindice_remote = (partitions - 1) * partition_blocksize * blocksize;

	int start_colindice_remote = (partitions - 1) * partition_blocksize * blocksize - blocksize;

	// Assuming comm.recv is equivalent to direct assignment
	inMatrix.block(0, 0, blocksize, 2*blocksize) = (eigenA[partitions-1]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize));
    } else {
	// Upper right double block of process-local A_schur
        int start_rowindice_remote = (rank  * partition_blocksize) * blocksize;
        
        int start_colindice_remote = (rank * partition_blocksize - 1) * blocksize;
        
        inMatrix.block(0, 0, blocksize, 2*blocksize) =  (eigenA[rank]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize));

        // Upper right single block of process-local A_schur
        start_colindice_remote += partition_blocksize * blocksize;

        inMatrix.block(0, 2*blocksize, blocksize, blocksize) = (eigenA[rank]->block(start_rowindice_remote, start_colindice_remote, blocksize, blocksize));

        // Lower left single block of process-local A_schur
        start_rowindice_remote = ((rank + 1) * partition_blocksize - 1) * blocksize;
        start_colindice_remote = (rank * partition_blocksize) * blocksize;

        inMatrix.block(0, 3*blocksize, blocksize, blocksize) = (eigenA[rank]->block(start_rowindice_remote, start_colindice_remote, blocksize, blocksize));

        // Lower right double block of process-local A_schur
        start_colindice_remote += (partition_blocksize -1 ) * blocksize;

        inMatrix.block(0, 4*blocksize, blocksize, 2*blocksize) = (eigenA[rank]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize));
    }
}

void fill_buffer_2(Eigen::MatrixXcd& inMatrix, Eigen::MatrixXcd processA, int partition_blocksize, int blocksize, int rank, int partitions) {

    if(rank == 0) {
	int start_rowindice_remote = (0 + partition_blocksize - 1) * blocksize;

	int start_colindice_remote = (partition_blocksize - 1) * blocksize;

	inMatrix.block(0, 0, blocksize, 2*blocksize) = (processA.block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize));
    } else if (rank == partitions-1) {
	// Assuming comm.recv is equivalent to direct assignment
	inMatrix.block(0, 0, blocksize, 2*blocksize) = (processA.block(0, 0, blocksize, 2 * blocksize));
    } else {
	// Upper right double block of process-local A_schur
        int start_rowindice_remote = (rank  * partition_blocksize) * blocksize;
        
        int start_colindice_remote = (rank * partition_blocksize - 1) * blocksize;
        
        inMatrix.block(0, 0, blocksize, 2*blocksize) =  (processA.block(0, 0, blocksize, 2 * blocksize));

        // Upper right single block of process-local A_schur
        start_colindice_remote = partition_blocksize * blocksize;

        inMatrix.block(0, 2*blocksize, blocksize, blocksize) = (processA.block(0, start_colindice_remote, blocksize, blocksize));

        // Lower left single block of process-local A_schur
        start_rowindice_remote = (partition_blocksize - 1) * blocksize;

        inMatrix.block(0, 3*blocksize, blocksize, blocksize) = (processA.block(start_rowindice_remote, blocksize, blocksize, blocksize));

        // Lower right double block of process-local A_schur
        start_colindice_remote = partition_blocksize * blocksize;

        inMatrix.block(0, 4*blocksize, blocksize, 2*blocksize) = (processA.block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize));
    }
}



void fill_reduced_schur_matrix(Eigen::MatrixXcd& A_schur, double* comm_buf, int in_buf_size, int blocksize, int partitions) {

    const int rowSize = blocksize;
    const int colSizeCorner = blocksize << 1;
    const int colSizeMiddle = colSizeCorner + blocksize;
    const int half_buf_size = 6*blocksize*blocksize;
   
    // Fill in the 2 blocks from the top process (i.e. rank 0)
    A_schur.block( 0, 0, rowSize, colSizeCorner) = 
       	    Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
	    ( (std::complex<double>*) comm_buf, rowSize, colSizeCorner);
    
    //Fill in the 2 blocks from the bottom process (i.e rank (partitions-1))
    A_schur.block( (2*partitions-3)*blocksize, (2*partitions-4)*blocksize, rowSize, colSizeCorner) = 
            Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
	    ( (std::complex<double>*) (comm_buf + (partitions-1)*in_buf_size), rowSize, colSizeCorner);

    //Fill in the the 6 blocks over two rows from the processes in the middle (i.e. rank > 0 && rank < (partitions - 1))
    for(int i = 1; i < partitions-1; ++i) {
	A_schur.block( (2*i-1)*blocksize, (2*i-2)*blocksize, rowSize, colSizeMiddle) = 
		Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
		( (std::complex<double>*) (comm_buf + i*in_buf_size), rowSize, colSizeMiddle);

	A_schur.block( (2*i)*blocksize, (2*i-1)*blocksize, rowSize, colSizeMiddle) = 
		Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
		( (std::complex<double>*) (comm_buf + i*in_buf_size + half_buf_size), rowSize, colSizeMiddle);
    }

}

Eigen::MatrixXcd psr_seqsolve(int N,
                             int blocksize,
                             int n_blocks,
                             int partitions,
                             int partition_blocksize,
                             int rank,
                             int n_blocks_schursystem,
                             Eigen::MatrixXcd& eigenA_read_in,
                             bool compare_reference
){

    Eigen::MatrixXcd** eigenA = new Eigen::MatrixXcd*[partitions];
    for(int i = 0; i < partitions; ++i) {
        eigenA[i] = new Eigen::MatrixXcd(N, N);
        *eigenA[i] = eigenA_read_in;
    }

    // Referece inverse
    Eigen::MatrixXcd full_inverse;
    if(compare_reference){
        full_inverse = eigenA_read_in.inverse();
    }

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

    aggregate_reduced_system_locally(A_schur, eigenA, n_blocks_schursystem, partition_blocksize, blocksize, partitions);

    auto G_schur = A_schur.inverse();

    writeback_inverted_system_locally(G_schur, G_matrices, n_blocks_schursystem, partition_blocksize, blocksize, partitions);

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

    if(compare_reference){
        compareSINV_referenceInverse_byblock(n_blocks,
                                     blocksize,
                                     G_final,
                                     full_inverse
        );
    }

    for(int i = 0; i < partitions; ++i) {
        delete eigenA[i];
        delete G_matrices[i];
        delete L_matrices[i];
        delete U_matrices[i];
    }

    delete[] eigenA;
    delete[] G_matrices;
    delete[] L_matrices;
    delete[] U_matrices;

    return G_final;


}


Eigen::MatrixXcd psr_solve(int N,
                             int blocksize,
                             int n_blocks,
                             int partitions,
                             int partition_blocksize,
                             int rank,
                             int n_blocks_schursystem,
                             Eigen::MatrixXcd& eigenA_read_in,
                             bool compare_reference
){  

    Eigen::MatrixXcd eigenA2 = eigenA_read_in;

    // Referece inverse
    Eigen::MatrixXcd full_inverse;
    if(compare_reference){
        full_inverse = eigenA_read_in.inverse();
    }
	
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

    if(compare_reference){
        compareSINV_referenceInverse_byblock(n_blocks,
                                     blocksize,
                                     G_final,
                                     full_inverse
        );
    }

    return G_final;
}

