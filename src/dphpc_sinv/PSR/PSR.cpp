#include "PSR.h"

void myFunction(Eigen::MatrixXcd& A) {
    // Your function logic here
    std::cout << A << std::endl;
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
        int ip1_rowindice = (i_blockrow + 1) * blocksize;

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
        int ip2_rowindice = (i_blockrow + 2) * blocksize;

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
        int ip1_rowindice = (i_blockrow + 1) * blocksize;

        int top_rowindice = start_blockrow * blocksize;
        int topp1_rowindice = (start_blockrow + 1) * blocksize;

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
        int ip1_rowindice = (i + 1) * blocksize;

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
        int ip2_rowindice = (i + 2) * blocksize;

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
    int topp3_rowindice = (top_blockrow + 3) * blocksize;

    int botm1_rowindice = (bottom_blockrow - 2) * blocksize;
    int bot_rowindice = (bottom_blockrow - 1) * blocksize;
    int botp1_rowindice = bottom_blockrow * blocksize;

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
        int ip2_rowindice = (i + 2) * blocksize;

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
        int ip2_rowindice = (i + 2) * blocksize;

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


