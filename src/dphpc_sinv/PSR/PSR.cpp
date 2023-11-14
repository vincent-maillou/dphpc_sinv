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

    std::cout << "extracting left top corner block done!" << std::endl;
    
    // Then, A_schur will aggregate the Schur complement rows of the central processes.
    // Each central process sends 2 rows (4 distinct blocks that have been locally aggregated
    // by the sending process) to the root.
    for (int process_i = 1; process_i < partitions - 1; ++process_i) {
        // Assuming comm.recv is equivalent to direct assignment
        std::cout << "process_i: " << process_i << std::endl;
        // Upper left double block of process-local A_schur
        start_rowindice = blocksize + (process_i - 1) * 2 * blocksize;

        start_rowindice_remote = (process_i  * partition_blocksize) * blocksize;
        
        start_colindice = 2 * (process_i - 1) * blocksize;

        start_colindice_remote = (process_i * partition_blocksize - 1) * blocksize;

        std::cout << "start_rowindice: " << start_rowindice << std::endl;
        std::cout << "start_rowindice_remote: " << start_rowindice_remote << std::endl;

        std::cout << "start_colindice: " << start_colindice << std::endl;
        std::cout << "start_colindice_remote: " << start_colindice_remote << std::endl;

        
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

    //start_rowindice = 24;

    //start_colindice_remote = 72;

    start_colindice_remote = (partitions - 1) * partition_blocksize * blocksize - blocksize;

    start_colindice = (nblocks_schur_system - 2) * blocksize;

    // std::cout << start_rowindice_remote << std::endl;
    // std::cout << start_rowindice << std::endl;
    
    // Assuming comm.recv is equivalent to direct assignment
    A_schur.block(start_rowindice, start_colindice, blocksize, 2 * blocksize) =
        A_schur_processes[partitions-1]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize);
}


