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
