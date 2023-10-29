#include <Eigen/Dense>
#include <iostream>
#include <vector>

using namespace Eigen;

// Define the bsparse data structure
struct bsparse {
    std::vector<MatrixXcd> blocks;
    int blockorder;
};

bsparse rgf(bsparse A, bool sym_mat = false, bool save_off_diag = true) {
    // Storage for the full backward substitution
    bsparse G;
    G.blockorder = A.blockorder;
    G.blocks.resize(G.blockorder);

    // 0. Inverse of the first block
    G.blocks[0] = A.blocks[0].inverse();

    // 1. Forward substitution (performed left to right)
    for (int i = 1; i < A.blockorder; i++) {
        G.blocks[i] = (A.blocks[i] - A.blocks[i] * G.blocks[i-1] * A.blocks[i-1].adjoint()).inverse();
    }

    // 2. Backward substitution (performed right to left)
    for (int i = A.blockorder-2; i >= 0; i--) {
        MatrixXcd g_ii = G.blocks[i];
        MatrixXcd G_lowerfactor = G.blocks[i+1] * A.blocks[i+1].adjoint() * g_ii;

        if (save_off_diag) {
            G.blocks[i+1] = -G_lowerfactor;
            if (sym_mat) {
                G.blocks[i] = G.blocks[i+1].adjoint();
            } else {
                G.blocks[i] = -g_ii * A.blocks[i+1] * G.blocks[i+1];
            }
        }

        G.blocks[i] += g_ii * G_lowerfactor * A.blocks[i].adjoint();
    }

    return G;
}