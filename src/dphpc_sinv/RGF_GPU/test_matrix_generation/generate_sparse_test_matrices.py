# Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.

import matrix_utils

import numpy as np
import matplotlib.pyplot as plt
import os

SEED = 63
MAT_SIZE = 3000
BLOCKSIZE = 1000
SPARSITY = 0.5
PATH_TO_FILE = "/usr/scratch/mont-fort17/almaeder/rgf_test/"


if __name__ == "__main__":
    # Generate random matrix
    matrix = matrix_utils.generateSparseBandedMatrix(MAT_SIZE, BLOCKSIZE, SPARSITY, SEED)

    # Assert matrix to be invertible
    assert np.allclose( np.linalg.inv(matrix) @ matrix, np.eye(MAT_SIZE) )
    assert np.linalg.det(matrix) != 0

    # Compute inverse
    inv_matrix = np.linalg.inv(matrix)


    # Extract diagonal and off-diagonal blocks
    matrix_diag_blk = matrix_utils.extract_diagonal_blocks(matrix, MAT_SIZE, BLOCKSIZE)
    matrix_upper_blk = matrix_utils.extract_offdiagonal_blocks(matrix, MAT_SIZE, BLOCKSIZE, 1)
    matrix_lower_blk = matrix_utils.extract_offdiagonal_blocks(matrix, MAT_SIZE, BLOCKSIZE, -1)

    matrix_inv_diag_blk = matrix_utils.extract_diagonal_blocks(inv_matrix, MAT_SIZE, BLOCKSIZE)
    matrix_inv_upper_blk = matrix_utils.extract_offdiagonal_blocks(inv_matrix, MAT_SIZE, BLOCKSIZE, 1)
    matrix_inv_lower_blk = matrix_utils.extract_offdiagonal_blocks(inv_matrix, MAT_SIZE, BLOCKSIZE, -1)

    """ plt.matshow(matrix.real)
    plt.matshow(matrix_diag_blk.real)
    plt.matshow(matrix_upper_blk.real)
    plt.matshow(matrix_lower_blk.real)
    plt.show() """
    plt.matshow(matrix.real)
    plt.savefig("matrix.png")

    matrix_utils.write_dense_matrix_to_sparse_blocks_file(
        PATH_TO_FILE,
        matrix,
        BLOCKSIZE
    )

    # Save matrices to file
    filename = "sparse_blocks_matrix_0_diagblk.bin"
    matrix_utils.write_matrix_to_file(PATH_TO_FILE+filename, matrix_diag_blk)
    filename = "sparse_blocks_matrix_0_upperblk.bin"
    matrix_utils.write_matrix_to_file(PATH_TO_FILE+filename, matrix_upper_blk)
    filename = "sparse_blocks_matrix_0_lowerblk.bin"
    matrix_utils.write_matrix_to_file(PATH_TO_FILE+filename, matrix_lower_blk)

    filename = "sparse_blocks_matrix_0_inverse_diagblk.bin"
    matrix_utils.write_matrix_to_file(PATH_TO_FILE+filename, matrix_inv_diag_blk)
    filename = "sparse_blocks_matrix_0_inverse_upperblk.bin"
    matrix_utils.write_matrix_to_file(PATH_TO_FILE+filename, matrix_inv_upper_blk)
    filename = "sparse_blocks_matrix_0_inverse_lowerblk.bin"
    matrix_utils.write_matrix_to_file(PATH_TO_FILE+filename, matrix_inv_lower_blk)

    filename = "sparse_blocks_matrix_0_parameters.txt"
    matrix_utils.write_matrix_parameters(PATH_TO_FILE+filename, MAT_SIZE, BLOCKSIZE)
