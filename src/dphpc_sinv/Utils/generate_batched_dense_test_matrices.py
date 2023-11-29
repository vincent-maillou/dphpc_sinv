# Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.

from dphpc_sinv.Utils import matrix_utils

import numpy as np
import matplotlib.pyplot as plt
import os

SEED = 8000
MAT_SIZE = 5000
BLOCKSIZE = 500
BATCHSIZE = 100
#PATH_TO_FILE = "../../../tests/tests_cases/"
PATH_TO_FILE = "/usr/scratch/mont-fort17/almaeder/rgf_test/"

if __name__ == "__main__":

    filename = "batched_matrix_parameters.txt"
    matrix_utils.write_matrix_parameters_batched(
        PATH_TO_FILE+filename, MAT_SIZE, BLOCKSIZE, BATCHSIZE)

    # Generate random matrices
    for i in range(BATCHSIZE):
        matrix = matrix_utils.generateBandedMatrix(MAT_SIZE, BLOCKSIZE, SEED)
        # Assert matrix to be invertible
        assert np.allclose(np.linalg.inv(matrix) @ matrix, np.eye(MAT_SIZE))
        assert np.linalg.det(matrix) != 0

        # Compute inverse
        inv_matrix = np.linalg.inv(matrix)

        # Extract diagonal and off-diagonal blocks
        matrix_diag_blk = matrix_utils.extract_diagonal_blocks(
            matrix, MAT_SIZE, BLOCKSIZE)
        matrix_upper_blk = matrix_utils.extract_offdiagonal_blocks(
            matrix, MAT_SIZE, BLOCKSIZE, 1)
        matrix_lower_blk = matrix_utils.extract_offdiagonal_blocks(
            matrix, MAT_SIZE, BLOCKSIZE, -1)

        matrix_inv_diag_blk = matrix_utils.extract_diagonal_blocks(
            inv_matrix, MAT_SIZE, BLOCKSIZE)
        matrix_inv_upper_blk = matrix_utils.extract_offdiagonal_blocks(
            inv_matrix, MAT_SIZE, BLOCKSIZE, 1)
        matrix_inv_lower_blk = matrix_utils.extract_offdiagonal_blocks(
            inv_matrix, MAT_SIZE, BLOCKSIZE, -1)

        """ plt.matshow(matrix.real)
        plt.matshow(matrix_diag_blk.real)
        plt.matshow(matrix_upper_blk.real)
        plt.matshow(matrix_lower_blk.real)
        plt.show() """

        # Save matrices to file
        filename = "dense_blocks_matrix_" + str(i) + "_diagblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, matrix_diag_blk)
        filename = "dense_blocks_matrix_" + str(i) + "_upperblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, matrix_upper_blk)
        filename = "dense_blocks_matrix_" + str(i) + "_lowerblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, matrix_lower_blk)

        filename = "dense_blocks_matrix_" + str(i) + "_inverse_diagblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, matrix_inv_diag_blk)
        filename = "dense_blocks_matrix_" + str(i) + "_inverse_upperblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, matrix_inv_upper_blk)
        filename = "dense_blocks_matrix_" + str(i) + "_inverse_lowerblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, matrix_inv_lower_blk)
