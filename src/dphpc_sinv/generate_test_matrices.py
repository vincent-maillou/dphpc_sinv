"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-09

Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

SEED = 63
MAT_SIZE = 10
BLOCKSIZE = 2


def generate_random_matrix():
    np.random.seed(SEED)
    matrix = np.random.rand(MAT_SIZE, MAT_SIZE) + 1j * \
        np.random.rand(MAT_SIZE, MAT_SIZE)

    value_diag = np.sum(np.abs(matrix), axis=1)
    np.fill_diagonal(matrix, value_diag)

    return matrix


def write_matrix_to_file(
    path_to_file: str,
    matrix: np.ndarray,
    matrix_size: int,
    blocksize: int,
):
    with open(path_to_file, "wb") as f:
        f.write(matrix.tobytes())


def print_matrix(
    matrix: np.ndarray,
    matrix_size: int
):
    for i in range(matrix_size):
        for j in range(matrix_size):
            print(matrix[i, j], end=" ")


def show_matrix(
    matrix: np.ndarray,
    matrix_to_compare_to: np.ndarray = None,
):
    plt.matshow(matrix.real)
    if matrix_to_compare_to is not None:
        plt.matshow(matrix_to_compare_to.real)
    plt.show()


def extract_diagonal_blocks(
    matrix: np.ndarray,
    matrix_size: int,
    blocksize: int,
) -> np.ndarray:
    """ Extract the diagonal blocks of a dense matrix and store them in a 
    contiguous array.
    """
    num_blocks = matrix_size // blocksize
    blocks = np.empty((blocksize, num_blocks*blocksize), dtype=matrix.dtype)
    for i in range(num_blocks):
        row_start = i * blocksize
        row_end = row_start + blocksize
        col_start = i * blocksize
        col_end = col_start + blocksize
        blocks[0:blocksize, row_start:row_end] = matrix[row_start:row_end,
                                                        col_start:col_end]
    return blocks


def extract_offdiagonal_blocks(
    matrix: np.ndarray,
    matrix_size: int,
    blocksize: int,
    n_off_diag: int,
) -> np.ndarray:
    """ Extract the n_off_diag off-diagonal blocks of a dense matrix and store
    them in a contiguous array.
    """
    num_blocks = matrix_size // blocksize - abs(n_off_diag)
    blocks = np.empty((blocksize, num_blocks*blocksize), dtype=matrix.dtype)

    if n_off_diag > 0:
        for i in range(num_blocks):
            row_start = i * blocksize
            row_end = row_start + blocksize
            col_start = (i + n_off_diag) * blocksize
            col_end = col_start + blocksize
            blocks[0:blocksize, row_start:row_end] = matrix[row_start:row_end,
                                                            col_start:col_end]
    else:
        for i in range(num_blocks):
            row_start = (i - n_off_diag) * blocksize
            row_end = row_start + blocksize
            col_start = i * blocksize
            col_end = col_start + blocksize
            blocks[0:blocksize, col_start:col_end] = matrix[row_start:row_end,
                                                            col_start:col_end]

    return blocks


def write_matrix_parameters(
    path_to_file: str,
    matrix_size: int,
    blocksize: int,
):
    with open(path_to_file, "w") as f:
        f.write(str(matrix_size) + "\n")
        f.write(str(blocksize) + "\n")


if __name__ == "__main__":
    # Generate random matrix
    matrix = generate_random_matrix()

    # assert matrix to be invertible
    assert np.allclose( np.linalg.inv(matrix) @ matrix, np.eye(MAT_SIZE) )
    assert np.linalg.det(matrix) != 0


    # Compute inverse
    inv_matrix = np.linalg.inv(matrix)

    # Extract diagonal and off-diagonal blocks
    matrix_diag_blk = extract_diagonal_blocks(matrix, MAT_SIZE, BLOCKSIZE)
    matrix_upper_blk = extract_offdiagonal_blocks(
        matrix, MAT_SIZE, BLOCKSIZE, 1)
    matrix_lower_blk = extract_offdiagonal_blocks(
        matrix, MAT_SIZE, BLOCKSIZE, -1)

    matrix_inv_diag_blk = extract_diagonal_blocks(inv_matrix, MAT_SIZE, BLOCKSIZE)
    matrix_inv_upper_blk = extract_offdiagonal_blocks(
        inv_matrix, MAT_SIZE, BLOCKSIZE, 1)
    matrix_inv_lower_blk = extract_offdiagonal_blocks(
        inv_matrix, MAT_SIZE, BLOCKSIZE, -1)

    # # Save matrices to file
    # path_to_file = "../../tests/tests_cases/"

    # # create test folder if it does not exist
    # if not os.path.exists(path_to_file):
    #     os.makedirs(path_to_file)

    print(matrix)
    print(matrix_diag_blk)
    print(matrix_upper_blk)
    print(matrix_lower_blk)

    filename = "matrix_0_diagblk.bin"
    write_matrix_to_file(path_to_file+filename, matrix_diag_blk, MAT_SIZE, 1)
    filename = "matrix_0_upperblk.bin"
    write_matrix_to_file(path_to_file+filename, matrix_upper_blk, MAT_SIZE, 1)
    filename = "matrix_0_lowerblk.bin"
    write_matrix_to_file(path_to_file+filename, matrix_lower_blk, MAT_SIZE, 1)

    filename = "matrix_0_inverse_diagblk.bin"
    write_matrix_to_file(path_to_file+filename, matrix_inv_diag_blk, MAT_SIZE, 1)
    filename = "matrix_0_inverse_upperblk.bin"
    write_matrix_to_file(path_to_file+filename, matrix_inv_upper_blk, MAT_SIZE, 1)
    filename = "matrix_0_inverse_lowerblk.bin"
    write_matrix_to_file(path_to_file+filename, matrix_inv_lower_blk, MAT_SIZE, 1)

    filename = "mat_parameters_0.txt"
    write_matrix_parameters(path_to_file+filename, MAT_SIZE, BLOCKSIZE)
