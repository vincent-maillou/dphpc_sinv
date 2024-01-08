# Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.sparse import csr_matrix
import os

def generate_random_matrix(
    matrice_size: int,
    seed: int = None,
):
    np.random.seed(seed)
    matrix = np.random.rand(matrice_size, matrice_size) + 1j * \
        np.random.rand(matrice_size, matrice_size)

    value_diag = np.sum(np.abs(matrix), axis=1)
    np.fill_diagonal(matrix, value_diag)

    return matrix


def generateBandedMatrix(
    matrice_size: int, 
    bandwidth: int,
    seed: int = None
) -> np.ndarray:
    """ Generate a banded diagonal matrix of shape: matrice_size^2 with a 
    bandwidth = matrice_bandwidth, filled with random numbers.

    Parameters
    ----------
    matrice_size : int
        Size of the matrice to generate.
    matrice_bandwidth : int
        Bandwidth of the matrice to generate.
    seed : int, optional
        Seed for the random number generator. The default is no seed.
        
    Returns
    -------
    A : np.ndarray
        The generated matrice.
    """
    np.random.seed(seed)
    A = np.random.rand(matrice_size, matrice_size) + 1j*np.random.rand(matrice_size, matrice_size)
    
    for i in range(matrice_size):
        for j in range(matrice_size):
            if i - j >= bandwidth or j - i >= bandwidth:
                A[i, j] = 0

    return A


def sparsifyMatrix(
    matrice: np.ndarray,
    sparsity: float,
    seed: int = None
) -> np.ndarray:
    """ Sparsify a matrice by setting sparsity% of its entries to zero.

    Parameters
    ----------
    matrice : np.ndarray
        The matrice to sparsify.
    sparsity : float
        Sparsity of the matrice to generate.
    seed : int, optional
        Seed for the random number generator. The default is no seed.
        
    Returns
    -------
    A : np.ndarray
        The sparsified matrice.
    """
    np.random.seed(seed)
    A = matrice.copy()
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.random.rand() > sparsity:
                A[i, j] = 0

    return A


def generateSparseBandedMatrix(
    matrice_size: int,
    bandwidth: int,
    sparsity: float,
    seed: int = None
) -> np.ndarray:
    """ Generate a sparse banded diagonal matrix of shape: matrice_size^2 with a 
    bandwidth = matrice_bandwidth, filled with random numbers.

    Parameters
    ----------
    matrice_size : int
        Size of the matrice to generate.
    matrice_bandwidth : int
        Bandwidth of the matrice to generate.
    sparsity : float
        Sparsity of the matrice to generate.
    seed : int, optional
        Seed for the random number generator. The default is no seed.
        
    Returns
    -------
    A : np.ndarray
        The generated matrice.
    """
    A = generateBandedMatrix(matrice_size, bandwidth, seed)
    
    A = sparsifyMatrix(A, sparsity, seed)

    # assert that the matrix will be invertible
    value_diag = np.sum(np.abs(A), axis=1)
    np.fill_diagonal(A, value_diag)

    return A


def write_matrix_to_file(
    path_to_file: str,
    matrix: np.ndarray,
):
    with open(path_to_file, "wb") as f:
        f.write(matrix.tobytes())

def write_dense_matrix_to_sparse_blocks_file(
    path_to_file: str,
    matrix: np.ndarray,
    blocksize: int
):
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.shape[0] % blocksize == 0
    matrix_size = matrix.shape[0]
    number_of_blocks = matrix_size // blocksize

    if not os.path.exists(path_to_file):
        os.mkdir(path_to_file)

    save_path = os.path.join(path_to_file, "sparse_matrices_"+str(matrix_size)+"_"+str(blocksize)+"/")
    if not os.path.exists(save_path):
        os.mkdir(save_path)


    diag_nnz = []
    for i in range(number_of_blocks):
        sparse_block_diag = csr_matrix(matrix[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize])
        diag_nnz.append(sparse_block_diag.nnz)
        with open(os.path.join(save_path, "diag_data" + str(i) + ".bin"), "wb") as f:
            f.write(sparse_block_diag.data.tobytes())
        with open(os.path.join(save_path, "diag_indices" + str(i) + ".bin"), "wb") as f:
            f.write(sparse_block_diag.indices.tobytes())
        with open(os.path.join(save_path, "diag_indptr" + str(i) + ".bin"), "wb") as f:
            f.write(sparse_block_diag.indptr.tobytes())

    upper_nnz = []
    lower_nnz = []
    for i in range(number_of_blocks-1):
        sparse_block_upper = csr_matrix(matrix[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize])
        upper_nnz.append(sparse_block_upper.nnz)
        with open(os.path.join(save_path, "upper_data" + str(i) + ".bin"), "wb") as f:
            f.write(sparse_block_upper.data.tobytes())
        with open(os.path.join(save_path, "upper_indices" + str(i) + ".bin"), "wb") as f:
            f.write(sparse_block_upper.indices.tobytes())
        with open(os.path.join(save_path, "upper_indptr" + str(i) + ".bin"), "wb") as f:
            f.write(sparse_block_upper.indptr.tobytes())
        sparse_block_lower = csr_matrix(matrix[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize])
        lower_nnz.append(sparse_block_lower.nnz)
        with open(os.path.join(save_path, "lower_data" + str(i) + ".bin"), "wb") as f:
            f.write(sparse_block_lower.data.tobytes())
        with open(os.path.join(save_path, "lower_indices" + str(i) + ".bin"), "wb") as f:
            f.write(sparse_block_lower.indices.tobytes())
        with open(os.path.join(save_path, "lower_indptr" + str(i) + ".bin"), "wb") as f:
            f.write(sparse_block_lower.indptr.tobytes())
    print(diag_nnz)
    print(save_path)
    with open(os.path.join(save_path, "diag_nnz.txt"), "w", encoding="utf-8") as f:
        for i in range(number_of_blocks):
            f.write(str(diag_nnz[i]) + "\n")
    with open(os.path.join(save_path, "upper_nnz.txt"), "w", encoding="utf-8") as f:
        for i in range(number_of_blocks-1):
            f.write(str(upper_nnz[i]) + "\n")
    with open(os.path.join(save_path, "lower_nnz.txt"), "w", encoding="utf-8") as f:
        for i in range(number_of_blocks-1):
            f.write(str(lower_nnz[i]) + "\n")
    print("Diag nnz: ", diag_nnz)
    print("Upper nnz: ", upper_nnz)
    print("Lower nnz: ", lower_nnz)


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

def write_matrix_parameters_batched(
    path_to_file: str,
    matrix_size: int,
    blocksize: int,
    batchsize: int
):
    with open(path_to_file, "w") as f:
        f.write(str(matrix_size) + "\n")
        f.write(str(blocksize) + "\n")
        f.write(str(batchsize) + "\n")
