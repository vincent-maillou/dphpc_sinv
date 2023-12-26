import numpy as np

def assert_block_tridiagonal(
        A: np.ndarray,
        blocksize: int
):
    number_of_blocks = int(A.shape[0] / blocksize)
    zero_block = np.zeros((blocksize, blocksize))
    for row in range(number_of_blocks):
        for col in range(number_of_blocks):
            if col < row - 1 or col > row + 1:
                assert np.allclose(A[row*blocksize:(row+1)*blocksize, col*blocksize:(col+1)*blocksize], zero_block)

def assert_block_diagonal(
        A: np.ndarray,
        blocksize: int
):
    number_of_blocks = int(A.shape[0] / blocksize)
    zero_block = np.zeros((blocksize, blocksize))
    for row in range(number_of_blocks):
        for col in range(number_of_blocks):
            if col != row:
                assert np.allclose(A[row*blocksize:(row+1)*blocksize, col*blocksize:(col+1)*blocksize], zero_block)


def cut_to_diag(
    A: np.ndarray,
    blocksize : int
):
    number_of_blocks = int(A.shape[0] / blocksize)
    zero_block = np.zeros((blocksize, blocksize))
    for row in range(number_of_blocks):
        for col in range(number_of_blocks):
            if col != row:
                A[row*blocksize:(row+1)*blocksize, col*blocksize:(col+1)*blocksize] = zero_block


def cut_to_tridiag(
    A: np.ndarray,
    blocksize : int
):
    # Delete elements of A that are outside of the tridiagonal block structure
    number_of_blocks = int(A.shape[0] / blocksize)
    zero_block = np.zeros((blocksize, blocksize))
    for row in range(number_of_blocks):
        for col in range(number_of_blocks):
            if col < row - 1 or col > row + 1:
                A[row*blocksize:(row+1)*blocksize, col*blocksize:(col+1)*blocksize] = zero_block

def cut_to_upper_half(
    A: np.ndarray,
    blocksize : int
):
    number_of_blocks = int(A.shape[0] / blocksize)
    zero_block = np.zeros((blocksize, blocksize))
    for row in range(number_of_blocks):
        for col in range(number_of_blocks):
            if col < row:
                A[row*blocksize:(row+1)*blocksize, col*blocksize:(col+1)*blocksize] = zero_block

def cut_to_lower_half(
    A: np.ndarray,
    blocksize : int
):
    number_of_blocks = int(A.shape[0] / blocksize)
    zero_block = np.zeros((blocksize, blocksize))
    for row in range(number_of_blocks):
        for col in range(number_of_blocks):
            if col > row:
                A[row*blocksize:(row+1)*blocksize, col*blocksize:(col+1)*blocksize] = zero_block

