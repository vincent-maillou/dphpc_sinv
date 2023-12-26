import numpy as np

seed = 10

def create_invertible_block(
    blocksize: int,
    seedd: int = seed
):
    rng = np.random.default_rng(seed=seedd)
    A = rng.uniform(size=(blocksize, blocksize)) + 1j * rng.uniform(size=(blocksize, blocksize))
    value_diag = np.sum(np.abs(A), axis=1)
    np.fill_diagonal(A, value_diag)
    return A

def create_tridiagonal_matrix(
        matrix_size: int,
        blocksize: int
):
    A = np.zeros((matrix_size, matrix_size), dtype=np.complex128)
    assert matrix_size % blocksize == 0
    number_of_blocks = int(matrix_size / blocksize)
    for i in range(number_of_blocks):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        A[i_, i_] = create_invertible_block(blocksize, seedd=seed+i)
    for i in range(number_of_blocks-1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_plus_one_ = slice((i+1)*blocksize, (i+2)*blocksize)
        A[i_, i_plus_one_] = np.random.rand(blocksize, blocksize) + 1j * np.random.rand(blocksize, blocksize)
        A[i_plus_one_, i_] = np.random.rand(blocksize, blocksize) + 1j * np.random.rand(blocksize, blocksize)
    return A
