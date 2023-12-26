import matrix_modification

import numpy as np
import pytest

seed = 10

@pytest.mark.parametrize(
        "matrix_size, blocksize",
        [(10, 1),
         (20, 2),
         (30, 5),
         (50, 10),
        ]
)
def test_cut_to_diag(
        matrix_size: int,
        blocksize: int
):
    rng = np.random.default_rng(seed=seed)
    A = rng.uniform(size=(matrix_size, matrix_size)) + 1j * rng.uniform(size=(matrix_size, matrix_size))
    matrix_modification.cut_to_diag(A, blocksize)
    matrix_modification.assert_block_diagonal(A, blocksize)

@pytest.mark.parametrize(
        "matrix_size, blocksize",
        [(10, 1),
         (20, 2),
         (30, 5),
         (50, 10),
        ]
)
def test_cut_to_tridiag(
        matrix_size: int,
        blocksize: int
):
    rng = np.random.default_rng(seed=seed)
    A = rng.uniform(size=(matrix_size, matrix_size)) + 1j * rng.uniform(size=(matrix_size, matrix_size))
    matrix_modification.cut_to_tridiag(A, blocksize)
    matrix_modification.assert_block_tridiagonal(A, blocksize)

@pytest.mark.parametrize(
        "matrix_size, blocksize",
        [(10, 1),
         (20, 2),
         (30, 5),
         (50, 10),
        ]
)
def test_cut_to_upper_half(
        matrix_size: int,
        blocksize: int
):
    number_of_blocks = int(matrix_size / blocksize)
    rng = np.random.default_rng(seed=seed)
    A = rng.uniform(size=(matrix_size, matrix_size)) + 1j * rng.uniform(size=(matrix_size, matrix_size))
    matrix_modification.cut_to_upper_half(A, blocksize)
    zero_block = np.zeros((blocksize, blocksize))
    for i in range(number_of_blocks):
        for j in range(number_of_blocks):
            if i > j:
                assert np.allclose(A[i*blocksize:(i+1)*blocksize, j*blocksize:(j+1)*blocksize], zero_block)

@pytest.mark.parametrize(
        "matrix_size, blocksize",
        [(10, 1),
         (20, 2),
         (30, 5),
         (50, 10),
        ]
)
def test_cut_to_lower_half(
        matrix_size: int,
        blocksize: int
):
    number_of_blocks = int(matrix_size / blocksize)
    rng = np.random.default_rng(seed=seed)
    A = rng.uniform(size=(matrix_size, matrix_size)) + 1j * rng.uniform(size=(matrix_size, matrix_size))
    matrix_modification.cut_to_lower_half(A, blocksize)
    zero_block = np.zeros((blocksize, blocksize))
    for i in range(number_of_blocks):
        for j in range(number_of_blocks):
            if i < j:
                assert np.allclose(A[i*blocksize:(i+1)*blocksize, j*blocksize:(j+1)*blocksize], zero_block)
