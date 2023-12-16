import matrix_creation
import matrix_modification

import numpy as np
import pytest

@pytest.mark.parametrize(
        "blocksize",
        [(1),
         (2),
         (5),
         (10),
         (20),
         (50),
         (100),
        ]
)
def test_create_invertible_block(
    blocksize: int
):
    A = matrix_creation.create_invertible_block(blocksize)
    A_inverse = np.linalg.inv(A)
    assert np.linalg.norm(A_inverse @ A - np.eye(blocksize)) / np.linalg.norm(A) < 1e-10


@pytest.mark.parametrize(
        "matrix_size, blocksize",
        [(10, 1),
         (20, 2),
         (100, 1),
         (200, 2),
         (30, 5),
         (50, 10),
        ]
)
def test_create_tridiagonal_matrix(
        matrix_size: int,
        blocksize: int
):
    A = matrix_creation.create_tridiagonal_matrix(matrix_size, blocksize)
    A_inverse = np.linalg.inv(A)
    assert np.linalg.norm(A_inverse @ A - np.eye(matrix_size)) / np.linalg.norm(A) < 1e-10
    matrix_modification.assert_block_tridiagonal(A, blocksize)
