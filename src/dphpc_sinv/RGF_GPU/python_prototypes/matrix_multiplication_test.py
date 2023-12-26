import matrix_multiplication
import matrix_modification

import numpy as np
import pytest

seed = 10

@pytest.mark.parametrize(
        "matrix_size, blocksize",
        [(100, 1),
         (200, 2),
         (30, 5),
         (50, 10),
        ]
)
def test_mult_block_tridiagonal_dense_hermite(
        matrix_size: int,
        blocksize: int
):
    rng = np.random.default_rng(seed=seed)
    A = rng.uniform(size=(matrix_size, matrix_size)) + 1j * rng.uniform(size=(matrix_size, matrix_size))
    B = rng.uniform(size=(matrix_size, matrix_size)) + 1j * rng.uniform(size=(matrix_size, matrix_size))
    matrix_modification.cut_to_tridiag(A, blocksize)
    reference_mult = A @ B.T.conj()
    test_mult = matrix_multiplication.mult_block_tridiagonal_dense_hermite(A, B, blocksize)
    assert np.allclose(test_mult, reference_mult)


@pytest.mark.parametrize(
        "matrix_size, blocksize",
        [(100, 1),
         (200, 2),
         (30, 5),
         (50, 10),
        ]
)
def test_mult_selected_tridiagonal_dense_dense(
        matrix_size: int,
        blocksize: int
):
    rng = np.random.default_rng(seed=seed)
    A = rng.uniform(size=(matrix_size, matrix_size)) + 1j * rng.uniform(size=(matrix_size, matrix_size))
    B = rng.uniform(size=(matrix_size, matrix_size)) + 1j * rng.uniform(size=(matrix_size, matrix_size))
    reference_mult = A @ B
    matrix_modification.cut_to_tridiag(reference_mult, blocksize)
    test_mult = matrix_multiplication.mult_selected_tridiagonal_dense_dense(A, B, blocksize)
    assert np.allclose(test_mult, reference_mult)

@pytest.mark.parametrize(
        "matrix_size, blocksize",
        [(100, 1),
         (200, 2),
         (30, 5),
         (50, 10),
        ]
)
def test_mult_block_tridiagonal_dense(
        matrix_size: int,
        blocksize: int
):
    rng = np.random.default_rng(seed=seed)
    A = rng.uniform(size=(matrix_size, matrix_size)) + 1j * rng.uniform(size=(matrix_size, matrix_size))
    B = rng.uniform(size=(matrix_size, matrix_size)) + 1j * rng.uniform(size=(matrix_size, matrix_size))
    matrix_modification.cut_to_tridiag(A, blocksize)
    reference_mult = A @ B
    test_mult = matrix_multiplication.mult_block_tridiagonal_dense(A, B, blocksize)
    assert np.allclose(test_mult, reference_mult)

@pytest.mark.parametrize(
        "matrix_size, blocksize",
        [(10, 1),
         (20, 2),
         (30, 5),
         (50, 10),
        ]
)
def test_mult_selected_tridiagonal_dense_tridiagonal_dense_hermite(
        matrix_size: int,
        blocksize: int
):
    rng = np.random.default_rng(seed=seed)
    A = rng.uniform(size=(matrix_size, matrix_size)) + 1j * rng.uniform(size=(matrix_size, matrix_size))
    B = rng.uniform(size=(matrix_size, matrix_size)) + 1j * rng.uniform(size=(matrix_size, matrix_size))
    C = rng.uniform(size=(matrix_size, matrix_size)) + 1j * rng.uniform(size=(matrix_size, matrix_size))
    matrix_modification.cut_to_tridiag(B, blocksize)
    reference_mult = A @ B @ C.T.conj()
    matrix_modification.cut_to_tridiag(reference_mult, blocksize)
    test_mult = matrix_multiplication.mult_selected_tridiagonal_dense_tridiagonal_dense_hermite(
        A, B, C, blocksize)
    assert np.allclose(test_mult, reference_mult)
