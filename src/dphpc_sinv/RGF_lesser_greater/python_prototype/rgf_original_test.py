import rgf_original
import matrix_modification
import matrix_creation

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
def test_rgf_retarded(
        matrix_size: int,
        blocksize: int
):
    A = matrix_creation.create_tridiagonal_matrix(matrix_size, blocksize)
    reference_inverse = np.linalg.inv(A)
    assert np.linalg.norm(reference_inverse @ A- np.eye(matrix_size))/np.linalg.norm(A) < 1e-10
    matrix_modification.cut_to_tridiag(reference_inverse, blocksize)
    assert np.allclose(rgf_original.rgf_retarded(A, blocksize), reference_inverse)


@pytest.mark.parametrize(
        "matrix_size, blocksize",
        [(5, 1),
         (10, 1),
         (6, 2),
         (30, 5),
         (50, 10),
        ]
)
def test_rgf_lesser_greater_left_connected(
        matrix_size: int,
        blocksize: int
):
    rng = np.random.default_rng(seed=seed)
    B = rng.uniform(size=(matrix_size, matrix_size)) + 1j * rng.uniform(size=(matrix_size, matrix_size))
    B = (B - B.conj().T) / 2
    matrix_modification.cut_to_tridiag(B, blocksize)
    A = matrix_creation.create_tridiagonal_matrix(matrix_size, blocksize)
    inverse_A = np.linalg.inv(A)
    assert np.linalg.norm(inverse_A @ A- np.eye(matrix_size))/np.linalg.norm(A) < 1e-10
    assert np.allclose(B, -B.conj().T)

    matrix_modification.cut_to_tridiag(A, blocksize)
    matrix_modification.cut_to_tridiag(B, blocksize)

    reference_inverse = inverse_A @ B @ inverse_A.T.conj()
    matrix_modification.cut_to_diag(reference_inverse, blocksize)

    test_inverse = rgf_original.rgf_lesser_greater_left_connected(A, B, blocksize)

    matrix_modification.assert_block_diagonal(reference_inverse, blocksize)
    matrix_modification.assert_block_diagonal(test_inverse, blocksize)
    assert np.allclose(test_inverse[-blocksize:,-blocksize:], reference_inverse[-blocksize:,-blocksize:])
    assert np.allclose(reference_inverse, -reference_inverse.conj().T)
    assert np.allclose(test_inverse, -test_inverse.conj().T)
    assert np.allclose(test_inverse, reference_inverse)

@pytest.mark.parametrize(
        "matrix_size, blocksize",
        [(3, 1),
         (5, 1),
         (10, 1),
         (6, 2),
         (30, 5),
         (50, 10),
        ]
)
def test_rgf_lesser_greater_left_connected_tridiag(
        matrix_size: int,
        blocksize: int
):
    rng = np.random.default_rng(seed=seed)
    B = rng.uniform(size=(matrix_size, matrix_size)) + 1j * rng.uniform(size=(matrix_size, matrix_size))
    B = (B - B.conj().T) / 2
    matrix_modification.cut_to_tridiag(B, blocksize)
    A = matrix_creation.create_tridiagonal_matrix(matrix_size, blocksize)
    inverse_A = np.linalg.inv(A)
    assert np.linalg.norm(inverse_A @ A- np.eye(matrix_size))/np.linalg.norm(A) < 1e-10
    assert np.allclose(B, -B.conj().T)

    reference_inverse = inverse_A @ B @ inverse_A.T.conj()

    test_inverse = rgf_original.rgf_lesser_greater_left_connected_tridiag(A, B, blocksize)
    matrix_modification.cut_to_tridiag(reference_inverse, blocksize)

    matrix_modification.assert_block_tridiagonal(reference_inverse, blocksize)
    matrix_modification.assert_block_tridiagonal(test_inverse, blocksize)
    assert np.allclose(test_inverse[-blocksize:,-blocksize:], reference_inverse[-blocksize:,-blocksize:])
    assert np.allclose(reference_inverse, -reference_inverse.conj().T)
    assert np.allclose(test_inverse, -test_inverse.conj().T)
    assert np.allclose(test_inverse, reference_inverse)

@pytest.mark.parametrize(
        "matrix_size, blocksize",
        [(3, 1),
         (5, 1),
         (10, 1),
         (6, 2),
         (30, 5),
         (50, 10),
        ]
)
def test_rgf_lesser_greater_left_connected_tridiag_opt(
        matrix_size: int,
        blocksize: int
):
    rng = np.random.default_rng(seed=seed)
    B = rng.uniform(size=(matrix_size, matrix_size)) + 1j * rng.uniform(size=(matrix_size, matrix_size))
    B = (B - B.conj().T) / 2
    matrix_modification.cut_to_tridiag(B, blocksize)
    A = matrix_creation.create_tridiagonal_matrix(matrix_size, blocksize)
    inverse_A = np.linalg.inv(A)
    assert np.linalg.norm(inverse_A @ A- np.eye(matrix_size))/np.linalg.norm(A) < 1e-10
    assert np.allclose(B, -B.conj().T)

    reference_inverse = inverse_A @ B @ inverse_A.T.conj()

    test_inverse = rgf_original.rgf_lesser_greater_left_connected_tridiag_opt(A, B, blocksize)
    matrix_modification.cut_to_tridiag(reference_inverse, blocksize)

    matrix_modification.assert_block_tridiagonal(reference_inverse, blocksize)
    matrix_modification.assert_block_tridiagonal(test_inverse, blocksize)
    assert np.allclose(test_inverse[-blocksize:,-blocksize:], reference_inverse[-blocksize:,-blocksize:])
    assert np.allclose(reference_inverse, -reference_inverse.conj().T)
    assert np.allclose(test_inverse, -test_inverse.conj().T)
    assert np.allclose(test_inverse, reference_inverse)

@pytest.mark.parametrize(
        "matrix_size, blocksize",
        [(3, 1),
         (10, 1),
         (6, 2),
         (30, 5),
         (50, 10),
        ]
)
def test_rgf_lesser_greater_right_connected(
        matrix_size: int,
        blocksize: int
):
    rng = np.random.default_rng(seed=seed)
    B = rng.uniform(size=(matrix_size, matrix_size)) + 1j * rng.uniform(size=(matrix_size, matrix_size))
    B = (B - B.conj().T) / 2
    matrix_modification.cut_to_tridiag(B, blocksize)
    A = matrix_creation.create_tridiagonal_matrix(matrix_size, blocksize)
    inverse_A = np.linalg.inv(A)
    assert np.linalg.norm(inverse_A @ A- np.eye(matrix_size))/np.linalg.norm(A) < 1e-10
    assert np.allclose(B, -B.conj().T)

    matrix_modification.cut_to_tridiag(A, blocksize)
    matrix_modification.cut_to_tridiag(B, blocksize)

    reference_inverse = inverse_A @ B @ inverse_A.T.conj()
    matrix_modification.cut_to_diag(reference_inverse, blocksize)

    test_retarded, test_lesser_greater = rgf_original.rgf_lesser_greater_right_connected(A, B, blocksize)

    matrix_modification.assert_block_diagonal(reference_inverse, blocksize)
    matrix_modification.assert_block_diagonal(test_lesser_greater, blocksize)
    matrix_modification.cut_to_diag(inverse_A, blocksize)

    assert np.allclose(test_retarded, inverse_A)
    assert np.allclose(test_lesser_greater[:blocksize,:blocksize], reference_inverse[:blocksize,:blocksize])
    assert np.allclose(reference_inverse, -reference_inverse.conj().T)
    assert np.allclose(test_lesser_greater, -test_lesser_greater.conj().T)
    assert np.allclose(test_lesser_greater, reference_inverse)


@pytest.mark.parametrize(
        "matrix_size, blocksize",
        [(3, 1),
         (10, 1),
         (6, 2),
         (30, 5),
         (50, 10),
        ]
)
def test_rgf_lesser_greater_right_connected_tridiag(
        matrix_size: int,
        blocksize: int
):
    rng = np.random.default_rng(seed=seed)
    B = rng.uniform(size=(matrix_size, matrix_size)) + 1j * rng.uniform(size=(matrix_size, matrix_size))
    B = (B - B.conj().T) / 2
    matrix_modification.cut_to_tridiag(B, blocksize)
    A = matrix_creation.create_tridiagonal_matrix(matrix_size, blocksize)
    inverse_A = np.linalg.inv(A)
    assert np.linalg.norm(inverse_A @ A- np.eye(matrix_size))/np.linalg.norm(A) < 1e-10
    assert np.allclose(B, -B.conj().T)

    matrix_modification.cut_to_tridiag(A, blocksize)
    matrix_modification.cut_to_tridiag(B, blocksize)

    reference_inverse = inverse_A @ B @ inverse_A.T.conj()
    matrix_modification.cut_to_tridiag(reference_inverse, blocksize)

    test_retarded, test_lesser_greater = rgf_original.rgf_lesser_greater_right_connected_tridiag(A, B, blocksize)

    # matrix_modification.assert_block_diagonal(reference_inverse, blocksize)
    # matrix_modification.assert_block_diagonal(test_lesser_greater, blocksize)
    matrix_modification.cut_to_diag(inverse_A, blocksize)

    assert np.allclose(test_retarded, inverse_A)
    assert np.allclose(test_lesser_greater[:blocksize,:blocksize], reference_inverse[:blocksize,:blocksize])
    # assert np.allclose(reference_inverse, -reference_inverse.conj().T)
    # assert np.allclose(test_lesser_greater, -test_lesser_greater.conj().T)
    assert np.allclose(test_lesser_greater, reference_inverse)

