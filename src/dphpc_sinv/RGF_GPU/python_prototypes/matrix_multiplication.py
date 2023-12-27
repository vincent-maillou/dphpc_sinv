import numpy as np


def mult_selected_tridiagonal_dense_tridiagonal_dense_hermite(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    blocksize: int
):
    assert B.shape[0] % blocksize == 0
    assert B.shape[1] % blocksize == 0
    number_of_blocks = int(B.shape[0] / blocksize)
    D = np.zeros_like(B)
    E = np.zeros_like(B)

    # # block tridigonal time dense matrix multiplication
    # for i in range(number_of_blocks):
    #     i_ = slice(i*blocksize, (i+1)*blocksize)
    #     for j in range(number_of_blocks):
    #         j_ = slice(j*blocksize, (j+1)*blocksize)
    #         for k in range(max(i-1,0), min(i+2,number_of_blocks)):
    #             k_ = slice(k*blocksize, (k+1)*blocksize)
    #             D[i_,j_] += B[i_,k_] @ C[j_,k_].conj().T

    # # block tridigonal time dense matrix multiplication
    # for i in range(number_of_blocks):
    #     i_ = slice(i*blocksize, (i+1)*blocksize)
    #     for j in range(max(i-1,0),min(i+1, number_of_blocks - 1)+1):
    #         j_ = slice(j*blocksize, (j+1)*blocksize)
    #         for k in range(number_of_blocks):
    #             k_ = slice(k*blocksize, (k+1)*blocksize)
    #             E[i_,j_] += A[i_,k_] @ D[k_,j_]

    for i in range(number_of_blocks):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        for j in range(max(i-1,0),min(i+1, number_of_blocks - 1)+1):
            j_ = slice(j*blocksize, (j+1)*blocksize)
            for k in range(number_of_blocks):
                k_ = slice(k*blocksize, (k+1)*blocksize)
                for g in range(max(k-1,0),min(k+1, number_of_blocks - 1)+1):
                    g_ = slice(g*blocksize, (g+1)*blocksize)
                    E[i_,j_] += A[i_,k_] @ B[k_,g_] @ C[j_,g_].conj().T

    return E


def mult_selected_tridiagonal_dense_dense(
    A: np.ndarray,
    B: np.ndarray,
    blocksize: int
):
    assert B.shape[0] % blocksize == 0
    assert B.shape[1] % blocksize == 0
    number_of_blocks = int(B.shape[0] / blocksize)
    C = np.zeros_like(B)
    # block tridigonal time dense matrix multiplication
    for i in range(number_of_blocks):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        for j in range(max(i-1,0),min(i+1, number_of_blocks - 1)+1):
            j_ = slice(j*blocksize, (j+1)*blocksize)
            for k in range(number_of_blocks):
                k_ = slice(k*blocksize, (k+1)*blocksize)
                C[i_,j_] += A[i_,k_] @ B[k_,j_]
    return C


def mult_block_tridiagonal_dense_hermite(
    A: np.ndarray,
    B: np.ndarray,
    blocksize: int
):
    assert B.shape[0] % blocksize == 0
    assert B.shape[1] % blocksize == 0
    number_of_blocks = int(B.shape[0] / blocksize)
    C = np.zeros_like(B)
    # block tridigonal time dense matrix multiplication
    for i in range(number_of_blocks):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        for j in range(number_of_blocks):
            j_ = slice(j*blocksize, (j+1)*blocksize)
            for k in range(max(i-1,0), min(i+2,number_of_blocks)):
                k_ = slice(k*blocksize, (k+1)*blocksize)
                C[i_,j_] += A[i_,k_] @ B[j_,k_].conj().T
    return C


def mult_block_tridiagonal_dense(
    A: np.ndarray,
    B: np.ndarray,
    blocksize: int
):
    assert B.shape[0] % blocksize == 0
    assert B.shape[1] % blocksize == 0
    number_of_blocks = int(B.shape[0] / blocksize)
    C = np.zeros_like(B)
    # block tridigonal time dense matrix multiplication
    for i in range(number_of_blocks):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        for j in range(number_of_blocks):
            j_ = slice(j*blocksize, (j+1)*blocksize)
            for k in range(max(i-1,0), min(i+2,number_of_blocks)):
                k_ = slice(k*blocksize, (k+1)*blocksize)
                C[i_,j_] += A[i_,k_] @ B[k_,j_]
    return C
