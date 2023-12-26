import numpy as np

def rgf_retarded(
    System_matrix: np.ndarray,
    blocksize: int
):
    # Storage for the full backward substitution
    G_retarded = np.zeros_like(System_matrix)
    g_retarded = np.zeros_like(System_matrix)
    assert System_matrix.shape[0] % blocksize == 0
    assert System_matrix.shape[1] % blocksize == 0
    number_of_blocks = int(System_matrix.shape[0] / blocksize)

    # 0. Inverse of the first block
    g_retarded[0:blocksize, 0:blocksize] = np.linalg.inv(System_matrix[0:blocksize, 0:blocksize])

    # 1. Forward substitution (performed left to right)
    for i in range(1, number_of_blocks, 1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_minus_one_ = slice((i-1)*blocksize, i*blocksize) 
        g_retarded[i_, i_] = np.linalg.inv(System_matrix[i_, i_] - System_matrix[i_, i_minus_one_] @ g_retarded[i_minus_one_, i_minus_one_] @ System_matrix[i_minus_one_, i_])

    G_retarded[-blocksize:,-blocksize:] = g_retarded[-blocksize:,-blocksize:]
    # 2. Backward substitution (performed right to left)
    for i in range(number_of_blocks-2, -1, -1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_plus_one_ = slice((i+1)*blocksize, (i+2)*blocksize)
        g_ii = g_retarded[i_, i_]
        G_lowerfactor = G_retarded[i_plus_one_, i_plus_one_] @ System_matrix[i_plus_one_, i_] @ g_ii
        G_retarded[i_plus_one_, i_] = -G_lowerfactor
        G_retarded[i_, i_plus_one_] = -g_ii @ System_matrix[i_, i_plus_one_] @ G_retarded[i_plus_one_, i_plus_one_]
        G_retarded[i_, i_]   =  g_ii + g_ii @ System_matrix[i_, i_plus_one_] @ G_lowerfactor

    return G_retarded


def rgf_lesser_greater_left_connected(
    System_matrix: np.ndarray,
    Sigma_lesser_greater: np.ndarray,
    blocksize: int
):
    # Storage for the full backward substitution
    G_lesser_greater = np.zeros_like(System_matrix)
    G_retarded = np.zeros_like(System_matrix)

    g_retarded = np.zeros_like(System_matrix)
    g_lesser_greater = np.zeros_like(System_matrix)
    assert System_matrix.shape[0] % blocksize == 0
    assert System_matrix.shape[1] % blocksize == 0
    assert Sigma_lesser_greater.shape[0] % blocksize == 0
    assert Sigma_lesser_greater.shape[1] % blocksize == 0
    assert np.allclose(Sigma_lesser_greater, -Sigma_lesser_greater.conj().T)
    number_of_blocks = int(System_matrix.shape[0] / blocksize)

    # 0. Inverse of the first block retarded
    g_retarded[0:blocksize, 0:blocksize] = \
        np.linalg.inv(System_matrix[0:blocksize, 0:blocksize])

    # 1. Forward substitution (performed left to right)
    for i in range(1, number_of_blocks, 1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_minus_one_ = slice((i-1)*blocksize, i*blocksize) 
        g_retarded[i_, i_] = np.linalg.inv(System_matrix[i_, i_] - System_matrix[i_, i_minus_one_] @ g_retarded[i_minus_one_, i_minus_one_] @ System_matrix[i_minus_one_, i_])

    G_retarded[-blocksize:,-blocksize:] = g_retarded[-blocksize:,-blocksize:]
    # 2. Backward substitution (performed right to left)
    for i in range(number_of_blocks-2, -1, -1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_plus_one_ = slice((i+1)*blocksize, (i+2)*blocksize)
        g_ii = g_retarded[i_, i_]
        G_lowerfactor = G_retarded[i_plus_one_, i_plus_one_] @ System_matrix[i_plus_one_, i_] @ g_ii
        G_retarded[i_plus_one_, i_] = -G_lowerfactor
        G_retarded[i_, i_plus_one_] = -g_ii @ System_matrix[i_, i_plus_one_] @ G_retarded[i_plus_one_, i_plus_one_]
        G_retarded[i_, i_]   =  g_ii + g_ii @ System_matrix[i_, i_plus_one_] @ G_lowerfactor

    # 3. Forward substitution lesser greater
    g_lesser_greater[0:blocksize, 0:blocksize] =\
        g_retarded[0:blocksize, 0:blocksize] @ \
        Sigma_lesser_greater[0:blocksize, 0:blocksize] @ \
        g_retarded[0:blocksize, 0:blocksize].conj().T


    for i in range(1, number_of_blocks):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_minus_one_ = slice((i-1)*blocksize, i*blocksize)
        g_lesser_greater[i_, i_] = g_retarded[i_, i_]  @ \
            (Sigma_lesser_greater[i_, i_]\
            + \
            (System_matrix[i_,i_minus_one_] @ \
            g_lesser_greater[i_minus_one_, i_minus_one_] @ \
            System_matrix.conj().T[i_minus_one_,i_])\
            - \
            (Sigma_lesser_greater[i_,i_minus_one_] @ \
            g_retarded.conj().T[i_minus_one_, i_minus_one_] @ \
            System_matrix.conj().T[i_minus_one_,i_])\
            - \
            (System_matrix[i_,i_minus_one_] @ \
            g_retarded[i_minus_one_, i_minus_one_] @ \
            Sigma_lesser_greater[i_minus_one_, i_])\
            ) @ \
            g_retarded.conj().T[i_, i_]


    G_lesser_greater[-blocksize:,-blocksize:] = g_lesser_greater[-blocksize:,-blocksize:]

    # 4. Backward substitution lesser greater
    for i in range(number_of_blocks-2, -1, -1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_plus_one_ = slice((i+1)*blocksize, (i+2)*blocksize)

        AL = g_retarded[i_,i_] @\
            Sigma_lesser_greater[i_,i_plus_one_] @\
            G_retarded[i_plus_one_,i_plus_one_].T.conj() @\
            System_matrix[i_,i_plus_one_].T.conj() @\
            g_retarded[i_,i_].T.conj()

        BL = g_retarded[i_,i_] @\
            System_matrix[i_,i_plus_one_] @\
            G_retarded[i_plus_one_,i_plus_one_] @\
            System_matrix[i_plus_one_,i_] @\
            g_lesser_greater[i_,i_]

        G_lesser_greater[i_,i_] = g_lesser_greater[i_,i_] +\
                            g_retarded[i_,i_] @\
                            System_matrix[i_,i_plus_one_] @\
                            G_lesser_greater[i_plus_one_,i_plus_one_] @\
                            System_matrix[i_,i_plus_one_].T.conj() @\
                            g_retarded[i_,i_].T.conj() -\
                            (AL - AL.T.conj()) + (BL - BL.T.conj())



    return G_lesser_greater


def rgf_lesser_greater_left_connected_tridiag(
    System_matrix: np.ndarray,
    Sigma_lesser_greater: np.ndarray,
    blocksize: int
):
    # Storage for the full backward substitution
    G_lesser_greater = np.zeros_like(System_matrix)
    G_retarded = np.zeros_like(System_matrix)

    g_retarded = np.zeros_like(System_matrix)
    g_lesser_greater = np.zeros_like(System_matrix)
    assert System_matrix.shape[0] % blocksize == 0
    assert System_matrix.shape[1] % blocksize == 0
    assert Sigma_lesser_greater.shape[0] % blocksize == 0
    assert Sigma_lesser_greater.shape[1] % blocksize == 0
    # assert np.allclose(Sigma_lesser_greater, -Sigma_lesser_greater.conj().T)
    number_of_blocks = int(System_matrix.shape[0] / blocksize)

    # 0. Inverse of the first block retarded
    g_retarded[0:blocksize, 0:blocksize] = \
        np.linalg.inv(System_matrix[0:blocksize, 0:blocksize])

    # 1. Forward substitution (performed left to right)
    for i in range(1, number_of_blocks, 1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_minus_one_ = slice((i-1)*blocksize, i*blocksize) 
        g_retarded[i_, i_] = np.linalg.inv(System_matrix[i_, i_] - System_matrix[i_, i_minus_one_] @ g_retarded[i_minus_one_, i_minus_one_] @ System_matrix[i_minus_one_, i_])

    G_retarded[-blocksize:,-blocksize:] = g_retarded[-blocksize:,-blocksize:]
    # 2. Backward substitution (performed right to left)
    for i in range(number_of_blocks-2, -1, -1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_plus_one_ = slice((i+1)*blocksize, (i+2)*blocksize)
        G_lowerfactor = G_retarded[i_plus_one_, i_plus_one_] @ System_matrix[i_plus_one_, i_] @ g_retarded[i_, i_]
        G_retarded[i_, i_]   =  g_retarded[i_, i_] + g_retarded[i_, i_] @ System_matrix[i_, i_plus_one_] @ G_lowerfactor

    # 3. Forward substitution lesser greater
    g_lesser_greater[0:blocksize, 0:blocksize] =\
        g_retarded[0:blocksize, 0:blocksize] @ \
        Sigma_lesser_greater[0:blocksize, 0:blocksize] @ \
        g_retarded[0:blocksize, 0:blocksize].conj().T


    for i in range(1, number_of_blocks):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_minus_one_ = slice((i-1)*blocksize, i*blocksize)

        g_lesser_greater[i_, i_] = (
            g_retarded[i_, i_]
            @ (
                Sigma_lesser_greater[i_, i_]
                + System_matrix[i_, i_minus_one_] @
                    g_lesser_greater[i_minus_one_, i_minus_one_] @
                    System_matrix[i_, i_minus_one_].conj().T
                - Sigma_lesser_greater[i_, i_minus_one_] @
                    g_retarded[i_minus_one_, i_minus_one_].conj().T @
                    System_matrix[i_, i_minus_one_].conj().T
                - System_matrix[i_, i_minus_one_] @
                    g_retarded[i_minus_one_, i_minus_one_] @
                    Sigma_lesser_greater[i_minus_one_, i_]
            )
            @ g_retarded[i_, i_].conj().T
        )

    G_lesser_greater[-blocksize:,-blocksize:] = g_lesser_greater[-blocksize:,-blocksize:]

    # 4. Backward substitution lesser greater
    for i in range(number_of_blocks-2, -1, -1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_plus_one_ = slice((i+1)*blocksize, (i+2)*blocksize)

        temp_1 = (
            g_retarded[i_, i_]
            @ (
                Sigma_lesser_greater[i_, i_plus_one_] @
                G_retarded[i_plus_one_, i_plus_one_].conj().T @
                System_matrix[i_, i_plus_one_].conj().T
                + System_matrix[i_, i_plus_one_] @
                G_retarded[i_plus_one_, i_plus_one_] @
                Sigma_lesser_greater[i_plus_one_, i_]
            )
            @ g_retarded[i_, i_].conj().T
        )
        buf4 = (g_retarded[i_, i_] @
                  System_matrix[i_, i_plus_one_]
                  @ G_retarded[i_plus_one_, i_plus_one_] @
                  System_matrix[i_plus_one_, i_] @
                  g_lesser_greater[i_, i_])

        G_lesser_greater[i_, i_] = (
            g_lesser_greater[i_, i_]
            + g_retarded[i_, i_] @
            System_matrix[i_, i_plus_one_] @
            G_lesser_greater[i_plus_one_, i_plus_one_]
            @ System_matrix[i_, i_plus_one_].conj().T @
            g_retarded[i_, i_].conj().T
            - temp_1
            + (buf4 - buf4.conj().T)
        )


        G_lesser_greater[i_plus_one_, i_] =(
            G_retarded[i_plus_one_, i_plus_one_] @
            Sigma_lesser_greater[i_plus_one_, i_] @
            g_retarded[i_, i_].conj().T
            - G_retarded[i_plus_one_, i_plus_one_] @
            System_matrix[i_plus_one_, i_] @
            g_lesser_greater[i_, i_]
            - G_lesser_greater[i_plus_one_, i_plus_one_] @
            System_matrix[i_, i_plus_one_].conj().T @
            g_retarded[i_, i_].conj().T
        )

        G_lesser_greater[i_, i_plus_one_] = -G_lesser_greater[i_plus_one_, i_] .conj().T


    return G_lesser_greater

def rgf_lesser_greater_left_connected_tridiag_opt(
    System_matrix: np.ndarray,
    Sigma_lesser_greater: np.ndarray,
    blocksize: int
):
    # Storage for the full backward substitution
    G_lesser_greater = np.zeros_like(System_matrix)

    g_retarded = np.zeros_like(System_matrix)
    g_lesser_greater = np.zeros_like(System_matrix)
    assert System_matrix.shape[0] % blocksize == 0
    assert System_matrix.shape[1] % blocksize == 0
    assert Sigma_lesser_greater.shape[0] % blocksize == 0
    assert Sigma_lesser_greater.shape[1] % blocksize == 0
    # assert np.allclose(Sigma_lesser_greater, -Sigma_lesser_greater.conj().T)
    number_of_blocks = int(System_matrix.shape[0] / blocksize)

    # 0. Inverse of the first block retarded
    g_retarded[0:blocksize, 0:blocksize] = \
        np.linalg.inv(System_matrix[0:blocksize, 0:blocksize])

    # 3. Forward substitution lesser greater
    g_lesser_greater[0:blocksize, 0:blocksize] =\
        g_retarded[0:blocksize, 0:blocksize] @ \
        Sigma_lesser_greater[0:blocksize, 0:blocksize] @ \
        g_retarded[0:blocksize, 0:blocksize].conj().T

    # 1. Forward substitution (performed left to right)
    for i in range(1, number_of_blocks, 1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_minus_one_ = slice((i-1)*blocksize, i*blocksize)

        tmp = System_matrix[i_, i_minus_one_] @ g_retarded[i_minus_one_, i_minus_one_]
        g_retarded[i_, i_] = np.linalg.inv(System_matrix[i_, i_] - tmp @ System_matrix[i_minus_one_, i_])

        g_lesser_greater[i_, i_] = (
            g_retarded[i_, i_]
            @ (
                Sigma_lesser_greater[i_, i_]
                - tmp @
                    Sigma_lesser_greater[i_minus_one_, i_]    
                + (System_matrix[i_, i_minus_one_] @
                    g_lesser_greater[i_minus_one_, i_minus_one_]
                + Sigma_lesser_greater[i_minus_one_, i_].conj().T @
                    g_retarded[i_minus_one_, i_minus_one_].conj().T )@
                    System_matrix[i_, i_minus_one_].conj().T

            )
            @ g_retarded[i_, i_].conj().T
        )

    G_lesser_greater[-blocksize:,-blocksize:] = g_lesser_greater[-blocksize:,-blocksize:]
    G_tmp = g_retarded[-blocksize:,-blocksize:]


    # 2. Backward substitution (performed right to left)
    for i in range(number_of_blocks-2, -1, -1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_plus_one_ = slice((i+1)*blocksize, (i+2)*blocksize)

        buf2 = (G_tmp @
                System_matrix[i_plus_one_, i_])
        buf1 = (g_retarded[i_, i_] @
                  System_matrix[i_, i_plus_one_])

        buf4 = -(G_tmp @
            Sigma_lesser_greater[i_, i_plus_one_].conj().T @
            g_retarded[i_, i_].conj().T)



        buf7 = buf2 @ g_retarded[i_, i_]
        G_tmp   =  g_retarded[i_, i_] + (buf1 @ buf7)
    
        buf5 = (buf2 @ g_lesser_greater[i_, i_])

        buf3 = (
            G_lesser_greater[i_plus_one_, i_plus_one_] @
            buf1.conj().T
        )
        
        G_lesser_greater[i_, i_plus_one_] = (
            - buf4.conj().T
            + buf5.conj().T
            + buf3.conj().T
        )

        buf6 = (buf1 @ buf4)

        buf8 = (buf1 @ buf5)


        G_lesser_greater[i_, i_] = (
            g_lesser_greater[i_, i_]
            + buf1 @ buf3
            - (buf6 - buf6.conj().T)
            + (buf8 - buf8.conj().T)
        )


        G_lesser_greater[i_plus_one_, i_] = -G_lesser_greater[i_, i_plus_one_].conj().T


    return G_lesser_greater



def rgf_lesser_greater_retarded_left_connected_tridiag_opt(
    System_matrix: np.ndarray,
    Sigma_lesser_greater: np.ndarray,
    blocksize: int
):
    # Storage for the full backward substitution
    G_lesser_greater = np.zeros_like(System_matrix)
    G_retarded = np.zeros_like(System_matrix)

    g_retarded = np.zeros_like(System_matrix)
    g_lesser_greater = np.zeros_like(System_matrix)
    assert System_matrix.shape[0] % blocksize == 0
    assert System_matrix.shape[1] % blocksize == 0
    assert Sigma_lesser_greater.shape[0] % blocksize == 0
    assert Sigma_lesser_greater.shape[1] % blocksize == 0
    # assert np.allclose(Sigma_lesser_greater, -Sigma_lesser_greater.conj().T)
    number_of_blocks = int(System_matrix.shape[0] / blocksize)

    # 0. Inverse of the first block retarded
    g_retarded[0:blocksize, 0:blocksize] = \
        np.linalg.inv(System_matrix[0:blocksize, 0:blocksize])

    # 3. Forward substitution lesser greater
    g_lesser_greater[0:blocksize, 0:blocksize] =\
        g_retarded[0:blocksize, 0:blocksize] @ \
        Sigma_lesser_greater[0:blocksize, 0:blocksize] @ \
        g_retarded[0:blocksize, 0:blocksize].conj().T

    # 1. Forward substitution (performed left to right)
    for i in range(1, number_of_blocks, 1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_minus_one_ = slice((i-1)*blocksize, i*blocksize)

        tmp = System_matrix[i_, i_minus_one_] @ g_retarded[i_minus_one_, i_minus_one_]
        g_retarded[i_, i_] = np.linalg.inv(System_matrix[i_, i_] - tmp @ System_matrix[i_minus_one_, i_])

        g_lesser_greater[i_, i_] = (
            g_retarded[i_, i_]
            @ (
                Sigma_lesser_greater[i_, i_]
                - tmp @
                    Sigma_lesser_greater[i_minus_one_, i_]    
                + (System_matrix[i_, i_minus_one_] @
                    g_lesser_greater[i_minus_one_, i_minus_one_]
                + Sigma_lesser_greater[i_minus_one_, i_].conj().T @
                    g_retarded[i_minus_one_, i_minus_one_].conj().T )@
                    System_matrix[i_, i_minus_one_].conj().T

            )
            @ g_retarded[i_, i_].conj().T
        )

    G_lesser_greater[-blocksize:,-blocksize:] = g_lesser_greater[-blocksize:,-blocksize:]
    G_tmp = g_retarded[-blocksize:,-blocksize:]
    G_retarded[-blocksize:,-blocksize:] = g_retarded[-blocksize:,-blocksize:] 

    # 2. Backward substitution (performed right to left)
    for i in range(number_of_blocks-2, -1, -1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_plus_one_ = slice((i+1)*blocksize, (i+2)*blocksize)

        buf4 = -(G_tmp @
            Sigma_lesser_greater[i_, i_plus_one_].conj().T @
            g_retarded[i_, i_].conj().T)

        buf2 = (G_tmp @
                System_matrix[i_plus_one_, i_])
        buf1 = (g_retarded[i_, i_] @
                  System_matrix[i_, i_plus_one_])

        buf7 = -buf2 @ g_retarded[i_, i_]

        G_retarded[i_plus_one_, i_] = buf7
        G_retarded[i_, i_plus_one_] = -buf1 @ G_tmp
        G_tmp   =  g_retarded[i_, i_] - (buf1 @ buf7)
        G_retarded[i_, i_] = G_tmp

        buf5 = (buf2 @ g_lesser_greater[i_, i_])

        buf3 = (
            G_lesser_greater[i_plus_one_, i_plus_one_] @
            buf1.conj().T
        )
        
        G_lesser_greater[i_, i_plus_one_] = (
            - buf4.conj().T
            + buf5.conj().T
            + buf3.conj().T
        )

        buf6 = (buf1 @ buf4)

        buf8 = (buf1 @ buf5)


        G_lesser_greater[i_, i_] = (
            g_lesser_greater[i_, i_]
            + buf1 @ buf3
            - (buf6 - buf6.conj().T)
            + (buf8 - buf8.conj().T)
        )


        G_lesser_greater[i_plus_one_, i_] = -G_lesser_greater[i_, i_plus_one_].conj().T


    return G_lesser_greater, G_retarded


def rgf_lesser(
    System_matrix: np.ndarray,
    Sigma_lesser_greater: np.ndarray,
    blocksize: int
):

    y = np.zeros_like(System_matrix)
    x = np.zeros_like(System_matrix)

    y[0:blocksize, 0:blocksize]= np.linalg.inv(System_matrix[0:blocksize, 0:blocksize])
    x[0:blocksize, 0:blocksize] = y[0:blocksize, 0:blocksize] @ Sigma_lesser_greater[0:blocksize, 0:blocksize] @ y[0:blocksize, 0:blocksize].conj().T

    number_of_blocks = int(System_matrix.shape[0] / blocksize)

    # Forwards sweep.
    for i in range(1, number_of_blocks):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_minus_one_ = slice((i-1)*blocksize, i*blocksize)

        y[i_, i_] = np.linalg.inv(System_matrix[i_, i_] - System_matrix[i_, i_minus_one_] @ y[i_minus_one_, i_minus_one_] @ System_matrix[i_minus_one_, i_])

        x[i_, i_] = (
            y[i_, i_]
            @ (
                Sigma_lesser_greater[i_, i_]
                + System_matrix[i_, i_minus_one_] @ x[i_minus_one_, i_minus_one_] @ System_matrix[i_, i_minus_one_].conj().T
                - Sigma_lesser_greater[i_, i_minus_one_] @ y[i_minus_one_, i_minus_one_].conj().T @ System_matrix[i_, i_minus_one_].conj().T
                - System_matrix[i_, i_minus_one_] @ y[i_minus_one_, i_minus_one_] @ Sigma_lesser_greater[i_minus_one_, i_]
            )
            @ y[i_, i_].conj().T
        )


    # Backwards sweep.
    for i in range(number_of_blocks - 2, -1, -1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_plus_one_ = slice((i+1)*blocksize, (i+2)*blocksize)

        temp_1 = (
            y[i_, i_]
            @ (
                Sigma_lesser_greater[i_, i_plus_one_] @ y[i_plus_one_, i_plus_one_].conj().T @ System_matrix[i_, i_plus_one_].conj().T
                + System_matrix[i_, i_plus_one_] @ y[i_plus_one_, i_plus_one_] @ Sigma_lesser_greater[i_plus_one_, i_]
            )
            @ y[i_, i_].conj().T
        )
        buf4 = y[i_, i_] @ System_matrix[i_, i_plus_one_] @ y[i_plus_one_, i_plus_one_] @ System_matrix[i_plus_one_, i_] @ x[i_, i_]

        x[i_plus_one_, i_] = (
            -(
                x[i_plus_one_, i_plus_one_] @ System_matrix[i_, i_plus_one_].conj().T @ y[i_, i_].conj().T
                + y[i_plus_one_, i_plus_one_] @ System_matrix[i_plus_one_, i_] @ x[i_, i_]
                - y[i_plus_one_, i_plus_one_] @ Sigma_lesser_greater[i_plus_one_, i_]@ y[i_, i_].conj().T
            )
            
        )
        
        x[i_, i_] = (
            x[i_, i_]
            + y[i_, i_] @ System_matrix[i_, i_plus_one_] @ x[i_plus_one_, i_plus_one_] @ System_matrix[i_, i_plus_one_].conj().T @ y[i_, i_].conj().T
            - temp_1
            + (buf4 - buf4.conj().T)
        )

        x[i_, i_plus_one_] = -y[i_, i_] @ (
            System_matrix[i_, i_plus_one_] @ x[i_plus_one_, i_plus_one_]
            + Sigma_lesser_greater[i_, i_] @ y[i_, i_].conj().T @ System_matrix[i_plus_one_, i_].conj().T @ y[i_plus_one_, i_plus_one_].conj().T
            - Sigma_lesser_greater[i_, i_plus_one_] @ y[i_plus_one_, i_plus_one_].conj().T
        )


        y[i_, i_] = y[i_, i_] + y[i_, i_] @ System_matrix[i_, i_plus_one_] @ y[i_plus_one_, i_plus_one_] @ System_matrix[i_plus_one_, i_] @ y[i_, i_]


    return x


def rgf_lesser_greater_right_connected(
    System_matrix: np.ndarray,
    Sigma_lesser_greater: np.ndarray,
    blocksize: int
):
    # Storage for the full backward substitution
    G_lesser_greater = np.zeros_like(System_matrix)
    G_retarded = np.zeros_like(System_matrix)
    Sigma_boundary = np.zeros_like(System_matrix)

    g_retarded = np.zeros_like(System_matrix)
    g_lesser_greater = np.zeros_like(System_matrix)
    assert System_matrix.shape[0] % blocksize == 0
    assert System_matrix.shape[1] % blocksize == 0
    assert Sigma_lesser_greater.shape[0] % blocksize == 0
    assert Sigma_lesser_greater.shape[1] % blocksize == 0
    assert np.allclose(Sigma_lesser_greater, -Sigma_lesser_greater.conj().T)
    number_of_blocks = int(System_matrix.shape[0] / blocksize)

    # First step of iteration
    g_retarded[-blocksize:,-blocksize:] =\
        np.linalg.inv(System_matrix[-blocksize:,-blocksize:])

    for i in range(number_of_blocks - 2, -1, -1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_plus_one_ = slice((i+1)*blocksize, (i+2)*blocksize)

        g_retarded[i_,i_] = np.linalg.inv(System_matrix[i_,i_] \
                            - System_matrix[i_,i_plus_one_] \
                            @ g_retarded[i_plus_one_,i_plus_one_] \
                            @ System_matrix[i_plus_one_,i_])

    G_retarded[:blocksize,:blocksize] = g_retarded[:blocksize,:blocksize]

    for i in range(1, number_of_blocks):

        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_minus_one_ = slice((i-1)*blocksize, i*blocksize)

        G_retarded[i_,i_] = g_retarded[i_,i_] + g_retarded[i_,i_] @\
                        System_matrix[i_,i_minus_one_] @\
                        G_retarded[i_minus_one_,i_minus_one_] @\
                        System_matrix[i_minus_one_,i_] @\
                        g_retarded[i_,i_]
        

    g_lesser_greater[-blocksize:,-blocksize:] = g_retarded[-blocksize:,-blocksize:] @\
        Sigma_lesser_greater[-blocksize:,-blocksize:] @\
        g_retarded[-blocksize:,-blocksize:].T.conj()

    for i in range(number_of_blocks - 2, -1, -1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_plus_one_ = slice((i+1)*blocksize, (i+2)*blocksize)

        AL = System_matrix[i_,i_plus_one_] @\
            g_retarded[i_plus_one_,i_plus_one_] @\
            Sigma_lesser_greater[i_plus_one_,i_]

        Sigma_boundary[i_,i_] = System_matrix[i_,i_plus_one_] @ \
                            g_lesser_greater[i_plus_one_,i_plus_one_] @ \
                            System_matrix[i_,i_plus_one_].T.conj() -\
                            (AL - AL.T.conj())


        g_lesser_greater[i_,i_] = g_retarded[i_,i_] @ \
                            (Sigma_lesser_greater[i_,i_] +\
                            Sigma_boundary[i_,i_])  @\
                            g_retarded[i_,i_].T.conj()


    G_lesser_greater[:blocksize,:blocksize] = g_lesser_greater[:blocksize,:blocksize]

    for i in range(1, number_of_blocks):

        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_minus_one_ = slice((i-1)*blocksize, i*blocksize)

        AL = g_retarded[i_,i_] @\
            Sigma_lesser_greater[i_,i_minus_one_] @\
            G_retarded[i_minus_one_,i_minus_one_].T.conj() @\
            System_matrix[i_,i_minus_one_].T.conj() @\
            g_retarded[i_,i_].T.conj()

        BL = g_retarded[i_,i_] @\
            System_matrix[i_,i_minus_one_] @\
            G_retarded[i_minus_one_,i_minus_one_] @\
            System_matrix[i_minus_one_,i_] @\
            g_lesser_greater[i_,i_]

        G_lesser_greater[i_,i_] = g_lesser_greater[i_,i_] +\
                            g_retarded[i_,i_] @\
                            System_matrix[i_,i_minus_one_] @\
                            G_lesser_greater[i_minus_one_,i_minus_one_] @\
                            System_matrix[i_,i_minus_one_].T.conj() @\
                            g_retarded[i_,i_].T.conj() -\
                            (AL - AL.T.conj()) + (BL - BL.T.conj())


    return G_retarded, G_lesser_greater

def rgf_lesser_greater_right_connected_tridiag(
    System_matrix: np.ndarray,
    Sigma_lesser_greater: np.ndarray,
    blocksize: int
):
    # Storage for the full backward substitution
    G_lesser_greater = np.zeros_like(System_matrix)
    G_retarded = np.zeros_like(System_matrix)
    Sigma_boundary = np.zeros_like(System_matrix)

    g_retarded = np.zeros_like(System_matrix)
    g_lesser_greater = np.zeros_like(System_matrix)
    assert System_matrix.shape[0] % blocksize == 0
    assert System_matrix.shape[1] % blocksize == 0
    assert Sigma_lesser_greater.shape[0] % blocksize == 0
    assert Sigma_lesser_greater.shape[1] % blocksize == 0
    assert np.allclose(Sigma_lesser_greater, -Sigma_lesser_greater.conj().T)
    number_of_blocks = int(System_matrix.shape[0] / blocksize)

    # First step of iteration
    g_retarded[-blocksize:,-blocksize:] =\
        np.linalg.inv(System_matrix[-blocksize:,-blocksize:])

    for i in range(number_of_blocks - 2, -1, -1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_plus_one_ = slice((i+1)*blocksize, (i+2)*blocksize)

        g_retarded[i_,i_] = np.linalg.inv(System_matrix[i_,i_] \
                            - System_matrix[i_,i_plus_one_] \
                            @ g_retarded[i_plus_one_,i_plus_one_] \
                            @ System_matrix[i_plus_one_,i_])

    G_retarded[:blocksize,:blocksize] = g_retarded[:blocksize,:blocksize]

    for i in range(1, number_of_blocks):

        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_minus_one_ = slice((i-1)*blocksize, i*blocksize)

        G_retarded[i_,i_] = g_retarded[i_,i_] + g_retarded[i_,i_] @\
                        System_matrix[i_,i_minus_one_] @\
                        G_retarded[i_minus_one_,i_minus_one_] @\
                        System_matrix[i_minus_one_,i_] @\
                        g_retarded[i_,i_]
        

    g_lesser_greater[-blocksize:,-blocksize:] = g_retarded[-blocksize:,-blocksize:] @\
        Sigma_lesser_greater[-blocksize:,-blocksize:] @\
        g_retarded[-blocksize:,-blocksize:].T.conj()

    for i in range(number_of_blocks - 2, -1, -1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_plus_one_ = slice((i+1)*blocksize, (i+2)*blocksize)

        AL = System_matrix[i_,i_plus_one_] @\
            g_retarded[i_plus_one_,i_plus_one_] @\
            Sigma_lesser_greater[i_plus_one_,i_]

        Sigma_boundary[i_,i_] = System_matrix[i_,i_plus_one_] @ \
                            g_lesser_greater[i_plus_one_,i_plus_one_] @ \
                            System_matrix[i_,i_plus_one_].T.conj() -\
                            (AL - AL.T.conj())


        g_lesser_greater[i_,i_] = g_retarded[i_,i_] @ \
                            (Sigma_lesser_greater[i_,i_] +\
                            Sigma_boundary[i_,i_])  @\
                            g_retarded[i_,i_].T.conj()


    G_lesser_greater[:blocksize,:blocksize] = g_lesser_greater[:blocksize,:blocksize]

    for i in range(1, number_of_blocks):

        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_minus_one_ = slice((i-1)*blocksize, i*blocksize)

        AL = g_retarded[i_,i_] @\
            Sigma_lesser_greater[i_,i_minus_one_] @\
            G_retarded[i_minus_one_,i_minus_one_].T.conj() @\
            System_matrix[i_,i_minus_one_].T.conj() @\
            g_retarded[i_,i_].T.conj()

        BL = g_retarded[i_,i_] @\
            System_matrix[i_,i_minus_one_] @\
            G_retarded[i_minus_one_,i_minus_one_] @\
            System_matrix[i_minus_one_,i_] @\
            g_lesser_greater[i_,i_]

        G_lesser_greater[i_,i_] = g_lesser_greater[i_,i_] +\
                            g_retarded[i_,i_] @\
                            System_matrix[i_,i_minus_one_] @\
                            G_lesser_greater[i_minus_one_,i_minus_one_] @\
                            System_matrix[i_,i_minus_one_].T.conj() @\
                            g_retarded[i_,i_].T.conj() -\
                            (AL - AL.T.conj()) + (BL - BL.T.conj())

        G_lesser_greater[i_minus_one_, i_] =(
            G_retarded[i_minus_one_, i_minus_one_] @
            Sigma_lesser_greater[i_minus_one_, i_] @
            g_retarded[i_, i_].conj().T
            - G_retarded[i_minus_one_, i_minus_one_] @
            System_matrix[i_minus_one_, i_] @
            g_lesser_greater[i_, i_]
            - G_lesser_greater[i_minus_one_, i_minus_one_] @
            System_matrix[i_, i_minus_one_].conj().T @
            g_retarded[i_, i_].conj().T
        )

        G_lesser_greater[i_, i_minus_one_] = -G_lesser_greater[i_minus_one_, i_].conj().T

    return G_retarded, G_lesser_greater