import numpy as np
import time
import mkl

def rgf_lesser_greater_retarded_left_connected_tridiag_memcon(
    System_matrix_diag: np.ndarray,
    System_matrix_upper: np.ndarray,
    System_matrix_lower: np.ndarray,
    Sigma_lesser_diag: np.ndarray,
    Sigma_lesser_upper: np.ndarray,
    Sigma_greater_diag: np.ndarray,
    Sigma_greater_upper: np.ndarray,
    number_of_blocks: int,
    batch_size: int
):
    # Storage for the full backward substitution
    G_lesser_diag = np.zeros_like(System_matrix_diag)
    G_lesser_upper = np.zeros_like(System_matrix_upper)
    G_greater_diag = np.zeros_like(System_matrix_diag)
    G_greater_upper = np.zeros_like(System_matrix_upper)

    G_retarded_diag = np.zeros_like(System_matrix_diag)
    G_retarded_upper = np.zeros_like(System_matrix_upper)
    G_retarded_lower = np.zeros_like(System_matrix_lower)

    g_retarded_diag = np.zeros_like(System_matrix_diag)
    g_lesser_diag = np.zeros_like(System_matrix_diag)
    g_greater_diag = np.zeros_like(System_matrix_diag)


    for batch in range(batch_size):

        # 0. Inverse of the first block retarded
        g_retarded_diag[0, batch , :, :] = \
            np.linalg.inv(System_matrix_diag[0, batch , :, :])

        # 3. Forward substitution lesser greater
        g_lesser_diag[0, batch , :, :] =\
            g_retarded_diag[0, batch , :, :] @ \
            Sigma_lesser_diag[0, batch , :, :] @ \
            g_retarded_diag[0, batch , :, :].conj().T
        g_greater_diag[0, batch , :, :] =\
            g_retarded_diag[0, batch , :, :] @ \
            Sigma_greater_diag[0, batch , :, :] @ \
            g_retarded_diag[0, batch , :, :].conj().T


        # 1. Forward substitution (performed left to right)
        for i in range(1, number_of_blocks, 1):

            tmp = System_matrix_lower[i-1, batch , :, :] @ g_retarded_diag[i-1, batch , :, :]
            g_retarded_diag[i, batch , :, :] = np.linalg.inv(System_matrix_diag[i, batch , :, :] - tmp @ System_matrix_upper[i-1, batch , :, :])

            g_lesser_diag[i, batch , :, :] = (
                g_retarded_diag[i, batch , :, :]
                @ (
                    Sigma_lesser_diag[i, batch , :, :]
                    - tmp @
                        Sigma_lesser_upper[i-1, batch , :, :]
                    + (System_matrix_lower[i-1, batch , :, :] @
                        g_lesser_diag[i-1, batch , :, :]
                    + Sigma_lesser_upper[i-1, batch , :, :].conj().T @
                        g_retarded_diag[i-1, batch , :, :].conj().T )@
                        System_matrix_lower[i-1, batch , :, :].conj().T

                )
                @ g_retarded_diag[i, batch , :, :].conj().T
            )
            g_greater_diag[i, batch , :, :] = (
                g_retarded_diag[i, batch , :, :]
                @ (
                    Sigma_greater_diag[i, batch , :, :]
                    - tmp @
                        Sigma_greater_upper[i-1, batch , :, :]    
                    + (System_matrix_lower[i-1, batch , :, :] @
                        g_greater_diag[i-1, batch , :, :]
                    + Sigma_greater_upper[i-1, batch , :, :].conj().T @
                        g_retarded_diag[i-1, batch , :, :].conj().T )@
                        System_matrix_lower[i-1, batch , :, :].conj().T

                )
                @ g_retarded_diag[i, batch , :, :].conj().T
            )

        G_lesser_diag[number_of_blocks-1, batch , :, :] = g_lesser_diag[number_of_blocks-1, batch , :, :]
        G_greater_diag[number_of_blocks-1, batch , :, :] = g_greater_diag[number_of_blocks-1, batch , :, :]
        G_tmp = g_retarded_diag[number_of_blocks-1, batch , :, :]
        G_retarded_diag[number_of_blocks-1, batch , :, :] = g_retarded_diag[number_of_blocks-1, batch , :, :]

        # 2. Backward substitution (performed right to left)
        for i in range(number_of_blocks-2, -1, -1):

            buf4_lesser = -(G_tmp @
                Sigma_lesser_upper[i, batch , :, :].conj().T @
                g_retarded_diag[i, batch , :, :].conj().T)
            buf4_greater = -(G_tmp @
                Sigma_greater_upper[i, batch , :, :].conj().T @
                g_retarded_diag[i, batch , :, :].conj().T)


            buf2 = (G_tmp @
                    System_matrix_lower[i, batch , :, :])
            buf1 = (g_retarded_diag[i, batch , :, :] @
                      System_matrix_upper[i, batch , :, :])

            buf7 = -buf2 @ g_retarded_diag[i, batch , :, :]

            G_retarded_lower[i, batch , :, :] = buf7
            G_retarded_upper[i, batch , :, :] = -buf1 @ G_tmp
            G_tmp   =  g_retarded_diag[i, batch , :, :] - (buf1 @ buf7)
            G_retarded_diag[i, batch , :, :] = G_tmp

            buf5_lesser = (buf2 @ g_lesser_diag[i, batch , :, :])
            buf5_greater = (buf2 @ g_greater_diag[i, batch , :, :])

            buf3_lesser = (
                G_lesser_diag[i-1, batch , :, :] @
                buf1.conj().T
            )
            buf3_greater = (
                G_greater_diag[i-1, batch , :, :] @
                buf1.conj().T
            )
            
            G_lesser_upper[i, batch , :, :] = (
                - buf4_lesser.conj().T
                + buf5_lesser.conj().T
                + buf3_lesser.conj().T
            )
            G_greater_upper[i, batch , :, :] = (
                - buf4_greater.conj().T
                + buf5_greater.conj().T
                + buf3_greater.conj().T
            )

            buf6_lesser = (buf1 @ buf4_lesser)
            buf6_greater = (buf1 @ buf4_greater)

            buf8_lesser = (buf1 @ buf5_lesser)
            buf8_greater = (buf1 @ buf5_greater)


            G_lesser_diag[i, batch , :, :] = (
                g_lesser_diag[i, batch , :, :]
                + buf1 @ buf3_lesser
                - (buf6_lesser - buf6_lesser.conj().T)
                + (buf8_lesser - buf8_lesser.conj().T)
            )
            G_greater_diag[i, batch , :, :] = (
                g_greater_diag[i, batch , :, :]
                + buf1 @ buf3_greater
                - (buf6_greater - buf6_greater.conj().T)
                + (buf8_greater - buf8_greater.conj().T)
            )


    return G_retarded_diag, G_retarded_upper, G_retarded_lower, G_lesser_diag, G_lesser_upper, G_greater_diag, G_greater_upper


if __name__ == "__main__":
    # Test
    # System

    SEED = 8000
    BATCHSIZES = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
    BLOCKSIZES = [64, 128, 256, 512, 768, 1024]
    NUM_OF_BLOCKS = [2, 4, 6, 8, 10, 12, 14]

    path_times = "/usr/scratch/mont-fort23/almaeder/rgf_times_batched/"
    nmeas = 22
    rng = np.random.default_rng()
    mkl.set_num_threads(14)
    for BATCHSIZE in BATCHSIZES:
        for BLOCKSIZE in BLOCKSIZES:
            for NUM_OF_BLOCK in NUM_OF_BLOCKS:
                memory_con = 10*NUM_OF_BLOCK*BLOCKSIZE*BLOCKSIZE*BATCHSIZE * 16 / (10**9)
                print("Memory consumption: " + str(memory_con) + " GB")
                print("Generating matrices for batchsize " + str(BATCHSIZE))
                print("Generating matrices for blocksize " + str(BLOCKSIZE))
                print("Generating matrices for number of blocks " + str(NUM_OF_BLOCK))

                if memory_con > 100:
                    continue
                else:
                    system_matrix_diagblk = (
                        rng.random((NUM_OF_BLOCK, BATCHSIZE, BLOCKSIZE, BLOCKSIZE), dtype=np.float64)
                        + 1j*rng.random((NUM_OF_BLOCK, BATCHSIZE, BLOCKSIZE, BLOCKSIZE), dtype=np.float64)
                    )
                    system_matrix_upperblk = (
                        rng.random((NUM_OF_BLOCK, BATCHSIZE, BLOCKSIZE, BLOCKSIZE), dtype=np.float64)
                        + 1j*rng.random((NUM_OF_BLOCK, BATCHSIZE, BLOCKSIZE, BLOCKSIZE), dtype=np.float64)
                    )
                    system_matrix_lowerblk = (
                        rng.random((NUM_OF_BLOCK, BATCHSIZE, BLOCKSIZE, BLOCKSIZE), dtype=np.float64)
                        + 1j*rng.random((NUM_OF_BLOCK, BATCHSIZE, BLOCKSIZE, BLOCKSIZE), dtype=np.float64)
                    )

                    self_energy_lesser_diagblk = (
                        rng.random((NUM_OF_BLOCK, BATCHSIZE, BLOCKSIZE, BLOCKSIZE), dtype=np.float64)
                        + 1j*rng.random((NUM_OF_BLOCK, BATCHSIZE, BLOCKSIZE, BLOCKSIZE), dtype=np.float64)
                    )
                    self_energy_lesser_upperblk = (
                        rng.random((NUM_OF_BLOCK, BATCHSIZE, BLOCKSIZE, BLOCKSIZE), dtype=np.float64)
                        + 1j*rng.random((NUM_OF_BLOCK, BATCHSIZE, BLOCKSIZE, BLOCKSIZE), dtype=np.float64)
                    )
                    self_energy_greater_diagblk = (
                        rng.random((NUM_OF_BLOCK, BATCHSIZE, BLOCKSIZE, BLOCKSIZE), dtype=np.float64)
                        + 1j*rng.random((NUM_OF_BLOCK, BATCHSIZE, BLOCKSIZE, BLOCKSIZE), dtype=np.float64)
                    )
                    self_energy_greater_upperblk = (
                        rng.random((NUM_OF_BLOCK, BATCHSIZE, BLOCKSIZE, BLOCKSIZE), dtype=np.float64)
                        + 1j*rng.random((NUM_OF_BLOCK, BATCHSIZE, BLOCKSIZE, BLOCKSIZE), dtype=np.float64)
                    )
                    diag_blk = np.diag(10.0 * np.ones(BLOCKSIZE))
                    for block in range(NUM_OF_BLOCK):
                        for batcht in range(BATCHSIZE):
                            system_matrix_diagblk[block, batcht, :, :] += diag_blk
                            self_energy_lesser_diagblk[block, batcht, :, :] -= (self_energy_lesser_diagblk[block, batcht, :, :].conj().T)
                            self_energy_greater_diagblk[block, batcht, :, :] -= (self_energy_greater_diagblk[block, batcht, :, :].conj().T)


                    times = np.zeros(nmeas, dtype=np.float64)
                    for meas in range(nmeas):
                        times[meas] = - time.perf_counter()
                        rgf_lesser_greater_retarded_left_connected_tridiag_memcon(
                            system_matrix_diagblk,
                            system_matrix_upperblk,
                            system_matrix_lowerblk,
                            self_energy_lesser_diagblk,
                            self_energy_lesser_upperblk,
                            self_energy_greater_diagblk,
                            self_energy_greater_upperblk,
                            NUM_OF_BLOCK,
                            BATCHSIZE
                        )
                        times[meas] += time.perf_counter()
                        print("Time for meas " + str(meas) + ": " + str(times[meas]))
                    save_path = path_times + "times_lesser_greater_retarded_python_" + str(BLOCKSIZE*NUM_OF_BLOCK) + "_" + str(NUM_OF_BLOCK) + "_" + str(BATCHSIZE) + ".txt"
                    np.savetxt(save_path, times, fmt="%1.16f")
                    

