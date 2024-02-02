# Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.

import matrix_utils
import numpy as np



if __name__ == "__main__":

    SEED = 8000
    BATCHSIZES = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
    PATH_TO_FILE = "/usr/scratch/mont-fort23/almaeder/rgf_test/"
    BLOCKSIZES = [64, 128, 256, 512, 768, 1024]
    rng = np.random.default_rng()
    for BATCHSIZE in BATCHSIZES:
        print("Generating matrices for batchsize " + str(BATCHSIZE))
        for BLOCKSIZE in BLOCKSIZES:
            
            print("Generating matrices for blocksize " + str(BLOCKSIZE))
            diag_blk = np.diag(10.0 * np.ones(BLOCKSIZE)).flatten()
            # Generate random matrices
            system_matrix_diagblk = rng.random((BLOCKSIZE*BLOCKSIZE*BATCHSIZE), dtype=np.float64) + 1j*rng.random((BLOCKSIZE*BLOCKSIZE*BATCHSIZE), dtype=np.float64)
            for k in range(BATCHSIZE):
                system_matrix_diagblk[k*BLOCKSIZE*BLOCKSIZE:(k+1)*BLOCKSIZE*BLOCKSIZE] += diag_blk
            system_matrix_upperblk = rng.random((BLOCKSIZE*BLOCKSIZE*BATCHSIZE), dtype=np.float64) + 1j*rng.random((BLOCKSIZE*BLOCKSIZE*BATCHSIZE), dtype=np.float64)
            system_matrix_lowerblk = rng.random((BLOCKSIZE*BLOCKSIZE*BATCHSIZE), dtype=np.float64) + 1j*rng.random((BLOCKSIZE*BLOCKSIZE*BATCHSIZE), dtype=np.float64)

            self_energy_diagblk = rng.random((BLOCKSIZE*BLOCKSIZE*BATCHSIZE), dtype=np.float64) + 1j*rng.random((BLOCKSIZE*BLOCKSIZE*BATCHSIZE), dtype=np.float64)
            for k in range(BATCHSIZE):
                system_matrix_diagblk[k*BLOCKSIZE*BLOCKSIZE:(k+1)*BLOCKSIZE*BLOCKSIZE] -= (system_matrix_diagblk[k*BLOCKSIZE*BLOCKSIZE:(k+1)*BLOCKSIZE*BLOCKSIZE].reshape(BLOCKSIZE,BLOCKSIZE).conj().T).flatten()
            self_energy_upperblk = rng.random((BLOCKSIZE*BLOCKSIZE*BATCHSIZE), dtype=np.float64) + 1j*rng.random((BLOCKSIZE*BLOCKSIZE*BATCHSIZE), dtype=np.float64)

            
            filename = ("system_matrix_diagblk_" + str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
            matrix_utils.write_matrix_to_file(
                PATH_TO_FILE+filename, system_matrix_diagblk)
            filename = ("system_matrix_upperblk_" + str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
            matrix_utils.write_matrix_to_file(
                PATH_TO_FILE+filename, system_matrix_upperblk)
            filename = ("system_matrix_lowerblk_" + str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
            matrix_utils.write_matrix_to_file(
                PATH_TO_FILE+filename, system_matrix_lowerblk)



            filename = ("self_energy_diagblk_" + str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
            matrix_utils.write_matrix_to_file(
                PATH_TO_FILE+filename, self_energy_diagblk)
            filename = ("self_energy_upperblk_" + str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
            matrix_utils.write_matrix_to_file(
                PATH_TO_FILE+filename, self_energy_upperblk)
            

