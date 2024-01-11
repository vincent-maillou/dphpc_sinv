# Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.

import matrix_utils
import numpy as np


SEED = 8000
BATCHSIZE = 1
PATH_TO_FILE = "/usr/scratch/mont-fort23/almaeder/rgf_test/"

if __name__ == "__main__":

    # int nb_test = 8;
    # int bs_test = 4;
    # int n_blocks_input[nb_test] = {3*4, 3*8, 3*16, 3*32, 3*64, 3*128, 3*256, 3*512};
    # int blocksize_input[bs_test] = {64, 128, 256, 512};

    BLOCKSIZES = [64, 128, 256, 512]
    NUM_OF_BLOCKS = [3*4, 3*8, 3*16, 3*32, 3*64, 3*128, 3*256, 3*512]
    for BLOCKSIZE in BLOCKSIZES:
        
        MAT_SIZES = [BLOCKSIZE*NUM_OF_BLOCKS[i] for i in range(len(NUM_OF_BLOCKS))]

        # filename = "batched_matrix_parameters.txt"
        # matrix_utils.write_matrix_parameters_batched(
        #     PATH_TO_FILE+filename, MAT_SIZE, BLOCKSIZE, BATCHSIZE)
        print("Generating matrices for blocksize " + str(BLOCKSIZE))
        for j, MAT_SIZE in enumerate(MAT_SIZES):
            print("Generating matrix of size " + str(MAT_SIZE))
            # Generate random matrices
            for i in range(BATCHSIZE):
                # print("Generating matrix " + str(i+1) + " of " + str(BATCHSIZE))


                # print("Generating matrix System Matrix")
                # system_matrix = matrix_utils.generateBandedMatrix(MAT_SIZE, BLOCKSIZE, SEED) + i * np.eye(MAT_SIZE)

                # # Extract diagonal and off-diagonal blocks
                # system_matrix_diagblk = matrix_utils.extract_diagonal_blocks(
                #     system_matrix, MAT_SIZE, BLOCKSIZE)
                # system_matrix_upperblk = matrix_utils.extract_offdiagonal_blocks(
                #     system_matrix, MAT_SIZE, BLOCKSIZE, 1)
                # system_matrix_lowerblk = matrix_utils.extract_offdiagonal_blocks(
                #     system_matrix, MAT_SIZE, BLOCKSIZE, -1)

                rng = np.random.default_rng()
                system_matrix_diagblk = rng.random((BLOCKSIZE, MAT_SIZE), dtype=np.float64) + 1j*rng.random((BLOCKSIZE, MAT_SIZE), dtype=np.float64)
                system_matrix_upperblk = rng.random((BLOCKSIZE, MAT_SIZE - BLOCKSIZE), dtype=np.float64) + 1j*rng.random((BLOCKSIZE, MAT_SIZE - BLOCKSIZE), dtype=np.float64)
                system_matrix_lowerblk = rng.random((BLOCKSIZE, MAT_SIZE - BLOCKSIZE), dtype=np.float64) + 1j*rng.random((BLOCKSIZE, MAT_SIZE - BLOCKSIZE), dtype=np.float64)

                diag_blk = np.diag(10.0 * np.ones(BLOCKSIZE))

                # make diag dominant
                for k in range(NUM_OF_BLOCKS[j]):
                    system_matrix_diagblk[:,k*BLOCKSIZE:(k+1)*BLOCKSIZE] += diag_blk

                filename = ("system_matrix_" + str(i) + "_diagblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
                matrix_utils.write_matrix_to_file(
                    PATH_TO_FILE+filename, system_matrix_diagblk)
                filename = ("system_matrix_" + str(i) + "_upperblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
                matrix_utils.write_matrix_to_file(
                    PATH_TO_FILE+filename, system_matrix_upperblk)
                filename = ("system_matrix_" + str(i) + "_lowerblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
                matrix_utils.write_matrix_to_file(
                    PATH_TO_FILE+filename, system_matrix_lowerblk)
