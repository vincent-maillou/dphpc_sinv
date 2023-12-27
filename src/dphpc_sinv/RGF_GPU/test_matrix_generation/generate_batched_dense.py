# Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.

import matrix_utils
import numpy as np
import argparse

SEED = 8000
MAT_SIZE = 5408
BLOCKSIZE = 416
BATCHSIZE = 112
PATH_TO_FILE = "/usr/scratch/mont-fort17/almaeder/rgf_test/"
MAT_SIZE = 1000
BLOCKSIZE = 100
BATCHSIZE = 100
PATH_TO_FILE = "/usr/scratch/mont-fort17/almaeder/rgf_test/"

BLOCKSIZE = 512
MAT_SIZE = BLOCKSIZE*45
BATCHSIZE = 1
PATH_TO_FILE = "/usr/scratch/mont-fort17/almaeder/rgf_test/"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Matrix generation")
    parser.add_argument("-ms", "--mat_size", default=str(MAT_SIZE), required=False)
    parser.add_argument("-bls", "--blocksize", default=str(BLOCKSIZE), required=False)
    parser.add_argument("-bas", "--batchsize", default=str(BATCHSIZE), required=False)
    parser.add_argument("-ptf", "--path_to_file", default=PATH_TO_FILE, required=False)
    args = parser.parse_args()
    MAT_SIZE = int(args.mat_size)
    BLOCKSIZE = int(args.blocksize)
    BATCHSIZE = int(args.batchsize)
    PATH_TO_FILE = args.path_to_file

    filename = "batched_matrix_parameters.txt"
    matrix_utils.write_matrix_parameters_batched(
        PATH_TO_FILE+filename, MAT_SIZE, BLOCKSIZE, BATCHSIZE)

    # Generate random matrices
    for i in range(BATCHSIZE):
        print("Generating matrix " + str(i+1) + " of " + str(BATCHSIZE))


        print("Generating matrix System Matrix")
        system_matrix = matrix_utils.generateBandedMatrix(MAT_SIZE, BLOCKSIZE, SEED) + i * np.eye(MAT_SIZE)

        # Extract diagonal and off-diagonal blocks
        system_matrix_diagblk = matrix_utils.extract_diagonal_blocks(
            system_matrix, MAT_SIZE, BLOCKSIZE)
        system_matrix_upperblk = matrix_utils.extract_offdiagonal_blocks(
            system_matrix, MAT_SIZE, BLOCKSIZE, 1)
        system_matrix_lowerblk = matrix_utils.extract_offdiagonal_blocks(
            system_matrix, MAT_SIZE, BLOCKSIZE, -1)

        filename = ("system_matrix_" + str(i) + "_diagblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, system_matrix_diagblk)
        filename = ("system_matrix_" + str(i) + "_upperblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, system_matrix_upperblk)
        filename = ("system_matrix_" + str(i) + "_lowerblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, system_matrix_lowerblk)


        # print("Generating matrix inverse")
        # # Compute inv
        # inv_matrix = np.linalg.inv(system_matrix)

        # # Assert system_matrix to be invertible
        # assert np.allclose(inv_matrix @ system_matrix, np.eye(MAT_SIZE))

        # matrix_inv_diag_blk = matrix_utils.extract_diagonal_blocks(
        #     inv_matrix, MAT_SIZE, BLOCKSIZE)
        # matrix_inv_upper_blk = matrix_utils.extract_offdiagonal_blocks(
        #     inv_matrix, MAT_SIZE, BLOCKSIZE, 1)
        # matrix_inv_lower_blk = matrix_utils.extract_offdiagonal_blocks(
        #     inv_matrix, MAT_SIZE, BLOCKSIZE, -1)

        # filename = "retarded_" + str(i) + "_inv_diagblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin"
        # matrix_utils.write_matrix_to_file(
        #     PATH_TO_FILE+filename, matrix_inv_diag_blk)
        # filename = "retarded_" + str(i) + "_inv_upperblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin"
        # matrix_utils.write_matrix_to_file(
        #     PATH_TO_FILE+filename, matrix_inv_upper_blk)
        # filename = "retarded_" + str(i) + "_inv_lowerblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin"
        # matrix_utils.write_matrix_to_file(
        #     PATH_TO_FILE+filename, matrix_inv_lower_blk)

        # print("Generating matrix self energy")
        # self_energy_lesser = matrix_utils.generateBandedMatrix(MAT_SIZE, BLOCKSIZE, SEED+1) + 2*i * np.eye(MAT_SIZE)
        # self_energy_lesser = self_energy_lesser - self_energy_lesser.conj().T

        # self_energy_greater = matrix_utils.generateBandedMatrix(MAT_SIZE, BLOCKSIZE, SEED+2) + 3*i * np.eye(MAT_SIZE)
        # self_energy_greater = self_energy_greater - self_energy_greater.conj().T

        # self_energy_lesser_diagblk = matrix_utils.extract_diagonal_blocks(
        #     self_energy_lesser, MAT_SIZE, BLOCKSIZE)
        # self_energy_lesser_upperblk = matrix_utils.extract_offdiagonal_blocks(
        #     self_energy_lesser, MAT_SIZE, BLOCKSIZE, 1)
        # self_energy_lesser_lowerblk = matrix_utils.extract_offdiagonal_blocks(
        #     self_energy_lesser, MAT_SIZE, BLOCKSIZE, -1)
        
        # self_energy_greater_diagblk = matrix_utils.extract_diagonal_blocks(
        #     self_energy_greater, MAT_SIZE, BLOCKSIZE)
        # self_energy_greater_upperblk = matrix_utils.extract_offdiagonal_blocks(
        #     self_energy_greater, MAT_SIZE, BLOCKSIZE, 1)
        # self_energy_greater_lowerblk = matrix_utils.extract_offdiagonal_blocks(
        #     self_energy_greater, MAT_SIZE, BLOCKSIZE, -1)


        # filename = ("self_energy_lesser_" + str(i) + "_diagblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
        # matrix_utils.write_matrix_to_file(
        #     PATH_TO_FILE+filename, self_energy_lesser_diagblk)
        # filename = ("self_energy_lesser_" + str(i) + "_upperblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
        # matrix_utils.write_matrix_to_file(
        #     PATH_TO_FILE+filename, self_energy_lesser_upperblk)
        # filename = ("self_energy_lesser_" + str(i) + "_lowerblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
        # matrix_utils.write_matrix_to_file(
        #     PATH_TO_FILE+filename, self_energy_lesser_lowerblk)
        
        # filename = ("self_energy_greater_" + str(i) + "_diagblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
        # matrix_utils.write_matrix_to_file(
        #     PATH_TO_FILE+filename, self_energy_greater_diagblk)
        # filename = ("self_energy_greater_" + str(i) + "_upperblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
        # matrix_utils.write_matrix_to_file(
        #     PATH_TO_FILE+filename, self_energy_greater_upperblk)
        # filename = ("self_energy_greater_" + str(i) + "_lowerblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
        # matrix_utils.write_matrix_to_file(
        #     PATH_TO_FILE+filename, self_energy_greater_lowerblk)

        # print("Generating matrix lesser and greater")
        # lesser_inv = inv_matrix @ self_energy_lesser @ inv_matrix.conj().T
        # greater_inv = inv_matrix @ self_energy_greater @ inv_matrix.conj().T

        # lesser_diagblk = matrix_utils.extract_diagonal_blocks(
        #     lesser_inv, MAT_SIZE, BLOCKSIZE)
        # lesser_upperblk = matrix_utils.extract_offdiagonal_blocks(
        #     lesser_inv, MAT_SIZE, BLOCKSIZE, 1)
        # lesser_lowerblk = matrix_utils.extract_offdiagonal_blocks(
        #     lesser_inv, MAT_SIZE, BLOCKSIZE, -1)

        # greater_diagblk = matrix_utils.extract_diagonal_blocks(
        #     greater_inv, MAT_SIZE, BLOCKSIZE)
        # greater_upperblk = matrix_utils.extract_offdiagonal_blocks(
        #     greater_inv, MAT_SIZE, BLOCKSIZE, 1)
        # greater_lowerblk = matrix_utils.extract_offdiagonal_blocks(
        #     greater_inv, MAT_SIZE, BLOCKSIZE, -1)

        # filename = ("lesser_" + str(i) + "_inv_diagblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
        # matrix_utils.write_matrix_to_file(
        #     PATH_TO_FILE+filename, lesser_diagblk)
        # filename = ("lesser_" + str(i) + "_inv_upperblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
        # matrix_utils.write_matrix_to_file(
        #     PATH_TO_FILE+filename, lesser_upperblk)
        # filename = ("lesser_" + str(i) + "_inv_lowerblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
        # matrix_utils.write_matrix_to_file(
        #     PATH_TO_FILE+filename, lesser_lowerblk)

        # filename = ("greater_" + str(i) + "_inv_diagblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
        # matrix_utils.write_matrix_to_file(
        #     PATH_TO_FILE+filename, greater_diagblk)
        # filename = ("greater_" + str(i) + "_inv_upperblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
        # matrix_utils.write_matrix_to_file(
        #     PATH_TO_FILE+filename, greater_upperblk)
        # filename = ("greater_" + str(i) + "_inv_lowerblk_" + str(MAT_SIZE) + "_"+ str(BLOCKSIZE) + "_" + str(BATCHSIZE) +".bin")
        # matrix_utils.write_matrix_to_file(
        #     PATH_TO_FILE+filename, greater_lowerblk)
