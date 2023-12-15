# Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.

from dphpc_sinv.Utils import matrix_utils
import numpy as np

SEED = 8000
MAT_SIZE = 110
BLOCKSIZE = 11
BATCHSIZE = 13
PATH_TO_FILE = "/usr/scratch/mont-fort17/almaeder/rgf_test/"

if __name__ == "__main__":

    filename = "batched_matrix_parameters.txt"
    matrix_utils.write_matrix_parameters_batched(
        PATH_TO_FILE+filename, MAT_SIZE, BLOCKSIZE, BATCHSIZE)

    # Generate random matrices
    for i in range(BATCHSIZE):
        print("Generating matrix " + str(i+1) + " of " + str(BATCHSIZE))

        system_matrix = matrix_utils.generateBandedMatrix(MAT_SIZE, BLOCKSIZE, SEED) + i * np.eye(MAT_SIZE)

        self_energy_lesser = matrix_utils.generateBandedMatrix(MAT_SIZE, BLOCKSIZE, SEED+1) + 2*i * np.eye(MAT_SIZE)
        self_energy_lesser = self_energy_lesser - self_energy_lesser.conj().T

        self_energy_greater = matrix_utils.generateBandedMatrix(MAT_SIZE, BLOCKSIZE, SEED+2) + 3*i * np.eye(MAT_SIZE)
        self_energy_greater = self_energy_greater - self_energy_greater.conj().T

        # Assert system_matrix to be invertible
        assert np.allclose(np.linalg.inv(system_matrix) @ system_matrix, np.eye(MAT_SIZE))
        assert np.linalg.det(system_matrix) != 0

        # Compute inv
        inv_matrix = np.linalg.inv(system_matrix)

        lesser_inv = inv_matrix @ self_energy_lesser @ inv_matrix.conj().T
        greater_inv = inv_matrix @ self_energy_greater @ inv_matrix.conj().T



        # Extract diagonal and off-diagonal blocks
        system_matrix_diagblk = matrix_utils.extract_diagonal_blocks(
            system_matrix, MAT_SIZE, BLOCKSIZE)
        system_matrix_upperblk = matrix_utils.extract_offdiagonal_blocks(
            system_matrix, MAT_SIZE, BLOCKSIZE, 1)
        system_matrix_lowerblk = matrix_utils.extract_offdiagonal_blocks(
            system_matrix, MAT_SIZE, BLOCKSIZE, -1)

        self_energy_lesser_diagblk = matrix_utils.extract_diagonal_blocks(
            self_energy_lesser, MAT_SIZE, BLOCKSIZE)
        self_energy_lesser_upperblk = matrix_utils.extract_offdiagonal_blocks(
            self_energy_lesser, MAT_SIZE, BLOCKSIZE, 1)
        self_energy_lesser_lowerblk = matrix_utils.extract_offdiagonal_blocks(
            self_energy_lesser, MAT_SIZE, BLOCKSIZE, -1)
        
        self_energy_greater_diagblk = matrix_utils.extract_diagonal_blocks(
            self_energy_greater, MAT_SIZE, BLOCKSIZE)
        self_energy_greater_upperblk = matrix_utils.extract_offdiagonal_blocks(
            self_energy_greater, MAT_SIZE, BLOCKSIZE, 1)
        self_energy_greater_lowerblk = matrix_utils.extract_offdiagonal_blocks(
            self_energy_greater, MAT_SIZE, BLOCKSIZE, -1)

        lesser_diagblk = matrix_utils.extract_diagonal_blocks(
            lesser_inv, MAT_SIZE, BLOCKSIZE)
        lesser_upperblk = matrix_utils.extract_offdiagonal_blocks(
            lesser_inv, MAT_SIZE, BLOCKSIZE, 1)
        lesser_lowerblk = matrix_utils.extract_offdiagonal_blocks(
            lesser_inv, MAT_SIZE, BLOCKSIZE, -1)

        greater_diagblk = matrix_utils.extract_diagonal_blocks(
            greater_inv, MAT_SIZE, BLOCKSIZE)
        greater_upperblk = matrix_utils.extract_offdiagonal_blocks(
            greater_inv, MAT_SIZE, BLOCKSIZE, 1)
        greater_lowerblk = matrix_utils.extract_offdiagonal_blocks(
            greater_inv, MAT_SIZE, BLOCKSIZE, -1)


        # Save matrices to file
        filename = "system_matrix_" + str(i) + "_diagblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, system_matrix_diagblk)
        filename = "system_matrix_" + str(i) + "_upperblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, system_matrix_upperblk)
        filename = "system_matrix_" + str(i) + "_lowerblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, system_matrix_lowerblk)

        filename = "self_energy_lesser_" + str(i) + "_diagblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, self_energy_lesser_diagblk)
        filename = "self_energy_lesser_" + str(i) + "_upperblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, self_energy_lesser_upperblk)
        filename = "self_energy_lesser_" + str(i) + "_lowerblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, self_energy_lesser_lowerblk)
        
        filename = "self_energy_greater_" + str(i) + "_diagblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, self_energy_greater_diagblk)
        filename = "self_energy_greater_" + str(i) + "_upperblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, self_energy_greater_upperblk)
        filename = "self_energy_greater_" + str(i) + "_lowerblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, self_energy_greater_lowerblk)

        filename = "lesser_" + str(i) + "_inv_diagblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, lesser_diagblk)
        filename = "lesser_" + str(i) + "_inv_upperblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, lesser_upperblk)
        filename = "lesser_" + str(i) + "_inv_lowerblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, lesser_lowerblk)

        filename = "greater_" + str(i) + "_inv_diagblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, greater_diagblk)
        filename = "greater_" + str(i) + "_inv_upperblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, greater_upperblk)
        filename = "greater_" + str(i) + "_inv_lowerblk.bin"
        matrix_utils.write_matrix_to_file(
            PATH_TO_FILE+filename, greater_lowerblk)
