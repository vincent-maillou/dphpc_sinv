import numpy as np

SEED = 63
MAT_SIZE = 10
BLOCKSIZE = 1



def generate_random_matrix():
    np.random.seed(SEED)
    matrix = np.random.rand(MAT_SIZE, MAT_SIZE) + 1j*np.random.rand(MAT_SIZE, MAT_SIZE)
    
    return matrix


def write_matrix_to_file(
    path_to_file: str,
    matrix: np.ndarray,
    matrix_size: int,
    blocksize: int
):
    with open(path_to_file, "wb") as f:
        f.write(matrix.tobytes())
        

def print_matrix(
    matrix: np.ndarray, 
    matrix_size: int
):
    for i in range(matrix_size):
        for j in range(matrix_size):
            print(matrix[i, j], end=" ")
        print()
    
    
if __name__ == "__main__":
    # Generate random matrix
    matrix = generate_random_matrix()

    # Compute inverse
    inv_matrix = np.linalg.inv(matrix)

    # Save inverse matrix to binary file
    path_to_file = "../../tests/tests_cases/"
    
    filename = "matrix_0.bin"
    write_matrix_to_file(path_to_file+filename, matrix, MAT_SIZE, 1)
    
    filename = "matrix_0_inverse.bin"
    write_matrix_to_file(path_to_file+filename, inv_matrix, MAT_SIZE, 1)

    #print_matrix(matrix, MAT_SIZE)
    
    
    