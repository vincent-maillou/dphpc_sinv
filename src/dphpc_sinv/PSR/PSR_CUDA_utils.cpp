#include "PSR.h"

// Function 1 Extract Sub Block of Matrix from GPU memory into contiguous buffer
void extract_subblock_from_GPU(cuDoubleComplex* subblock, cuDoubleComplex* GPU_matrix, int blocksize, int stride, int rowBlock, int colBlock) {
    // Extracts a subblock from a GPU matrix into a contiguous buffer
    // subblock: pointer to the contiguous buffer
    // GPU_matrix: pointer to the GPU matrix
    // blocksize: size of the blocks in the matrix
    // stride: stride of the matrix
    // rowBlock: row index of the subblock
    // colBlock: column index of the subblock

    int GPU_row_offset = rowBlock * blocksize;
    int GPU_col_offset = colBlock * blocksize * stride;

    cudaErrchk(cudaMemcpy2D(subblock, blocksize * sizeof(std::complex<double>), GPU_matrix + GPU_row_offset + GPU_col_offset, stride * sizeof(std::complex<double>), blocksize * sizeof(std::complex<double>), blocksize, cudaMemcpyDeviceToDevice));
}

void copy_rowblocks_buffer2GPU(cuDoubleComplex* GPU_matrix, cuDoubleComplex* CPU_buffer, int blocksize, int stride, int rowBlocks, int rowBlock, int colBlock, int buffBlock) {
    // Copies a contiguous CPU buffer into a GPU pattern matrix
    // GPU_matrix: pointer to the GPU matrix
    // CPU_matrix: pointer to the CPU matrix
    // blocksize: size of the blocks in the matrix
    // stride: stride of the matrix
    // rowBlocks: number of row blocks in the matrix
    // colBlocks: number of column blocks in the matrix
    int GPU_row_offset = rowBlock * blocksize;
    int GPU_col_offset = colBlock * blocksize * stride;
    int buff_offset = buffBlock * blocksize * blocksize;

    cudaErrchk(cudaMemcpy2D(GPU_matrix + GPU_row_offset + GPU_col_offset, stride * sizeof(std::complex<double>), CPU_buffer + buff_offset, blocksize * sizeof(std::complex<double>), blocksize * sizeof(std::complex<double>), rowBlocks * blocksize, cudaMemcpyHostToDevice));
   
}


void copy_rowblocks_GPU2buffer(cuDoubleComplex* GPU_matrix, cuDoubleComplex* CPU_buffer, int blocksize, int stride, int rowBlocks, int rowBlock, int colBlock, int buffBlock) {
    // Copies a contiguous CPU buffer into a GPU pattern matrix
    // GPU_matrix: pointer to the GPU matrix
    // CPU_matrix: pointer to the CPU matrix
    // blocksize: size of the blocks in the matrix
    // stride: stride of the matrix
    // rowBlocks: number of row blocks in the matrix
    // colBlocks: number of column blocks in the matrix
    int GPU_row_offset = rowBlock * blocksize;
    int GPU_col_offset = colBlock * blocksize * stride;
    int buff_offset = buffBlock * blocksize * blocksize;

     cudaErrchk(cudaMemcpy2D(CPU_buffer + buff_offset, blocksize * sizeof(std::complex<double>), GPU_matrix + GPU_row_offset + GPU_col_offset,  stride * sizeof(std::complex<double>), blocksize * sizeof(std::complex<double>), rowBlocks * blocksize, cudaMemcpyDeviceToHost));
}

void copy_rowblocks_GPU2GPU(cuDoubleComplex* GPU_matrix1, cuDoubleComplex* GPU_matrix2, int blocksize, int stride1, int stride2, int rowBlocks, int rowBlock1, int colBlock1, int rowBlock2, int colBlock2) {
    // Copies a contiguous CPU buffer into a GPU pattern matrix
    // GPU_matrix: pointer to the GPU matrix
    // CPU_matrix: pointer to the CPU matrix
    // blocksize: size of the blocks in the matrix
    // stride: stride of the matrix
    // rowBlocks: number of row blocks in the matrix
    // colBlocks: number of column blocks in the matrix
    int GPU_row_offset1 = rowBlock1 * blocksize;
    int GPU_col_offset1 = colBlock1* blocksize * stride1;
    int GPU_row_offset2 = rowBlock2 * blocksize;
    int GPU_col_offset2 = colBlock2* blocksize * stride2;

    cudaErrchk(cudaMemcpy2D(GPU_matrix1 + GPU_row_offset1 + GPU_col_offset1, stride1 * sizeof(std::complex<double>), GPU_matrix2 + GPU_row_offset2 + GPU_col_offset2, stride2 * sizeof(std::complex<double>), blocksize * sizeof(std::complex<double>), rowBlocks * blocksize, cudaMemcpyDeviceToDevice));
}


// Function 2 Create identity matrix on GPU
void create_identity_GPU(cuDoubleComplex* I, int matrix_size) {
    // Initialize Idendity Matrix and a Copy of it which is also used as a buffer for the result of inversions.
    // create right hand side identity matrix
    std::complex<double>* I_array = new std::complex<double> [matrix_size];
    std::fill(I_array, I_array + matrix_size, std::complex<double>(1.0, 0.0));

    cudaErrchk(cudaMemcpy2D(I, (matrix_size + 1) * sizeof(cuDoubleComplex), (cuDoubleComplex*)I_array, sizeof(cuDoubleComplex), sizeof(cuDoubleComplex), matrix_size, cudaMemcpyHostToDevice));

    delete[] I_array;

}

// Function 3 Invert GPU Matrix using dense cusolver
void invert_GPU_matrix(cuDoubleComplex* GPU_matrix, cuDoubleComplex* I, int blocksize, cusolverDnHandle_t cusolverH, cuDoubleComplex* d_work, int* info_d, int info_h, int* ipiv_d) {
    // Inverts a GPU matrix using dense cusolver
    // GPU_matrix: pointer to the GPU matrix
    // blocksize: size of the blocks in the matrix
    // stride: stride of the matrix
    // rowBlocks: number of row blocks in the matrix
    // colBlocks: number of column blocks in the matrix

      cusolverErrchk(cusolverDnZgetrf(cusolverH, blocksize, blocksize,
                                      GPU_matrix, blocksize, d_work, ipiv_d, info_d));
      
      cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));
      int rank, size;

      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      if (info_h != 0) {
          std::printf("Error: LU factorization failed in invert_GPU_matrix\n");
          std::printf("rank = %d\n", rank);
          std::printf("info_h = %d\n", info_h);
          }

      cusolverErrchk(cusolverDnZgetrs(cusolverH, CUBLAS_OP_N, blocksize, blocksize,
                      GPU_matrix, blocksize, ipiv_d,
                      I, blocksize, info_d));

      cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));

      if (info_h != 0) {
          std::printf("Error: Inversion failed\n");
          std::printf("info_h = %d\n", info_h);
          } 
}

// Function 4 Invert GPU Matrix including creating all the buffers and probing the workspace
void invert_GPU_matrix_complete(cuDoubleComplex* GPU_matrix, cuDoubleComplex* A_schur_gpu, int blocksize, cusolverDnHandle_t cusolverH) {
    // Inverts a GPU matrix using dense cusolver
    // GPU_matrix: pointer to the GPU matrix
    // blocksize: size of the blocks in the matrix
    // stride: stride of the matrix
    // Allocation of buffers for GPU inversion
    int info_h = 0;
    int *ipiv_d = NULL;
    int *info_d = NULL;
    cudaErrchk(cudaMalloc((void**)&info_d, sizeof(int)))
    cudaErrchk(cudaMalloc((void**)&ipiv_d, blocksize * sizeof(int)));

    cuDoubleComplex *buffer = NULL;
    int bufferSize = 0;

    // Create workspace
    cusolverErrchk(cusolverDnZgetrf_bufferSize(cusolverH, blocksize, blocksize, GPU_matrix, blocksize, &bufferSize));
    cudaErrchk(cudaMalloc((void**)&buffer, sizeof(cuDoubleComplex)*bufferSize));
    
    // Create identity matrix
    create_identity_GPU(reinterpret_cast<cuDoubleComplex*>(GPU_matrix), blocksize);

    // Invert matrix
    invert_GPU_matrix(A_schur_gpu, GPU_matrix, blocksize, cusolverH, buffer, info_d, info_h, ipiv_d);

    if(ipiv_d) {
	    cudaErrchk(cudaFree(ipiv_d));
    }
    if(info_d) {
	    cudaErrchk(cudaFree(info_d));
    }
    if(A_schur_gpu) {
	    cudaErrchk(cudaFree(A_schur_gpu));
    }
    if(buffer) {
	    cudaErrchk(cudaFree(buffer));
    }
}

// Function 4 Use cudaMemcpy2dToArray to copy a contiguous CPU buffer into the GPU A_reduced_schur pattern matrix
void copy_to_GPU_pattern(std::complex<double>* GPU_matrix, std::complex<double>* CPU_matrix, int blocksize, int stride, int rowBlocks, int colBlocks) {
    // Copies a contiguous CPU buffer into a GPU pattern matrix
    // GPU_matrix: pointer to the GPU matrix
    // CPU_matrix: pointer to the CPU matrix
    // blocksize: size of the blocks in the matrix
    // stride: stride of the matrix
    // rowBlocks: number of row blocks in the matrix
    // colBlocks: number of column blocks in the matrix

    int subblock_size = blocksize * blocksize;

    int subblock_row_offset = 0;
    int subblock_col_offset = 0;

    int GPU_row_offset = 0;
    int GPU_col_offset = 0;

    // Copy each subblock
    for (int rowBlock = 0; rowBlock < rowBlocks; rowBlock++) {
        for (int colBlock = 0; colBlock < colBlocks; colBlock++) {
            subblock_row_offset = rowBlock * blocksize * stride;
            subblock_col_offset = colBlock * blocksize;

            GPU_row_offset = rowBlock * blocksize * stride * rowBlocks;
            GPU_col_offset = colBlock * blocksize * colBlocks;

            cudaMemcpy2D(GPU_matrix + GPU_row_offset + GPU_col_offset, stride * sizeof(std::complex<double>), CPU_matrix + subblock_row_offset + subblock_col_offset, subblock_size * sizeof(std::complex<double>), blocksize * sizeof(std::complex<double>), blocksize, cudaMemcpyHostToDevice);
        }
    }
}


