// Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
#include "single_sparse_retarded.h"

bool rgf_sparse_matrix_does_not_fit_gpu_memory_with_copy_compute_overlap(
    unsigned int blocksize,
    unsigned int matrix_size,
    int* diag_nnz,
    int* upper_nnz,
    int* lower_nnz,
    complex_h **diagblk_data_h,
    int **diagblk_indices_h,
    int **diagblk_indptr_h,
    complex_h **upperblk_data_h,
    int **upperblk_indices_h,
    int **upperblk_indptr_h,
    complex_h **lowerblk_data_h,
    int **lowerblk_indices_h,
    int **lowerblk_indptr_h,
    complex_h *inv_diagblk_h,
    complex_h *inv_upperblk_h,
    complex_h *inv_lowerblk_h)
{
    if(matrix_size % blocksize != 0){
        printf("Error: matrix_size is not a multiple of blocksize\n");
        return false;
    }
    unsigned int n_blocks = matrix_size / blocksize;
    bool success = true;

    // Init cuda stuff

    // need multiple streams for overlap
    int number_streams = 3;
    cudaStream_t stream[number_streams];
    for(int i = 0; i < number_streams; i++){
        cudaErrchk(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));
    }

    cusolverDnHandle_t cusolverDn_handle[number_streams];
    for(int i = 0; i < number_streams; i++){
        cusolverErrchk(cusolverDnCreate(&cusolverDn_handle[i]));
        cusolverErrchk(cusolverDnSetStream(cusolverDn_handle[i], stream[i]));
    }

    cublasHandle_t cublas_handle[number_streams];
    for(int i = 0; i < number_streams; i++){
        cublasErrchk(cublasCreate(&cublas_handle[i]));
        cublasErrchk(cublasSetStream(cublas_handle[i], stream[i]));
    }
    cusparseHandle_t cusparse_handle[number_streams];
    for(int i = 0; i < number_streams; i++){
        cusparseErrchk(cusparseCreate(&cusparse_handle[i]));
        cusparseErrchk(cusparseSetStream(cusparse_handle[i], stream[i]));
    }

    // sparse matrix desciptors
    cusparseSpMatDescr_t diagblk_descr_forward[n_blocks];
    cusparseSpMatDescr_t upperblk_descr_forward[n_blocks-1];
    cusparseSpMatDescr_t lowerblk_descr_forward[n_blocks-1];

    cusparseSpMatDescr_t upperblk_descr_backward[n_blocks-1];
    cusparseSpMatDescr_t lowerblk_descr_backward[n_blocks-1];





    cudaEvent_t schur_inverted[n_blocks];
    for(unsigned int i = 0; i < n_blocks; i++){
        cudaErrchk(cudaEventCreate(&schur_inverted[i]))
    }
    cudaEvent_t unload[n_blocks];
    for(unsigned int i = 0; i < n_blocks; i++){
        cudaErrchk(cudaEventCreate(&unload[i]))
    }
    complex_d alpha;
    complex_d beta;
    int stream_memload = 1;
    int stream_compute = 0;
    int stream_memunload = 2;




    // not allowed to load full matrix to device
    // allocate memory for the blocks
    
    complex_d* diagblk_data_d[2];
    int* diagblk_indices_d[2];
    int* diagblk_indptr_d[2];
    complex_d* upperblk_data_d[2];
    int* upperblk_indices_d[2];
    int* upperblk_indptr_d[2];
    complex_d* lowerblk_data_d[2];
    int* lowerblk_indices_d[2];
    int* lowerblk_indptr_d[2];

    int max_diag_nnz = 0;
    int max_upper_nnz = 0;
    int max_lower_nnz = 0;
    // find maximum nnz in each block
    // TODO if nnz is varying a lot, this is not optimal
    // since we allocate more memory than needed
    // better would be to allocate memory for each block before loading
    // and deallocate after use
    for(unsigned i = 0; i < n_blocks; i++){
        if(diag_nnz[i] > max_diag_nnz){
            max_diag_nnz = diag_nnz[i];
        }
    }
    for(unsigned i = 0; i < n_blocks -1; i++){
        if(upper_nnz[i] > max_upper_nnz){
            max_upper_nnz = upper_nnz[i];
        }
        if(lower_nnz[i] > max_lower_nnz){
            max_lower_nnz = lower_nnz[i];
        }    
    }
    // allocate single blocks of the matrix
    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&diagblk_data_d[i], max_diag_nnz * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&diagblk_indices_d[i], max_diag_nnz * sizeof(int)));
        cudaErrchk(cudaMalloc((void**)&diagblk_indptr_d[i], (blocksize+1) * sizeof(int)));
        cudaErrchk(cudaMalloc((void**)&upperblk_data_d[i], max_upper_nnz * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&upperblk_indices_d[i], max_upper_nnz * sizeof(int)));
        cudaErrchk(cudaMalloc((void**)&upperblk_indptr_d[i], (blocksize+1) * sizeof(int)));
        cudaErrchk(cudaMalloc((void**)&lowerblk_data_d[i], max_lower_nnz * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&lowerblk_indices_d[i], max_lower_nnz * sizeof(int)));
        cudaErrchk(cudaMalloc((void**)&lowerblk_indptr_d[i], (blocksize+1) * sizeof(int)));
    }



    for(unsigned int i = 0; i < n_blocks - 1; i++){
        int stream_memload_forward = (i+1) % 2;

        // create sparse matrix descriptors
        cusparseErrchk(
            cusparseCreateCsr(
                &diagblk_descr_forward[i+1],
                blocksize,
                blocksize,
                diag_nnz[i+1],
                diagblk_indptr_d[stream_memload_forward],
                diagblk_indices_d[stream_memload_forward],
                diagblk_data_d[stream_memload_forward],
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                CUDA_C_64F
            )
        );

        cusparseErrchk(
            cusparseCreateCsr(
                &upperblk_descr_forward[i],
                blocksize,
                blocksize,
                upper_nnz[i],
                upperblk_indptr_d[stream_memload_forward],
                upperblk_indices_d[stream_memload_forward],
                upperblk_data_d[stream_memload_forward],
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                CUDA_C_64F
            )
        );
        cusparseErrchk(
            cusparseCreateCsr(
                &lowerblk_descr_forward[i],
                blocksize,
                blocksize,
                lower_nnz[i],
                lowerblk_indptr_d[stream_memload_forward],
                lowerblk_indices_d[stream_memload_forward],
                lowerblk_data_d[stream_memload_forward],
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                CUDA_C_64F
            )
        );
    }
    for(int i = n_blocks-1; i > 0; --i){
        int stream_memload_backward = ((n_blocks-1) % 2 + (i - n_blocks + 2) ) % 2;
        cusparseErrchk(
            cusparseCreateCsr(
                &upperblk_descr_backward[i-1],
                blocksize,
                blocksize,
                upper_nnz[i-1],
                upperblk_indptr_d[stream_memload_backward],
                upperblk_indices_d[stream_memload_backward],
                upperblk_data_d[stream_memload_backward],
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                CUDA_C_64F
            )
        );
        cusparseErrchk(
            cusparseCreateCsr(
                &lowerblk_descr_backward[i-1],
                blocksize,
                blocksize,
                lower_nnz[i-1],
                lowerblk_indptr_d[stream_memload_backward],
                lowerblk_indices_d[stream_memload_backward],
                lowerblk_data_d[stream_memload_backward],
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                CUDA_C_64F
            )
        );
    }



    // allocate memory for the inverse
    complex_d* inv_diagblk_d = NULL;
    complex_d* inv_diagblk_small_d[2];
    complex_d* inv_upperblk_d = NULL;
    complex_d* inv_lowerblk_d = NULL;

    cudaErrchk(cudaMalloc((void**)&inv_diagblk_d, blocksize * blocksize * sizeof(complex_d)));
    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&inv_diagblk_small_d[i], blocksize * blocksize * sizeof(complex_d)));
    }
    
    cudaErrchk(cudaMalloc((void**)&inv_upperblk_d, blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&inv_lowerblk_d, blocksize * blocksize * sizeof(complex_d)));

    cusparseDnMatDescr_t dense_descr_in;
    cusparseDnMatDescr_t dense_descr_out;
    // set them inside the loop
    cusparseCreateDnMat(
        &dense_descr_in,
        blocksize,
        blocksize,
        blocksize,
        inv_diagblk_d,
        CUDA_C_64F,
        CUSPARSE_ORDER_COL
    );
    cusparseCreateDnMat(
        &dense_descr_out,
        blocksize,
        blocksize,
        blocksize,
        inv_diagblk_d,
        CUDA_C_64F,
        CUSPARSE_ORDER_COL
    );

    // ghetto solution
    // one would have to call cudaMalloc for all the different sparsity patterns
    // of the blocks
    // but cudaMalloc synchronizes all streams
    // solution: allocate just large enough buffer
    int buffer_size_spmm = 2*blocksize*blocksize;
    complex_d *buffer_spmm = NULL;
    cudaErrchk(cudaMalloc((void**)&buffer_spmm, sizeof(complex_d) * buffer_size_spmm));



    // cusolverSp only has for single rhs
    // transform first block to dense

    cudaErrchk(cudaMemcpy(diagblk_data_d[stream_compute],
                reinterpret_cast<const complex_d*>(diagblk_data_h[0]),
                diag_nnz[0]* sizeof(complex_d), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(diagblk_indices_d[stream_compute],
                reinterpret_cast<const int*>(diagblk_indices_h[0]),
                diag_nnz[0] * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(diagblk_indptr_d[stream_compute],
                reinterpret_cast<const int*>(diagblk_indptr_h[0]),
                (blocksize+1) * sizeof(int), cudaMemcpyHostToDevice));

    cusparseErrchk(
        cusparseCreateCsr(
            &diagblk_descr_forward[0],
            blocksize,
            blocksize,
            diag_nnz[0],
            diagblk_indptr_d[stream_compute],
            diagblk_indices_d[stream_compute],
            diagblk_data_d[stream_compute],
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_C_64F
        )
    );


    complex_d* first_block_d = NULL;
    cudaErrchk(cudaMalloc((void**)&first_block_d, blocksize * blocksize * sizeof(complex_d)));
    
    cusparseErrchk(cusparseDnMatSetValues(dense_descr_out, first_block_d));

    cusparseErrchk(cusparseSparseToDense(
        cusparse_handle[stream_compute],
        diagblk_descr_forward[0],
        dense_descr_out,
        CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
        buffer_spmm
    ));


    //memory for pivoting
    int *ipiv_d = NULL;
    int *info_d = NULL;
    cudaErrchk(cudaMalloc((void**)&info_d, sizeof(int)))
    cudaErrchk(cudaMalloc((void**)&ipiv_d, blocksize*sizeof(int)));


    // create right hand side identity matrix
    complex_h* identity_h;
    cudaErrchk(cudaMallocHost((void**)&identity_h, blocksize * blocksize * sizeof(complex_h)));
    complex_d* identity_d = NULL;
    complex_d* identity_cpy_d = NULL;
    cudaErrchk(cudaMalloc((void**)&identity_d, blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&identity_cpy_d, blocksize * blocksize * sizeof(complex_d)));


    for(unsigned int i = 0; i < blocksize * blocksize; i++){
        identity_h[i] = 0.0;
        if(i / blocksize == i % blocksize){
            identity_h[i] = 1.0;
        }
    }

    //figure out extra amount of memory needed
    complex_d *buffer_getrf = NULL;
    int buffer_size_getrf = 0;
    cusolverErrchk(cusolverDnZgetrf_bufferSize(cusolverDn_handle[stream_compute], blocksize, blocksize,
                                            (complex_d *)first_block_d,
                                              blocksize, &buffer_size_getrf));
    cudaErrchk(cudaMalloc((void**)&buffer_getrf, sizeof(complex_d) * buffer_size_getrf));

    // ----- END OF INIT SECTION -----

    // init right hand side identity matrix on device for backsub
    cudaErrchk(cudaMemcpyAsync(identity_d, reinterpret_cast<const complex_d*>(identity_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_compute]));
    cudaErrchk(cudaMemcpyAsync(inv_diagblk_d, identity_d,
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
    

    cusolverErrchk(cusolverDnZgetrf(cusolverDn_handle[stream_compute], blocksize, blocksize,
                                first_block_d, blocksize, buffer_getrf, ipiv_d, info_d));
    

    //back substitution
    cusolverErrchk(cusolverDnZgetrs(cusolverDn_handle[stream_compute], CUBLAS_OP_N, blocksize,
                                    blocksize, first_block_d, blocksize, ipiv_d,
                                    inv_diagblk_d, blocksize, info_d));



    // record finishing the inverse of the first block
    cudaErrchk(cudaEventRecord(schur_inverted[0], stream[stream_compute]));


    //wait for the inverse of the first block
    cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[0]));
    // 0. Inverse of the first block
    cudaErrchk(cudaMemcpyAsync(inv_diagblk_h, inv_diagblk_d,
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
    // unloading finished
    cudaErrchk(cudaEventRecord(unload[0], stream[stream_memunload]));


    // first memcpy happens before loop
    cudaErrchk(cudaMemcpyAsync(diagblk_data_d[stream_memload],
                reinterpret_cast<const complex_d*>(diagblk_data_h[1]),
                diag_nnz[1]* sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(diagblk_indices_d[stream_memload],
                reinterpret_cast<const int*>(diagblk_indices_h[1]),
                diag_nnz[1] * sizeof(int), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(diagblk_indptr_d[stream_memload],
                reinterpret_cast<const int*>(diagblk_indptr_h[1]),
                (blocksize+1) * sizeof(int), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(upperblk_data_d[stream_memload],
                reinterpret_cast<const complex_d*>(upperblk_data_h[0]),
               upper_nnz[0] * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(upperblk_indices_d[stream_memload],
                reinterpret_cast<const int*>(upperblk_indices_h[0]),
                upper_nnz[0] * sizeof(int), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(upperblk_indptr_d[stream_memload],
                reinterpret_cast<const int*>(upperblk_indptr_h[0]),
                (blocksize+1) * sizeof(int), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(lowerblk_data_d[stream_memload],
                reinterpret_cast<const complex_d*>(lowerblk_data_h[0]),
                lower_nnz[0] * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(lowerblk_indices_d[stream_memload],
                reinterpret_cast<const int*>(lowerblk_indices_h[0]),
                lower_nnz[0] * sizeof(int), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(lowerblk_indptr_d[stream_memload],
                reinterpret_cast<const int*>(lowerblk_indptr_h[0]),
                (blocksize+1) * sizeof(int), cudaMemcpyHostToDevice, stream[stream_memload]));



    // // 1. Forward substitution (performed left to right)
    for (unsigned int i = 1; i < n_blocks; ++i) {


        int stream_memload = (i+1) % 2;
        int stream_compute = i % 2;
        int stream_memunload = 2;


        if(i < n_blocks-1){
            // load the blocks for the next iteration
            cudaErrchk(cudaMemcpyAsync(diagblk_data_d[stream_memload],
                        reinterpret_cast<const complex_d*>(diagblk_data_h[i+1]),
                        diag_nnz[i+1]* sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(diagblk_indices_d[stream_memload],
                        reinterpret_cast<const int*>(diagblk_indices_h[i+1]),
                        diag_nnz[i+1] * sizeof(int), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(diagblk_indptr_d[stream_memload],
                        reinterpret_cast<const int*>(diagblk_indptr_h[i+1]),
                        (blocksize+1) * sizeof(int), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(upperblk_data_d[stream_memload],
                        reinterpret_cast<const complex_d*>(upperblk_data_h[i]),
                        upper_nnz[i] * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(upperblk_indices_d[stream_memload],
                        reinterpret_cast<const int*>(upperblk_indices_h[i]),
                        upper_nnz[i] * sizeof(int), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(upperblk_indptr_d[stream_memload],
                        reinterpret_cast<const int*>(upperblk_indptr_h[i]),
                        (blocksize+1) * sizeof(int), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(lowerblk_data_d[stream_memload],
                        reinterpret_cast<const complex_d*>(lowerblk_data_h[i]),
                        lower_nnz[i] * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(lowerblk_indices_d[stream_memload],
                        reinterpret_cast<const int*>(lowerblk_indices_h[i]),
                        lower_nnz[i] * sizeof(int), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(lowerblk_indptr_d[stream_memload],
                        reinterpret_cast<const int*>(lowerblk_indptr_h[i]),
                        (blocksize+1) * sizeof(int), cudaMemcpyHostToDevice, stream[stream_memload]));

        }

        //wait for the schur inverse from the previous iteration
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], schur_inverted[i-1]));

        // MatMul tmp = eig_lowerblk[i-1] * eig_inv_diagblk[i-1]
        // use the inv_diagblk_d from last iteration
        // use inv_lowerblk_d as tmp
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        //buffer_spmm

        cusparseErrchk(cusparseDnMatSetValues(dense_descr_in, inv_diagblk_d))
        cusparseErrchk(cusparseDnMatSetValues(dense_descr_out, inv_lowerblk_d))

        cusparseErrchk(cusparseSpMM(
            cusparse_handle[stream_compute],
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            lowerblk_descr_forward[i-1],
            dense_descr_in,
            &beta,
            dense_descr_out,
            CUDA_C_64F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            buffer_spmm
        ));


        // problem:
        // SpMM is C = op(A) @ op(B) + C
        // where only A is sparse
        // possible to have C = op(B) @ op(A) + op(C)
        // but for this C and B have to be in column major order
        // and not standard row major order
        // thus C have to be transposed before this kernel
        // and C has to be transposed back after the kernel

        cusparseErrchk(cusparseDnMatSetValues(dense_descr_in, inv_lowerblk_d));
        cusparseErrchk(cusparseDnMatSetValues(dense_descr_out, inv_upperblk_d));
        // C for spmm is dense
        cusparseErrchk(cusparseSparseToDense(
            cusparse_handle[stream_compute],
            diagblk_descr_forward[i],
            dense_descr_out,
            CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
            buffer_spmm
        ));

        // transpose both C 
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        // use identity_cpy_d as buffer for transposed
        cublasErrchk(
            cublasZgeam(
                cublas_handle[stream_compute],
                CUBLAS_OP_T, CUBLAS_OP_T,
                blocksize, blocksize,
                &alpha,
                inv_upperblk_d, blocksize,
                &beta,
                inv_lowerblk_d, blocksize,
                identity_cpy_d, blocksize
            )
        );
        cusparseErrchk(cusparseDnMatSetValues(dense_descr_out, identity_cpy_d));

        //MatMul tmp = eig_diagblk[i] - tmp * eig_upperblk[i-1]
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        cusparseErrchk(cusparseSpMM(
            cusparse_handle[stream_compute],
            CUSPARSE_OPERATION_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE,
            &alpha,
            upperblk_descr_forward[i-1],
            dense_descr_in,
            &beta,
            dense_descr_out,
            CUDA_C_64F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            buffer_spmm
        ));
        // use inv_upperblk_d as buffer for transposed
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(
            cublasZgeam(
                cublas_handle[stream_compute],
                CUBLAS_OP_T, CUBLAS_OP_T,
                blocksize, blocksize,
                &alpha,
                identity_cpy_d, blocksize,
                &beta,
                inv_lowerblk_d, blocksize,
                inv_upperblk_d, blocksize
            )
        );

        // more ghetto
        // inv_upperblk_d saves now schur complement


        // wait to not overwrite block to unload
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], unload[i-1]));
        //copy identity
        cudaErrchk(cudaMemcpyAsync(inv_diagblk_d, identity_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        // inverse schur complement
        cusolverErrchk(cusolverDnZgetrf(cusolverDn_handle[stream_compute], blocksize, blocksize,
                                    inv_upperblk_d,
                                    blocksize, buffer_getrf, ipiv_d, info_d));
        

        //back substitution
        cusolverErrchk(cusolverDnZgetrs(cusolverDn_handle[stream_compute], CUBLAS_OP_N, blocksize,
                                        blocksize,
                                        inv_upperblk_d,
                                        blocksize, ipiv_d,
                                        inv_diagblk_d, blocksize, info_d));
        // record finishing of computation in step i
        cudaErrchk(cudaEventRecord(schur_inverted[i], stream[stream_compute]));

        // wait to unload for the finish of computations
        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[i]));
        cudaErrchk(cudaMemcpyAsync(inv_diagblk_h + i*blocksize*blocksize,
                    inv_diagblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        // unloading finished
        cudaErrchk(cudaEventRecord(unload[i], stream[stream_memunload]));

    }

    int stream_memload_before = (n_blocks) % 2;
    int stream_compute_before = (n_blocks-1) % 2;

    // load sparse blocks for foward pass
    cudaErrchk(cudaMemcpyAsync(upperblk_data_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(upperblk_data_h[n_blocks-2]),
                upper_nnz[n_blocks-2] * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    cudaErrchk(cudaMemcpyAsync(upperblk_indices_d[stream_memload_before],
                reinterpret_cast<const int*>(upperblk_indices_h[n_blocks-2]),
                upper_nnz[n_blocks-2] * sizeof(int), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    cudaErrchk(cudaMemcpyAsync(upperblk_indptr_d[stream_memload_before],
                reinterpret_cast<const int*>(upperblk_indptr_h[n_blocks-2]),
                (blocksize+1) * sizeof(int), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    cudaErrchk(cudaMemcpyAsync(lowerblk_data_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(lowerblk_data_h[n_blocks-2]),
                lower_nnz[n_blocks-2] * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    cudaErrchk(cudaMemcpyAsync(lowerblk_indices_d[stream_memload_before],
                reinterpret_cast<const int*>(lowerblk_indices_h[n_blocks-2]),
                lower_nnz[n_blocks-2] * sizeof(int), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    cudaErrchk(cudaMemcpyAsync(lowerblk_indptr_d[stream_memload_before],
                reinterpret_cast<const int*>(lowerblk_indptr_h[n_blocks-2]),
                (blocksize+1) * sizeof(int), cudaMemcpyHostToDevice, stream[stream_memload_before]));


    // possible race condition with unloading of previous loop
    // not sure
    cudaErrchk(cudaMemcpyAsync(inv_diagblk_small_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(inv_diagblk_h  + (n_blocks-2)*blocksize*blocksize),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));


    // 2. Backward substitution (performed right to left)
    for(int i = n_blocks-2; i >= 0; --i){

        // fix stream compute to be the stream which loaded
        // blocks before the loop
        stream_memload = (stream_compute_before + (i - n_blocks + 2) ) % 2;
        stream_compute = (stream_memload_before + (i - n_blocks + 2) ) % 2;
        stream_memunload = 2;

        if(i > 0){

            cudaErrchk(cudaMemcpyAsync(upperblk_data_d[stream_memload],
                        reinterpret_cast<const complex_d*>(upperblk_data_h[i-1]),
                        upper_nnz[i-1] * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(upperblk_indices_d[stream_memload],
                        reinterpret_cast<const int*>(upperblk_indices_h[i-1]),
                        upper_nnz[i-1] * sizeof(int), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(upperblk_indptr_d[stream_memload],
                        reinterpret_cast<const int*>(upperblk_indptr_h[i-1]),
                        (blocksize+1) * sizeof(int), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(lowerblk_data_d[stream_memload],
                        reinterpret_cast<const complex_d*>(lowerblk_data_h[i-1]),
                        lower_nnz[i-1] * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(lowerblk_indices_d[stream_memload],
                        reinterpret_cast<const int*>(lowerblk_indices_h[i-1]),
                        lower_nnz[i-1] * sizeof(int), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(lowerblk_indptr_d[stream_memload],
                        reinterpret_cast<const int*>(lowerblk_indptr_h[i-1]),
                        (blocksize+1) * sizeof(int), cudaMemcpyHostToDevice, stream[stream_memload]));

            cudaErrchk(cudaMemcpyAsync(inv_diagblk_small_d[stream_memload],
                        reinterpret_cast<const complex_d*>(inv_diagblk_h  + (i-1)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));    
        }
    

        // wait for the block of the last iteration
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], schur_inverted[i+1]));

        //tmp = eig_inv_diagblk[i+1] * eig_lowerblk[i]
        // use identity_cpy_d as tmp
        // reuse inv_diagblk_d from last iteration
        // which is the last true inverse block

        // same problem as in forward pass
        // C = op(A) @ op(B) + C, but B is sparse in SpMM
        // but since beta is zero only the output has to be transposed

        cusparseErrchk(cusparseDnMatSetValues(dense_descr_in, inv_diagblk_d))
        // god have mercy on my soul
        // because using random buffers as temporary buffers
        // makes the code unreadable
        // identity is not used anymore since the inversions are done
        cusparseErrchk(cusparseDnMatSetValues(dense_descr_out, identity_d))


        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        // both operations transposed to get (A @ B)^T = B^T @ A^T where B is sparse
        // tmp = inv_diagblk_d * lowerblk
        // tmp^T = lowerblk^T * inv_diagblk_d^T
        cusparseErrchk(cusparseSpMM(
            cusparse_handle[stream_compute],
            CUSPARSE_OPERATION_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE,
            &alpha,
            lowerblk_descr_backward[i],
            dense_descr_in,
            &beta,
            dense_descr_out,
            CUDA_C_64F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            buffer_spmm
        ));

        //transpose output
        // B point does not matter since beta is zero
        cublasErrchk(
            cublasZgeam(
                cublas_handle[stream_compute],
                CUBLAS_OP_T, CUBLAS_OP_T,
                blocksize, blocksize,
                &alpha,
                identity_d, blocksize,
                &beta,
                NULL, blocksize,
                identity_cpy_d, blocksize
            )
        );


        // eig_inv_lowerblk[i] = -tmp*eig_inv_diagblk[i];
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        // use temporary first_block_d for inv_lowerblk_d
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            identity_cpy_d, blocksize,
            inv_diagblk_small_d[stream_compute], blocksize,
            &beta,
            first_block_d, blocksize));

        // tmp = eig_inv_diagblk[i] * eig_upperblk[i]
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        // again the same problem
        // dense descriptors wer already set,
        // thus no call to cusparseDnMatSetValues for dense_descr_out
        cusparseErrchk(cusparseDnMatSetValues(dense_descr_in, inv_diagblk_small_d[stream_compute]))
        cusparseErrchk(cusparseDnMatSetValues(dense_descr_out, identity_d))
        // both operations transposed to get (A @ B)^T = B^T @ A^T where B is sparse
        cusparseErrchk(cusparseSpMM(
            cusparse_handle[stream_compute],
            CUSPARSE_OPERATION_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE,
            &alpha,
            upperblk_descr_backward[i],
            dense_descr_in,
            &beta,
            dense_descr_out,
            CUDA_C_64F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            buffer_spmm
        ));


        //transpose output
        // B point does not matter since beta is zero
        cublasErrchk(
            cublasZgeam(
                cublas_handle[stream_compute],
                CUBLAS_OP_T, CUBLAS_OP_T,
                blocksize, blocksize,
                &alpha,
                identity_d, blocksize,
                &beta,
                NULL, blocksize,
                identity_cpy_d, blocksize
            )
        );


        //eig_inv_upperblk[i] = -tmp * eig_inv_diagblk[i+1];
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        // use temporary identity_d for inv_upperblk_d
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            identity_cpy_d, blocksize,
            inv_diagblk_d, blocksize,
            &beta,
            identity_d, blocksize));


        //eig_inv_diagblk[i] -= tmp * eig_inv_lowerblk[i];
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);

        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            identity_cpy_d, blocksize,
            first_block_d, blocksize,
            &beta,
            inv_diagblk_small_d[stream_compute], blocksize));
 

        // wait to not overwrite blocks to unload
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], unload[i+1]));

        // use allocated buffers for inv
        // since host2host is cheap
        // first_block_d is only used in forward pass
        cudaErrchk(cudaMemcpyAsync(inv_diagblk_d,
                    inv_diagblk_small_d[stream_compute],
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        cudaErrchk(cudaMemcpyAsync(inv_upperblk_d,
                    identity_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        cudaErrchk(cudaMemcpyAsync(inv_lowerblk_d,
                    first_block_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        cudaErrchk(cudaEventRecord(schur_inverted[i], stream[stream_compute]));

        // wait to unload for the finish of computations
        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[i]));

        cudaErrchk(cudaMemcpyAsync(inv_diagblk_h + i*blocksize*blocksize,
                    inv_diagblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        cudaErrchk(cudaMemcpyAsync(inv_upperblk_h + i*blocksize*blocksize,
                    inv_upperblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        cudaErrchk(cudaMemcpyAsync(inv_lowerblk_h + i*blocksize*blocksize,
                    inv_lowerblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        // unloading finished
        cudaErrchk(cudaEventRecord(unload[i], stream[stream_memunload]));
    }
    // synchronize all the streams
    for(int j = 0; j < number_streams; j++){
        cudaErrchk(cudaStreamSynchronize(stream[j]));
    }
    // deallocate device memory
    for(int i = 0; i < number_streams; i++){
        if (stream[i]) {
            cudaErrchk(cudaStreamDestroy(stream[i]));
        }
        if(cublas_handle[i]) {
            cublasErrchk(cublasDestroy(cublas_handle[i]));
        }
        if(cusolverDn_handle[i]) {
            cusolverErrchk(cusolverDnDestroy(cusolverDn_handle[i]));
        }
        if(cusparse_handle[i]){
            cusparseErrchk(cusparseDestroy(cusparse_handle[i]));
        }
    }
    for(unsigned int i = 0; i < n_blocks; i++){
        if(diagblk_descr_forward[i]) {
            cusparseErrchk(cusparseDestroySpMat(diagblk_descr_forward[i]));
        }
    }
    for(unsigned int i = 0; i < n_blocks - 1; i++){
        if(upperblk_descr_forward[i]) {
            cusparseErrchk(cusparseDestroySpMat(upperblk_descr_forward[i]));
        }
        if(lowerblk_descr_forward[i]) {
            cusparseErrchk(cusparseDestroySpMat(lowerblk_descr_forward[i]));
        }
        if(upperblk_descr_backward[i]) {
            cusparseErrchk(cusparseDestroySpMat(upperblk_descr_backward[i]));
        }
        if(lowerblk_descr_backward[i]) {
            cusparseErrchk(cusparseDestroySpMat(lowerblk_descr_backward[i]));
        }
    }
    if(dense_descr_in){
        cusparseErrchk(cusparseDestroyDnMat(dense_descr_in));
    }
    if(dense_descr_out){
        cusparseErrchk(cusparseDestroyDnMat(dense_descr_out));
    }


    if(first_block_d){
        cudaErrchk(cudaFree(first_block_d));
    }
    if(inv_diagblk_d) {
        cudaErrchk(cudaFree(inv_diagblk_d));
    }
    if(inv_upperblk_d) {
        cudaErrchk(cudaFree(inv_upperblk_d));
    }
    if(inv_lowerblk_d) {
        cudaErrchk(cudaFree(inv_lowerblk_d));
    }
    if(identity_d){
        cudaErrchk(cudaFree(identity_d));
    }
    if(identity_cpy_d){
        cudaErrchk(cudaFree(identity_cpy_d));
    }

    for(int i = 0; i < 2; i++){
        if(inv_diagblk_small_d[i]){
            cudaErrchk(cudaFree(inv_diagblk_small_d[i]));
        }        
    }


    if(buffer_spmm){
        cudaErrchk(cudaFree(buffer_spmm));
    }
    if(buffer_getrf){
        cudaErrchk(cudaFree(buffer_getrf));
    }
    if(ipiv_d){
        cudaErrchk(cudaFree(ipiv_d));
    }
    if(info_d){
        cudaErrchk(cudaFree(info_d));
    }
    for(unsigned int i = 0; i < n_blocks; i++){
        if(schur_inverted[i]){
            cudaErrchk(cudaEventDestroy(schur_inverted[i]));
        }        
    }
    for(unsigned int i = 0; i < n_blocks; i++){
        if(unload[i]){
            cudaErrchk(cudaEventDestroy(unload[i]));
        }
    }

    return success;
}

int main() {

    if (cudaSetDevice(0) != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device.");
    }

    if(sizeof(complex_h) != sizeof(complex_d)){
        printf("Error: complex_h and complex_d have different sizes\n");
        return 1;
    }
    else{
        printf("complex_h and complex_d have the same size\n");
    }
 
    // Get matrix parameters
    char f_matparam[] = "../../../tests/tests_cases/sparse_blocks_matrix_0_parameters.txt";
    unsigned int matrix_size;
    unsigned int blocksize;

    load_matrix_parameters(f_matparam, &matrix_size, &blocksize);

    unsigned int n_blocks = matrix_size / blocksize;
    unsigned int off_diag_size = matrix_size - blocksize;

    // Print the matrix parameters
    printf("Matrix parameters:\n");
    printf("    Matrix size: %d\n", matrix_size);
    printf("    Block size: %d\n", blocksize);
    printf("    Number of blocks: %d\n", n_blocks);


    // Load matrix to invert
    std::complex<double>* matrix_diagblk = (std::complex<double>*) malloc(blocksize * matrix_size * sizeof(std::complex<double>));
    char f_mat_diagblk[] = "../../../tests/tests_cases/sparse_blocks_matrix_0_diagblk.bin";
    load_binary_matrix(f_mat_diagblk, matrix_diagblk, blocksize, matrix_size);

    std::complex<double>* matrix_upperblk = (std::complex<double>*) malloc(blocksize * (off_diag_size) * sizeof(std::complex<double>));
    char f_mat_upperblk[] = "../../../tests/tests_cases/sparse_blocks_matrix_0_upperblk.bin";
    load_binary_matrix(f_mat_upperblk, matrix_upperblk, blocksize, off_diag_size);

    std::complex<double>* matrix_lowerblk = (std::complex<double>*) malloc(blocksize * (off_diag_size) * sizeof(std::complex<double>));
    char f_mat_lowerblk[] = "../../../tests/tests_cases/sparse_blocks_matrix_0_lowerblk.bin";
    load_binary_matrix(f_mat_lowerblk, matrix_lowerblk, blocksize, off_diag_size);

    int diag_nnz[n_blocks];
    int upper_nnz[n_blocks-1];
    int lower_nnz[n_blocks-1];
    std::string path_sparse = "../../../tests/tests_cases/sparse_matrices_"+ std::to_string(matrix_size) +"_" + std::to_string(blocksize) ;
    load_text_array<int>( (path_sparse + "/diag_nnz.txt").c_str(), diag_nnz, n_blocks);
    load_text_array<int>( (path_sparse + "/upper_nnz.txt").c_str(), upper_nnz, n_blocks-1);
    load_text_array<int>( (path_sparse + "/lower_nnz.txt").c_str(), lower_nnz, n_blocks-1);

    complex_h* diagblk_data_h[n_blocks];
    complex_h* upperblk_data_h[n_blocks-1];
    complex_h* lowerblk_data_h[n_blocks-1];
    int* diagblk_indices_h[n_blocks];
    int* upperblk_indices_h[n_blocks-1];
    int* lowerblk_indices_h[n_blocks-1];
    int* diagblk_indptr_h[n_blocks]; 
    int* upperblk_indptr_h[n_blocks-1];
    int* lowerblk_indptr_h[n_blocks-1];

    for(unsigned int i = 0; i < n_blocks; i++){
        cudaMallocHost((void**)&diagblk_data_h[i], diag_nnz[i] * sizeof(complex_h));
        cudaMallocHost((void**)&diagblk_indices_h[i], diag_nnz[i]* sizeof(int));
        cudaMallocHost((void**)&diagblk_indptr_h[i], (blocksize+1) * sizeof(int));
        std::string path_data = path_sparse + "/diag_data" + std::to_string(i) + ".bin";
        std::string path_indices = path_sparse + "/diag_indices" + std::to_string(i) + ".bin";
        std::string path_indptr = path_sparse + "/diag_indptr" + std::to_string(i) + ".bin";

        load_binary_array<complex_h>(path_data, diagblk_data_h[i], diag_nnz[i]);
        load_binary_array<int>(path_indices, diagblk_indices_h[i], diag_nnz[i]);
        load_binary_array<int>(path_indptr, diagblk_indptr_h[i], blocksize+1);
    }

    for(unsigned int i = 0; i < n_blocks - 1; i++){
        cudaMallocHost((void**)&upperblk_data_h[i], upper_nnz[i] * sizeof(complex_h));
        cudaMallocHost((void**)&upperblk_indices_h[i], upper_nnz[i]* sizeof(int));
        cudaMallocHost((void**)&upperblk_indptr_h[i], (blocksize+1) * sizeof(int));
        std::string path_data = path_sparse + "/upper_data" + std::to_string(i) + ".bin";
        std::string path_indices = path_sparse + "/upper_indices" + std::to_string(i) + ".bin";
        std::string path_indptr = path_sparse + "/upper_indptr" + std::to_string(i) + ".bin";
        load_binary_array<complex_h>(path_data, upperblk_data_h[i], upper_nnz[i]);
        load_binary_array<int>(path_indices, upperblk_indices_h[i], upper_nnz[i]);
        load_binary_array<int>(path_indptr, upperblk_indptr_h[i], blocksize+1);
    }

    for(unsigned int i = 0; i < n_blocks - 1; i++){
        cudaMallocHost((void**)&lowerblk_data_h[i], lower_nnz[i] * sizeof(complex_h));
        cudaMallocHost((void**)&lowerblk_indices_h[i], lower_nnz[i]* sizeof(int));
        cudaMallocHost((void**)&lowerblk_indptr_h[i], (blocksize+1) * sizeof(int));
        std::string path_data = path_sparse + "/lower_data" + std::to_string(i) + ".bin";
        std::string path_indices = path_sparse + "/lower_indices" + std::to_string(i) + ".bin";
        std::string path_indptr = path_sparse + "/lower_indptr" + std::to_string(i) + ".bin";

        load_binary_array<complex_h>(path_data, lowerblk_data_h[i], lower_nnz[i]);
        load_binary_array<int>(path_indices, lowerblk_indices_h[i], lower_nnz[i]);
        load_binary_array<int>(path_indptr, lowerblk_indptr_h[i], blocksize+1);
    }




    // allocate memory for the inverse
    complex_h* inv_diagblk_h = NULL;
    complex_h* inv_upperblk_h = NULL;
    complex_h* inv_lowerblk_h = NULL;

    cudaMallocHost((void**)&inv_diagblk_h, blocksize * matrix_size * sizeof(complex_h));
    cudaMallocHost((void**)&inv_upperblk_h, blocksize * off_diag_size * sizeof(complex_h));
    cudaMallocHost((void**)&inv_lowerblk_h, blocksize * off_diag_size * sizeof(complex_h));

   
    if(!rgf_sparse_matrix_does_not_fit_gpu_memory_with_copy_compute_overlap(blocksize, matrix_size,
                                    diag_nnz,
                                    upper_nnz,
                                    lower_nnz,
                                    diagblk_data_h,
                                    diagblk_indices_h,
                                    diagblk_indptr_h,
                                    upperblk_data_h,
                                    upperblk_indices_h,
                                    upperblk_indptr_h,
                                    lowerblk_data_h,
                                    lowerblk_indices_h,
                                    lowerblk_indptr_h,
                                    inv_diagblk_h,
                                    inv_upperblk_h,
                                    inv_lowerblk_h)){
        printf("Error: rgf_dense_matrix_fits_gpu_memory_with_copy_compute_overlap failed\n");
    }
    else{
        printf("rgf_dense_matrix_fits_gpu_memory_with_copy_compute_overlap succeeded\n");
    }


    // ----- RESULT CHECKING SECTION -----

    // Load reference solution of the matrix inverse
    std::complex<double>* matrix_inv_diagblk_ref = (std::complex<double>*) malloc(blocksize * matrix_size * sizeof(std::complex<double>));
    char f_mat_inv_diagblk[] = "../../../tests/tests_cases/sparse_blocks_matrix_0_inverse_diagblk.bin";
    load_binary_matrix(f_mat_inv_diagblk, matrix_inv_diagblk_ref, blocksize, matrix_size);

    std::complex<double>* matrix_inv_upperblk_ref = (std::complex<double>*) malloc(blocksize * (off_diag_size) * sizeof(std::complex<double>));
    char f_mat_inv_upperblk[] = "../../../tests/tests_cases/sparse_blocks_matrix_0_inverse_upperblk.bin";
    load_binary_matrix(f_mat_inv_upperblk, matrix_inv_upperblk_ref, blocksize, off_diag_size);
    
    std::complex<double>* matrix_inv_lowerblk_ref = (std::complex<double>*) malloc(blocksize * (off_diag_size) * sizeof(std::complex<double>));
    char f_mat_inv_lowerblk[] = "../../../tests/tests_cases/sparse_blocks_matrix_0_inverse_lowerblk.bin";
    load_binary_matrix(f_mat_inv_lowerblk, matrix_inv_lowerblk_ref, blocksize, off_diag_size);


    // Transform the reference solution to contiguous blocks where the blocks have column-major order
    complex_h* inv_diagblk_ref = (complex_h*) malloc(blocksize * matrix_size * sizeof(complex_h));
    complex_h* inv_upperblk_ref = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));
    complex_h* inv_lowerblk_ref = (complex_h*) malloc(blocksize * (off_diag_size) * sizeof(complex_h));



    for(unsigned int i = 0; i < blocksize * matrix_size; i++){
        // block index
        int k = i / (blocksize * blocksize);
        // index inside block
        int h = i % (blocksize * blocksize);
        // row inside block
        int m = h % blocksize;
        // col inside block
        int n = h / blocksize;
        inv_diagblk_ref[i] = matrix_inv_diagblk_ref[m*matrix_size + k*blocksize + n];
    }
    for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
        // block index
        int k = i / (blocksize * blocksize);
        // index inside block
        int h = i % (blocksize * blocksize);
        // row inside block
        int m = h % blocksize;
        // col inside block
        int n = h / blocksize;
        inv_upperblk_ref[i] = matrix_inv_upperblk_ref[m*off_diag_size + k*blocksize + n];
        inv_lowerblk_ref[i] = matrix_inv_lowerblk_ref[m*off_diag_size + k*blocksize + n];
    }

    // // print last block of inverted matrix
    // for(unsigned int i = blocksize *(matrix_size-blocksize); i < blocksize * matrix_size; i++){
    //     // std::cout << "inv_diagblk_h[" << i << "] = " << inv_diagblk_h[i] << std::endl;
    //     // std::cout << "inv_diagblk_ref[" << i << "] = " << inv_diagblk_ref[i] << std::endl;
    //     std::cout << inv_diagblk_h[i] - inv_diagblk_ref[i] << std::endl;
    // }

    // // print sceond to last block of inverted matrix
    // for(unsigned int i = blocksize *(matrix_size-2*blocksize); i < blocksize * (matrix_size-blocksize); i++){
    //     std::cout << "inv_diagblk_h[" << i << "] = " << inv_diagblk_h[i] << std::endl;
    //     std::cout << "inv_diagblk_ref[" << i << "] = " << inv_diagblk_ref[i] << std::endl;
    //     std::cout << inv_diagblk_h[i] - inv_diagblk_ref[i] << std::endl;
    // }

    double norm_diagblk = 0.0;
    double norm_upperblk = 0.0;
    double norm_lowerblk = 0.0;
    double diff_diagblk = 0.0;
    double diff_upperblk = 0.0;
    double diff_lowerblk = 0.0;
    for(unsigned int i = 0; i < blocksize * matrix_size; i++){
        norm_diagblk += std::abs(inv_diagblk_ref[i]);
        diff_diagblk += std::abs(inv_diagblk_h[i] - inv_diagblk_ref[i]);
    }
    for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
        norm_upperblk += std::abs(inv_upperblk_ref[i]);
        norm_lowerblk += std::abs(inv_lowerblk_ref[i]);
        diff_upperblk += std::abs(inv_upperblk_h[i] - inv_upperblk_ref[i]);
        diff_lowerblk += std::abs(inv_lowerblk_h[i] - inv_lowerblk_ref[i]);
    }
    double eps = 1e-12;
    if(diff_diagblk/norm_diagblk > eps){
        std::cout << diff_diagblk/norm_diagblk << std::endl;
        printf("Error: inv_diagblk_h and inv_diagblk_ref are not equal\n");
    }
    else{
        printf("inv_diagblk_h and inv_diagblk_ref are equal\n");
    }
    if(diff_upperblk/norm_upperblk > eps){
        std::cout << diff_upperblk/norm_upperblk << std::endl;
        printf("Error: inv_upperblk_h and inv_upperblk_ref are not equal\n");
    }
    else{
        printf("inv_upperblk_h and inv_upperblk_ref are equal\n");
    }
    if(diff_lowerblk/norm_lowerblk > eps){
        std::cout << diff_lowerblk/norm_lowerblk << std::endl;
        printf("Error: inv_lowerblk_h and inv_lowerblk_ref are not equal\n");
    }
    else{
        printf("inv_lowerblk_h and inv_lowerblk_ref are equal\n");
    }


    if(inv_diagblk_h){
        cudaFreeHost(inv_diagblk_h);
    }
    if(inv_upperblk_h){
        cudaFreeHost(inv_upperblk_h);
    }
    if(inv_lowerblk_h){
        cudaFreeHost(inv_lowerblk_h);
    }
    for(unsigned int i = 0; i < n_blocks; i++){
        cudaFreeHost(diagblk_data_h[i]);
    }
    for(unsigned int i = 0; i < n_blocks-1; i++){
        cudaFreeHost(upperblk_data_h[i]);
    }
    for(unsigned int i = 0; i < n_blocks-1; i++){
        cudaFreeHost(lowerblk_data_h[i]);
    }

    if(matrix_diagblk){
        free(matrix_diagblk);
    }
    if(matrix_upperblk){
        free(matrix_upperblk);
    }
    if(matrix_lowerblk){
        free(matrix_lowerblk);
    }
    if(matrix_inv_diagblk_ref){
        free(matrix_inv_diagblk_ref);
    }
    if(matrix_inv_upperblk_ref){
        free(matrix_inv_upperblk_ref);
    }
    if(matrix_inv_lowerblk_ref){
        free(matrix_inv_lowerblk_ref);
    }
    if(inv_diagblk_ref){
        free(inv_diagblk_ref);
    }
    if(inv_upperblk_ref){
        free(inv_upperblk_ref);
    }
    if(inv_lowerblk_ref){
        free(inv_lowerblk_ref);
    }



    return 0;
}








