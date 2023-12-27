// Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
#include "single_lesser_greater.h"

void rgf_lesser_greater(
    unsigned int blocksize,
    unsigned int matrix_size,
    complex_h *system_matrix_diagblk_h,
    complex_h *system_matrix_upperblk_h,
    complex_h *system_matrix_lowerblk_h,
    complex_h *self_energy_lesser_diagblk_h,
    complex_h *self_energy_lesser_upperblk_h,
    complex_h *self_energy_greater_diagblk_h,
    complex_h *self_energy_greater_upperblk_h,
    complex_h *lesser_inv_diagblk_h,
    complex_h *lesser_inv_upperblk_h,
    complex_h *greater_inv_diagblk_h,
    complex_h *greater_inv_upperblk_h)
{
    
    if(matrix_size % blocksize != 0){
        printf("Error: matrix_size is not a multiple of blocksize\n");
    }
    unsigned int n_blocks = matrix_size / blocksize;

    // Init cuda stuff

    // need multiple streams for overlap
    int number_streams = 3;
    cudaStream_t stream[number_streams];
    for(int i = 0; i < number_streams; i++){
        cudaErrchk(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));
    }

    cusolverDnHandle_t cusolver_handle[number_streams];
    for(int i = 0; i < number_streams; i++){
        cusolverErrchk(cusolverDnCreate(&cusolver_handle[i]));
        cusolverErrchk(cusolverDnSetStream(cusolver_handle[i], stream[i]));
    }

    cublasHandle_t cublas_handle[number_streams];
    for(int i = 0; i < number_streams; i++){
        cublasErrchk(cublasCreate(&cublas_handle[i]));
        cublasErrchk(cublasSetStream(cublas_handle[i], stream[i]));
    }

    cudaEvent_t schur_inverted[n_blocks];
    for(unsigned int i = 0; i < n_blocks; i++){
        cudaErrchk(cudaEventCreate(&schur_inverted[i]))
    }

    cudaEvent_t lesser_greater_calculated[n_blocks];
    cudaEvent_t lesser_greater_calculated_upper[n_blocks];
    for(unsigned int i = 0; i < n_blocks; i++){
        cudaErrchk(cudaEventCreate(&lesser_greater_calculated[i]))
        cudaErrchk(cudaEventCreate(&lesser_greater_calculated_upper[i]))
    }

    cudaEvent_t unload_diag[n_blocks];
    cudaEvent_t unload_upper[n_blocks];
    for(unsigned int i = 0; i < n_blocks; i++){
        cudaErrchk(cudaEventCreate(&unload_diag[i]))
        cudaErrchk(cudaEventCreate(&unload_upper[i]))
    }
    complex_d alpha;
    complex_d beta;
    int stream_memload = 1;
    int stream_compute = 0;
    int stream_memunload = 2;

    // not allowed to load full matrix to device
    // allocate memory for the blocks
    
    complex_d* system_matrix_diagblk_d[2];
    complex_d* system_matrix_upperblk_d[2];
    complex_d* system_matrix_lowerblk_d[2];
    complex_d* self_energy_lesser_diagblk_d[2];
    complex_d* self_energy_lesser_upperblk_d[2];
    complex_d* self_energy_greater_diagblk_d[2];
    complex_d* self_energy_greater_upperblk_d[2];
    

    // allocate single blocks of the matrix
    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&system_matrix_diagblk_d[i], blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&system_matrix_upperblk_d[i], blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&system_matrix_lowerblk_d[i], blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&self_energy_lesser_diagblk_d[i], blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&self_energy_lesser_upperblk_d[i], blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&self_energy_greater_diagblk_d[i], blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&self_energy_greater_upperblk_d[i], blocksize * blocksize * sizeof(complex_d)));
    }


    // allocate memory for the inverse
    complex_d* inv_diagblk_d = NULL;
    complex_d* inv_diagblk_small_d[2];

    complex_d* lesser_inv_diagblk_d = NULL;
    complex_d* lesser_inv_upperblk_d = NULL;
    complex_d* greater_inv_diagblk_d = NULL;
    complex_d* greater_inv_upperblk_d = NULL;

    complex_d* lesser_inv_diagblk_small_d[2];
    complex_d* greater_inv_diagblk_small_d[2];



    cudaErrchk(cudaMalloc((void**)&inv_diagblk_d, blocksize * blocksize * sizeof(complex_d)));
    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&inv_diagblk_small_d[i], blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&lesser_inv_diagblk_small_d[i], blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&greater_inv_diagblk_small_d[i], blocksize * blocksize * sizeof(complex_d)));
    }
    
    cudaErrchk(cudaMalloc((void**)&lesser_inv_diagblk_d, blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&lesser_inv_upperblk_d, blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&greater_inv_diagblk_d, blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&greater_inv_upperblk_d, blocksize * blocksize * sizeof(complex_d)));

    //memory for pivoting
    int *ipiv_d = NULL;
    int *info_d = NULL;
    cudaErrchk(cudaMalloc((void**)&info_d, sizeof(int)))
    cudaErrchk(cudaMalloc((void**)&ipiv_d, blocksize*sizeof(int)));

    // memory for small g
    complex_h* small_inv_diagblk_h;
    cudaErrchk(cudaMallocHost((void**)&small_inv_diagblk_h, n_blocks * blocksize * blocksize * sizeof(complex_h)));

    // create right hand side identity matrix
    complex_h* identity_h;
    cudaErrchk(cudaMallocHost((void**)&identity_h, blocksize * blocksize * sizeof(complex_h)));
    complex_d* identity_d = NULL;
    cudaErrchk(cudaMalloc((void**)&identity_d, blocksize * blocksize * sizeof(complex_d)));


    for(unsigned int i = 0; i < blocksize * blocksize; i++){
        identity_h[i] = 0.0;
        if(i / blocksize == i % blocksize){
            identity_h[i] = 1.0;
        }
    }
    // init right hand side identity matrix on device for backsub
    cudaErrchk(cudaMemcpyAsync(identity_d, reinterpret_cast<const complex_d*>(identity_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_compute]));
    cudaErrchk(cudaFreeHost(identity_h));

    //figure out extra amount of memory needed
    complex_d *buffer = NULL;
    int bufferSize = 0;
    cusolverErrchk(cusolverDnZgetrf_bufferSize(cusolver_handle[stream_compute], blocksize, blocksize,
                                            (complex_d *)system_matrix_diagblk_d[stream_compute],
                                              blocksize, &bufferSize));
    cudaErrchk(cudaMalloc((void**)&buffer, sizeof(complex_d) * bufferSize));

    // ----- END OF INIT SECTION -----



    cudaErrchk(cudaMemcpyAsync(inv_diagblk_d, identity_d,
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
    

    cudaErrchk(cudaMemcpyAsync(system_matrix_diagblk_d[stream_compute], reinterpret_cast<const complex_d*>(system_matrix_diagblk_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_compute]));


    cudaErrchk(cudaMemcpyAsync(self_energy_lesser_diagblk_d[stream_compute], reinterpret_cast<const complex_d*>(self_energy_lesser_diagblk_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_compute]));

    cudaErrchk(cudaMemcpyAsync(self_energy_greater_diagblk_d[stream_compute], reinterpret_cast<const complex_d*>(self_energy_greater_diagblk_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_compute]));


    cusolverErrchk(cusolverDnZgetrf(cusolver_handle[stream_compute], blocksize, blocksize,
                                system_matrix_diagblk_d[stream_compute], blocksize, buffer, ipiv_d, info_d));
    

    //back substitution
    cusolverErrchk(cusolverDnZgetrs(cusolver_handle[stream_compute], CUBLAS_OP_N, blocksize,
                                    blocksize, system_matrix_diagblk_d[stream_compute], blocksize, ipiv_d,
                                    inv_diagblk_d, blocksize, info_d));

    // record finishing the inverse of the first block
    cudaErrchk(cudaEventRecord(schur_inverted[0], stream[stream_compute]));


    alpha = make_cuDoubleComplex(1.0, 0.0);
    beta = make_cuDoubleComplex(0.0, 0.0);

    // use self_energy_lesser_upperblk_d[stream_compute]
    // as temporary buffer
    cublasErrchk(cublasZgemm(
        cublas_handle[stream_compute],
        CUBLAS_OP_N, CUBLAS_OP_C,
        blocksize, blocksize, blocksize,
        &alpha,
        self_energy_lesser_diagblk_d[stream_compute], blocksize,
        inv_diagblk_d, blocksize,
        &beta,
        self_energy_lesser_upperblk_d[stream_compute], blocksize));
    cublasErrchk(cublasZgemm(
        cublas_handle[stream_compute],
        CUBLAS_OP_N, CUBLAS_OP_N,
        blocksize, blocksize, blocksize,
        &alpha,
        inv_diagblk_d, blocksize,
        self_energy_lesser_upperblk_d[stream_compute], blocksize,
        &beta,
        lesser_inv_diagblk_d, blocksize));
    cublasErrchk(cublasZgemm(
        cublas_handle[stream_compute],
        CUBLAS_OP_N, CUBLAS_OP_C,
        blocksize, blocksize, blocksize,
        &alpha,
        self_energy_greater_diagblk_d[stream_compute], blocksize,
        inv_diagblk_d, blocksize,
        &beta,
        self_energy_greater_upperblk_d[stream_compute], blocksize));
    cublasErrchk(cublasZgemm(
        cublas_handle[stream_compute],
        CUBLAS_OP_N, CUBLAS_OP_N,
        blocksize, blocksize, blocksize,
        &alpha,
        inv_diagblk_d, blocksize,
        self_energy_greater_upperblk_d[stream_compute], blocksize,
        &beta,
        greater_inv_diagblk_d, blocksize));
    cudaErrchk(cudaEventRecord(lesser_greater_calculated[0], stream[stream_compute]));

    //wait for the inverse of the first block
    cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[0]));
    // 0. Inverse of the first block
    cudaErrchk(cudaMemcpyAsync(small_inv_diagblk_h, inv_diagblk_d,
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
    cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], lesser_greater_calculated[0]));
    cudaErrchk(cudaMemcpyAsync(lesser_inv_diagblk_h, lesser_inv_diagblk_d,
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
    cudaErrchk(cudaMemcpyAsync(greater_inv_diagblk_h, greater_inv_diagblk_d,
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));

    // unloading finished
    cudaErrchk(cudaEventRecord(unload_diag[0], stream[stream_memunload]));


    // first memcpy happens before loop
    cudaErrchk(cudaMemcpyAsync(system_matrix_diagblk_d[stream_memload],
                reinterpret_cast<const complex_d*>(system_matrix_diagblk_h + blocksize*blocksize),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(system_matrix_upperblk_d[stream_memload],
                reinterpret_cast<const complex_d*>(system_matrix_upperblk_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(system_matrix_lowerblk_d[stream_memload],
                reinterpret_cast<const complex_d*>(system_matrix_lowerblk_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(self_energy_lesser_diagblk_d[stream_memload],
                reinterpret_cast<const complex_d*>(self_energy_lesser_diagblk_h + blocksize*blocksize),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(self_energy_lesser_upperblk_d[stream_memload],
                reinterpret_cast<const complex_d*>(self_energy_lesser_upperblk_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(self_energy_greater_diagblk_d[stream_memload],
                reinterpret_cast<const complex_d*>(self_energy_greater_diagblk_h + blocksize*blocksize),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(self_energy_greater_upperblk_d[stream_memload],
                reinterpret_cast<const complex_d*>(self_energy_greater_upperblk_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));



    // // 1. Forward substitution (performed left to right)
    for (unsigned int i = 1; i < n_blocks; ++i) {


        int stream_memload = (i+1) % 2;
        int stream_compute = i % 2;
        int stream_memunload = 2;


        if(i < n_blocks-1){
            // load the blocks for the next iteration
            cudaErrchk(cudaMemcpyAsync(system_matrix_diagblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(system_matrix_diagblk_h + (i+1)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(system_matrix_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(system_matrix_upperblk_h + (i)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(system_matrix_lowerblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(system_matrix_lowerblk_h  + (i)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(self_energy_lesser_diagblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(self_energy_lesser_diagblk_h + (i+1)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(self_energy_lesser_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(self_energy_lesser_upperblk_h + (i)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(self_energy_greater_diagblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(self_energy_greater_diagblk_h + (i+1)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(self_energy_greater_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(self_energy_greater_upperblk_h + (i)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
        }




        //wait for the schur inverse from the previous iteration
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], schur_inverted[i-1]));
        // without this the solution is not correct
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], lesser_greater_calculated[i-1]));

        // MatMul tmp = eig_lowerblk[i-1] * eig_inv_diagblk[i-1]
        // use the inv_diagblk_d from last iteration
        // use inv_diagblk_small_d[stream_compute] as tmp
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            system_matrix_lowerblk_d[stream_compute], blocksize,
            inv_diagblk_d, blocksize,
            &beta,
            inv_diagblk_small_d[stream_compute], blocksize));
        
        //MatMul schur complement = eig_diagblk[i] - tmp * eig_upperblk[i-1]
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);

        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            inv_diagblk_small_d[stream_compute], blocksize,
            system_matrix_upperblk_d[stream_compute], blocksize,
            &beta,
            system_matrix_diagblk_d[stream_compute], blocksize));

        // first temporary products for lesser and greater which use g_retarded[i_minus_one_, i_minus_one_]
        //calculate lesser and greater inverse
        //System_matrix[i_, i_minus_one_] @ g_lesser_greater[i_minus_one_, i_minus_one_]
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        // use  lesser_inv_upperblk_d as temporary buffer
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            system_matrix_lowerblk_d[stream_compute], blocksize,
            lesser_inv_diagblk_d, blocksize,
            &beta,
            lesser_inv_upperblk_d, blocksize));
    
        // System_matrix[i_, i_minus_one_] @
        //                 g_lesser_greater[i_minus_one_, i_minus_one_]
        //             + Sigma_lesser_greater[i_minus_one_, i_].conj().T @
        //                 g_retarded[i_minus_one_, i_minus_one_].conj().T

        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_C, CUBLAS_OP_C,
            blocksize, blocksize, blocksize,
            &alpha,
            self_energy_lesser_upperblk_d[stream_compute], blocksize,
            inv_diagblk_d, blocksize,
            &beta,
            lesser_inv_upperblk_d, blocksize));

        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        // use  greater_inv_upperblk_d as temporary buffer
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            system_matrix_lowerblk_d[stream_compute], blocksize,
            greater_inv_diagblk_d, blocksize,
            &beta,
            greater_inv_upperblk_d, blocksize));
    
        // System_matrix[i_, i_minus_one_] @
        //                 g_lesser_greater[i_minus_one_, i_minus_one_]
        //             + Sigma_lesser_greater[i_minus_one_, i_].conj().T @
        //                 g_retarded[i_minus_one_, i_minus_one_].conj().T

        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_C, CUBLAS_OP_C,
            blocksize, blocksize, blocksize,
            &alpha,
            self_energy_greater_upperblk_d[stream_compute], blocksize,
            inv_diagblk_d, blocksize,
            &beta,
            greater_inv_upperblk_d, blocksize));




        // wait to not overwrite block to unload_diag
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], unload_diag[i-1]));
        //copy identity
        cudaErrchk(cudaMemcpyAsync(inv_diagblk_d, identity_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        // inverse schur complement
        cusolverErrchk(cusolverDnZgetrf(cusolver_handle[stream_compute], blocksize, blocksize,
                                    system_matrix_diagblk_d[stream_compute],
                                    blocksize, buffer, ipiv_d, info_d));
        

        //back substitution
        cusolverErrchk(cusolverDnZgetrs(cusolver_handle[stream_compute], CUBLAS_OP_N, blocksize,
                                        blocksize,
                                        system_matrix_diagblk_d[stream_compute],
                                        blocksize, ipiv_d,
                                        inv_diagblk_d, blocksize, info_d));
        // record finishing of computation in step i
        cudaErrchk(cudaEventRecord(schur_inverted[i], stream[stream_compute]));

        //calculate lesser and greater inverse
        // Sigma_lesser_greater[i_, i_]
        // - tmp @
        //     Sigma_lesser_greater[i_minus_one_, i_] 
        // tmp = inv_diagblk_small_d[stream_compute]
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            inv_diagblk_small_d[stream_compute], blocksize,
            self_energy_lesser_upperblk_d[stream_compute], blocksize,
            &beta,
            self_energy_lesser_diagblk_d[stream_compute], blocksize));
        
        // Sigma_lesser_greater[i_, i_]
        // - tmp @
        //     Sigma_lesser_greater[i_minus_one_, i_]                
        // + (System_matrix[i_, i_minus_one_] @
        //     g_lesser_greater[i_minus_one_, i_minus_one_]
        // - Sigma_lesser_greater[i_, i_minus_one_] @
        //     g_retarded[i_minus_one_, i_minus_one_].conj().T )@
        //     System_matrix[i_, i_minus_one_].conj().T
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize, blocksize,
            &alpha,
            lesser_inv_upperblk_d, blocksize,
            system_matrix_lowerblk_d[stream_compute], blocksize,
            &beta,
            self_energy_lesser_diagblk_d[stream_compute], blocksize));

        // Sigma_lesser_greater[i_, i_]
        // - tmp @
        //     Sigma_lesser_greater[i_minus_one_, i_] 
        // tmp = inv_diagblk_small_d[stream_compute]
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            inv_diagblk_small_d[stream_compute], blocksize,
            self_energy_greater_upperblk_d[stream_compute], blocksize,
            &beta,
            self_energy_greater_diagblk_d[stream_compute], blocksize));
        
        // Sigma_lesser_greater[i_, i_]
        // - tmp @
        //     Sigma_lesser_greater[i_minus_one_, i_]                
        // + (System_matrix[i_, i_minus_one_] @
        //     g_lesser_greater[i_minus_one_, i_minus_one_]
        // - Sigma_lesser_greater[i_, i_minus_one_] @
        //     g_retarded[i_minus_one_, i_minus_one_].conj().T )@
        //     System_matrix[i_, i_minus_one_].conj().T
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize, blocksize,
            &alpha,
            greater_inv_upperblk_d, blocksize,
            system_matrix_lowerblk_d[stream_compute], blocksize,
            &beta,
            self_energy_greater_diagblk_d[stream_compute], blocksize));

        // g_lesser_greater[i_, i_] = (
        //     g_retarded[i_, i_]
        //     @ (
        //  self_energy_lesser_diagblk_d[stream_compute]/self_energy_greater_diagblk_d
        //     )
        //     @ g_retarded[i_, i_].conj().T
        // use self_energy_lesser_upperblk_d[stream_compute] as temporary buffer
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize, blocksize,
            &alpha,
            self_energy_lesser_diagblk_d[stream_compute], blocksize,
            inv_diagblk_d, blocksize,
            &beta,
            self_energy_lesser_upperblk_d[stream_compute], blocksize));
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            inv_diagblk_d, blocksize,
            self_energy_lesser_upperblk_d[stream_compute], blocksize,
            &beta,
            lesser_inv_diagblk_d, blocksize));
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize, blocksize,
            &alpha,
            self_energy_greater_diagblk_d[stream_compute], blocksize,
            inv_diagblk_d, blocksize,
            &beta,
            self_energy_greater_upperblk_d[stream_compute], blocksize));
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            inv_diagblk_d, blocksize,
            self_energy_greater_upperblk_d[stream_compute], blocksize,
            &beta,
            greater_inv_diagblk_d, blocksize));

        cudaErrchk(cudaEventRecord(lesser_greater_calculated[i], stream[stream_compute]));

        // wait to unload_diag for the finish of computations
        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[i]));
        cudaErrchk(cudaMemcpyAsync(small_inv_diagblk_h + i*blocksize*blocksize,
                    inv_diagblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));

        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], lesser_greater_calculated[i]));
        cudaErrchk(cudaMemcpyAsync(lesser_inv_diagblk_h + i*blocksize*blocksize,
                    lesser_inv_diagblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        cudaErrchk(cudaMemcpyAsync(greater_inv_diagblk_h + i*blocksize*blocksize,
                    greater_inv_diagblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        // unloading finished
        cudaErrchk(cudaEventRecord(unload_diag[i], stream[stream_memunload]));

    }



    int stream_memload_before = (n_blocks) % 2;
    int stream_compute_before = (n_blocks-1) % 2;
   

    cudaErrchk(cudaMemcpyAsync(system_matrix_upperblk_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(system_matrix_upperblk_h + (n_blocks-2)*blocksize*blocksize),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    cudaErrchk(cudaMemcpyAsync(system_matrix_lowerblk_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(system_matrix_lowerblk_h + (n_blocks-2)*blocksize*blocksize),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));

    cudaErrchk(cudaMemcpyAsync(self_energy_lesser_upperblk_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(self_energy_lesser_upperblk_h + (n_blocks-2)*blocksize*blocksize),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    cudaErrchk(cudaMemcpyAsync(self_energy_greater_upperblk_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(self_energy_greater_upperblk_h + (n_blocks-2)*blocksize*blocksize),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
  
    cudaErrchk(cudaStreamWaitEvent(stream[stream_memload_before], unload_diag[n_blocks-2])); 

    cudaErrchk(cudaMemcpyAsync(inv_diagblk_small_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(small_inv_diagblk_h  + (n_blocks-2)*blocksize*blocksize),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    cudaErrchk(cudaMemcpyAsync(lesser_inv_diagblk_small_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(lesser_inv_diagblk_h + (n_blocks-2)*blocksize*blocksize),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    cudaErrchk(cudaMemcpyAsync(greater_inv_diagblk_small_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(greater_inv_diagblk_h + (n_blocks-2)*blocksize*blocksize),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));



    // 2. Backward substitution (performed right to left)
    for(int i = n_blocks-2; i >= 0; --i){

        // fix stream compute to be the stream which loaded
        // blocks before the loop
        stream_memload = (stream_compute_before + (i - n_blocks + 2) ) % 2;
        stream_compute = (stream_memload_before + (i - n_blocks + 2) ) % 2;
        stream_memunload = 2;

        if(i > 0){
            cudaErrchk(cudaMemcpyAsync(system_matrix_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(system_matrix_upperblk_h + (i-1)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(system_matrix_lowerblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(system_matrix_lowerblk_h + (i-1)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(inv_diagblk_small_d[stream_memload],
                        reinterpret_cast<const complex_d*>(small_inv_diagblk_h  + (i-1)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));

            cudaErrchk(cudaMemcpyAsync(lesser_inv_diagblk_small_d[stream_memload],
                        reinterpret_cast<const complex_d*>(lesser_inv_diagblk_h + (i-1)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(greater_inv_diagblk_small_d[stream_memload],
                        reinterpret_cast<const complex_d*>(greater_inv_diagblk_h + (i-1)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
                    
            cudaErrchk(cudaMemcpyAsync(self_energy_lesser_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(self_energy_lesser_upperblk_h + (i-1)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(self_energy_greater_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(self_energy_greater_upperblk_h + (i-1)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
        }
    

        // wait for the block of the last iteration
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], lesser_greater_calculated[i+1]));
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], lesser_greater_calculated_upper[i+1]));

        // buf4 = -(G_tmp @
        //     Sigma_lesser_greater[i_, i_plus_one_].conj().T @
        //     g_retarded[i_, i_].conj().T)
        // self_energy_lesser_diagblk_d[stream_compute] is only used in the forward pass
        // buf4_lesser = self_energy_lesser_upperblk_d[stream_compute]
        // buf4_greater = self_energy_greater_upperblk_d[stream_compute]
        complex_d *buf4_lesser = self_energy_lesser_upperblk_d[stream_compute];
        complex_d *buf4_greater = self_energy_greater_upperblk_d[stream_compute];
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize, blocksize,
            &alpha,
            inv_diagblk_d, blocksize,
            self_energy_lesser_upperblk_d[stream_compute], blocksize,
            &beta,
            self_energy_lesser_diagblk_d[stream_compute], blocksize));
        // self_energy_lesser_upperblk_d[stream_compute] will be overwritten
        // okay since it not used anymore
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize, blocksize,
            &alpha,
            self_energy_lesser_diagblk_d[stream_compute], blocksize,
            inv_diagblk_small_d[stream_compute], blocksize,
            &beta,
            buf4_lesser, blocksize));

        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize, blocksize,
            &alpha,
            inv_diagblk_d, blocksize,
            self_energy_greater_upperblk_d[stream_compute], blocksize,
            &beta,
            self_energy_lesser_diagblk_d[stream_compute], blocksize));
        // self_energy_lesser_upperblk_d[stream_compute] will be overwritten
        // okay since it not used anymore
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize, blocksize,
            &alpha,
            self_energy_lesser_diagblk_d[stream_compute], blocksize,
            inv_diagblk_small_d[stream_compute], blocksize,
            &beta,
            buf4_greater, blocksize));



        // buf2 = (G_tmp @    
        //     System_matrix[i_plus_one_, i_]) 
        // buf2 = self_energy_lesser_diagblk_d[stream_compute]
        alpha = make_cuDoubleComplex(1.0, 0.0);
        complex_d *buf2 = self_energy_lesser_diagblk_d[stream_compute];
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            inv_diagblk_d, blocksize,
            system_matrix_lowerblk_d[stream_compute], blocksize,
            &beta,
            buf2, blocksize));

        // buf1 = (g_retarded[i_, i_] @
        //     System_matrix[i_, i_plus_one_])
        // self_energy_greater_diagblk_d[stream_compute] is not overwritten by memcpy
        // buf1 = self_energy_greater_diagblk_d[stream_compute]
        // since it is only used in the forward pass
        complex_d *buf1 = self_energy_greater_diagblk_d[stream_compute];
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            inv_diagblk_small_d[stream_compute], blocksize,
            system_matrix_upperblk_d[stream_compute], blocksize,
            &beta,
            buf1, blocksize));

        // buf7 = buf2 @
        //     g_retarded[i_, i_]      
        // self_energy_greater_diagblk_d[stream_memload] only used in the forward pass
        complex_d *buf7 = self_energy_greater_diagblk_d[stream_memload];
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            buf2, blocksize,
            inv_diagblk_small_d[stream_compute], blocksize,
            &beta,
            buf7, blocksize));

        // G_tmp   =  g_retarded[i_, i_] + (buf1 @
        //                                     buf7)
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            buf1, blocksize,
            buf7, blocksize,
            &beta,
            inv_diagblk_small_d[stream_compute], blocksize));
        // inv_diagblk_small_d[stream_compute] saves now G_tmp
        cudaErrchk(cudaMemcpyAsync(inv_diagblk_d,
                    inv_diagblk_small_d[stream_compute],
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));

        // buf5 = (buf2 @ g_lesser_greater[i_, i_])
        // buf5 = system_matrix_diagblk_d
        // buf5_lesser = system_matrix_diagblk_d[stream_compute]
        // buf5_greater = system_matrix_diagblk_d[stream_memload]
        complex_d *buf5_lesser = system_matrix_diagblk_d[stream_compute];
        complex_d *buf5_greater = system_matrix_diagblk_d[stream_memload];
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            buf2, blocksize,
            lesser_inv_diagblk_small_d[stream_compute], blocksize,
            &beta,
            buf5_lesser, blocksize));
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            buf2, blocksize,
            greater_inv_diagblk_small_d[stream_compute], blocksize,
            &beta,
            buf5_greater, blocksize));


        // buf3 is  for both lesser and greater and now memory for both seperately is needed
        // buf3_lesser = system_matrix_upperblk_d[stream_compute],
        // buf3_greater = system_matrix_lowerblk_d[stream_compute],
        // buf3 = (
        //     G_lesser_greater[i_plus_one_, i_plus_one_] @
        //     buf1.conj().T
        // )
        complex_d *buf3_lesser = system_matrix_upperblk_d[stream_compute];
        complex_d *buf3_greater = system_matrix_lowerblk_d[stream_compute];
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize, blocksize,
            &alpha,
            lesser_inv_diagblk_d, blocksize,
            buf1, blocksize,
            &beta,
            buf3_lesser, blocksize));
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize, blocksize,
            &alpha,
            greater_inv_diagblk_d, blocksize,
            buf1, blocksize,
            &beta,
            buf3_greater, blocksize));




        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], unload_upper[i+1]));
        // G_lesser_greater[i_plus_one_, i_] =(
        //     buf4
        //     - buf5
        //     - buf3
        // )
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgeam(
            cublas_handle[stream_compute],
            CUBLAS_OP_C, CUBLAS_OP_C,
            blocksize, blocksize,
            &alpha,
            buf4_lesser, blocksize,
            &beta,
            buf5_lesser, blocksize,
            lesser_inv_upperblk_d, blocksize
        ));
        cublasErrchk(cublasZgeam(
            cublas_handle[stream_compute],
            CUBLAS_OP_C, CUBLAS_OP_C,
            blocksize, blocksize,
            &alpha,
            buf4_greater, blocksize,
            &beta,
            buf5_greater, blocksize,
            greater_inv_upperblk_d, blocksize
        ));
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        // inplace
        cublasErrchk(cublasZgeam(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize,
            &alpha,
            lesser_inv_upperblk_d, blocksize,
            &beta,
            buf3_lesser, blocksize,
            lesser_inv_upperblk_d, blocksize
        ));
        cublasErrchk(cublasZgeam(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize,
            &alpha,
            greater_inv_upperblk_d, blocksize,
            &beta,
            buf3_greater, blocksize,
            greater_inv_upperblk_d, blocksize
        ));


        cudaErrchk(cudaEventRecord(lesser_greater_calculated_upper[i], stream[stream_compute]));

        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], lesser_greater_calculated_upper[i]));

        cudaErrchk(cudaMemcpyAsync(lesser_inv_upperblk_h + i*blocksize*blocksize,
                    lesser_inv_upperblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        cudaErrchk(cudaMemcpyAsync(greater_inv_upperblk_h + i*blocksize*blocksize,
                    greater_inv_upperblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));

        cudaErrchk(cudaEventRecord(unload_upper[i], stream[stream_memunload]));

        // buf6 = (buf1 @ buf4)
        // buf6_lesser = self_energy_lesser_diagblk_d[stream_compute]
        // buf6_greater = self_energy_lesser_diagblk_d[stream_memload]
        complex_d *buf6_lesser = self_energy_lesser_diagblk_d[stream_compute];
        complex_d *buf6_greater = self_energy_lesser_diagblk_d[stream_memload];
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            buf1, blocksize,
            buf4_lesser, blocksize,
            &beta,
            buf6_lesser, blocksize));
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            buf1, blocksize,
            buf4_greater, blocksize,
            &beta,
            buf6_greater, blocksize));

        // buf8 = (buf1 @ buf5)
        // buf8_lesser = self_energy_lesser_upperblk_d[stream_compute]
        // buf8_greater = self_energy_greater_upperblk_d[stream_compute]
        complex_d *buf8_lesser = self_energy_lesser_upperblk_d[stream_compute];
        complex_d *buf8_greater = self_energy_greater_upperblk_d[stream_compute];
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            buf1, blocksize,
            buf5_lesser, blocksize,
            &beta,
            buf8_lesser, blocksize));
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            buf1, blocksize,
            buf5_greater, blocksize,
            &beta,
            buf8_greater, blocksize));


        // g_lesser_greater[i_, i_]
        // + buf1 @ buf3
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            buf1, blocksize,
            buf3_lesser, blocksize,
            &beta,
            lesser_inv_diagblk_small_d[stream_compute], blocksize));
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            buf1, blocksize,
            buf3_greater, blocksize,
            &beta,
            greater_inv_diagblk_small_d[stream_compute], blocksize));


        // - buf6
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(-1.0, 0.0);
        cublasErrchk(cublasZgeam(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize,
            &alpha,
            lesser_inv_diagblk_small_d[stream_compute], blocksize,
            &beta,
            buf6_lesser, blocksize,
            lesser_inv_diagblk_small_d[stream_compute], blocksize
        ));
        cublasErrchk(cublasZgeam(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize,
            &alpha,
            greater_inv_diagblk_small_d[stream_compute], blocksize,
            &beta,
            buf6_greater, blocksize,
            greater_inv_diagblk_small_d[stream_compute], blocksize
        ));
        // + buf6.conj().T
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgeam(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize,
            &alpha,
            lesser_inv_diagblk_small_d[stream_compute], blocksize,
            &beta,
            buf6_lesser, blocksize,
            lesser_inv_diagblk_small_d[stream_compute], blocksize
        ));
        cublasErrchk(cublasZgeam(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize,
            &alpha,
            greater_inv_diagblk_small_d[stream_compute], blocksize,
            &beta,
            buf6_greater, blocksize,
            greater_inv_diagblk_small_d[stream_compute], blocksize
        ));
        // + buf4
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgeam(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize,
            &alpha,
            lesser_inv_diagblk_small_d[stream_compute], blocksize,
            &beta,
            buf8_lesser, blocksize,
            lesser_inv_diagblk_small_d[stream_compute], blocksize
        ));        
        cublasErrchk(cublasZgeam(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize,
            &alpha,
            greater_inv_diagblk_small_d[stream_compute], blocksize,
            &beta,
            buf8_greater, blocksize,
            greater_inv_diagblk_small_d[stream_compute], blocksize
        ));

        // - buf4.conj().T
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(-1.0, 0.0);
        cublasErrchk(cublasZgeam(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize,
            &alpha,
            lesser_inv_diagblk_small_d[stream_compute], blocksize,
            &beta,
            buf8_lesser, blocksize,
            lesser_inv_diagblk_small_d[stream_compute], blocksize
        ));
        cublasErrchk(cublasZgeam(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize,
            &alpha,
            greater_inv_diagblk_small_d[stream_compute], blocksize,
            &beta,
            buf8_greater, blocksize,
            greater_inv_diagblk_small_d[stream_compute], blocksize
        ));

        // wait to not overwrite blocks to unload_diag
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], unload_diag[i+1]));        

        cudaErrchk(cudaMemcpyAsync(lesser_inv_diagblk_d,
                    lesser_inv_diagblk_small_d[stream_compute],
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));

        cudaErrchk(cudaMemcpyAsync(greater_inv_diagblk_d,
                    greater_inv_diagblk_small_d[stream_compute],
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));

        cudaErrchk(cudaEventRecord(lesser_greater_calculated[i], stream[stream_compute]));

        // wait to unload_diag for the finish of computations
        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], lesser_greater_calculated[i]));

        cudaErrchk(cudaMemcpyAsync(lesser_inv_diagblk_h + i*blocksize*blocksize,
                    lesser_inv_diagblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        cudaErrchk(cudaMemcpyAsync(greater_inv_diagblk_h + i*blocksize*blocksize,
                    greater_inv_diagblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        // unloading finished
        cudaErrchk(cudaEventRecord(unload_diag[i], stream[stream_memunload]));
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
        if(cusolver_handle[i]) {
            cusolverErrchk(cusolverDnDestroy(cusolver_handle[i]));
        }
    }
    for(int i = 0; i < 2; i++){
        if(system_matrix_diagblk_d[i]) {
            cudaErrchk(cudaFree(system_matrix_diagblk_d[i]));
        }
        if(system_matrix_upperblk_d[i]) {
            cudaErrchk(cudaFree(system_matrix_upperblk_d[i]));
        }
        if(system_matrix_lowerblk_d[i]) {
            cudaErrchk(cudaFree(system_matrix_lowerblk_d[i]));
        }
        if(self_energy_lesser_diagblk_d[i]) {
            cudaErrchk(cudaFree(self_energy_lesser_diagblk_d[i]));
        }
        if(self_energy_lesser_upperblk_d[i]) {
            cudaErrchk(cudaFree(self_energy_lesser_upperblk_d[i]));
        }
        if(self_energy_greater_diagblk_d[i]) {
            cudaErrchk(cudaFree(self_energy_greater_diagblk_d[i]));
        }
        if(self_energy_greater_upperblk_d[i]) {
            cudaErrchk(cudaFree(self_energy_greater_upperblk_d[i]));
        }
    }
    if(inv_diagblk_d) {
        cudaErrchk(cudaFree(inv_diagblk_d));
    }
    if(lesser_inv_diagblk_d) {
        cudaErrchk(cudaFree(lesser_inv_diagblk_d));
    }
    if(lesser_inv_upperblk_d) {
        cudaErrchk(cudaFree(lesser_inv_upperblk_d));
    }
    if(greater_inv_diagblk_d) {
        cudaErrchk(cudaFree(greater_inv_diagblk_d));
    }
    if(greater_inv_upperblk_d) {
        cudaErrchk(cudaFree(greater_inv_upperblk_d));
    }
    if(identity_d){
        cudaErrchk(cudaFree(identity_d));
    }

    for(int i = 0; i < 2; i++){
        if(inv_diagblk_small_d[i]){
            cudaErrchk(cudaFree(inv_diagblk_small_d[i]));
        }   
    }

    if(small_inv_diagblk_h){
        cudaErrchk(cudaFreeHost(small_inv_diagblk_h));
    }

    if(buffer){
        cudaErrchk(cudaFree(buffer));
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
        if(lesser_greater_calculated[i]){
            cudaErrchk(cudaEventDestroy(lesser_greater_calculated[i]));
        }
        if(lesser_greater_calculated_upper[i]){
            cudaErrchk(cudaEventDestroy(lesser_greater_calculated_upper[i]));
        }
    }
    for(unsigned int i = 0; i < n_blocks; i++){
        if(unload_diag[i]){
            cudaErrchk(cudaEventDestroy(unload_diag[i]));
        }
        if(unload_upper[i]){
            cudaErrchk(cudaEventDestroy(unload_upper[i]));
        }
    }

}





