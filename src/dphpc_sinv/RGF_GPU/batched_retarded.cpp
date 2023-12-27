// Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
#include "batched_retarded.h"

void rgf_retarded_for(
    unsigned int blocksize,
    unsigned int matrix_size,
    unsigned int batch_size,
    complex_h **batch_diagblk_h,
    complex_h **batch_upperblk_h,
    complex_h **batch_lowerblk_h,
    complex_h **batch_inv_diagblk_h,
    complex_h **batch_inv_upperblk_h,
    complex_h **batch_inv_lowerblk_h)
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
    
    complex_d* matrix_diagblk_d[2];
    complex_d* matrix_upperblk_d[2];
    complex_d* matrix_lowerblk_d[2];

    // allocate single blocks of the matrix
    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&matrix_diagblk_d[i], blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&matrix_upperblk_d[i], blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&matrix_lowerblk_d[i], blocksize * blocksize * sizeof(complex_d)));
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
    cudaErrchk(cudaMemcpyAsync(identity_d, reinterpret_cast<const complex_d*>(identity_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_compute]));

    //figure out extra amount of memory needed
    complex_d *buffer = NULL;
    int bufferSize = 0;
    cusolverErrchk(cusolverDnZgetrf_bufferSize(cusolver_handle[stream_compute], blocksize, blocksize,
                                            (complex_d *)matrix_diagblk_d[stream_compute],
                                              blocksize, &bufferSize));
    cudaErrchk(cudaMalloc((void**)&buffer, sizeof(complex_d) * bufferSize));



    // ----- END OF INIT SECTION -----

    for(unsigned int batch = 0; batch < batch_size; batch++){

        // init right hand side identity matrix on device for backsubstitution
        cudaErrchk(cudaMemcpyAsync(inv_diagblk_d, identity_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        

        cudaErrchk(cudaMemcpyAsync(matrix_diagblk_d[stream_compute],
                    reinterpret_cast<const complex_d*>(batch_diagblk_h[0] + batch*blocksize*blocksize),
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_compute]));


        cusolverErrchk(cusolverDnZgetrf(cusolver_handle[stream_compute], blocksize, blocksize,
                                    matrix_diagblk_d[stream_compute], blocksize, buffer, ipiv_d, info_d));
        

        //back substitution
        cusolverErrchk(cusolverDnZgetrs(cusolver_handle[stream_compute], CUBLAS_OP_N, blocksize,
                                        blocksize, matrix_diagblk_d[stream_compute], blocksize, ipiv_d,
                                        inv_diagblk_d, blocksize, info_d));

        // record finishing the inverse of the first block
        cudaErrchk(cudaEventRecord(schur_inverted[0], stream[stream_compute]));


        //wait for the inverse of the first block
        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[0]));
        // 0. Inverse of the first block
        cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_h[0] + batch*blocksize*blocksize, inv_diagblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        // unloading finished
        cudaErrchk(cudaEventRecord(unload[0], stream[stream_memunload]));


        // first memcpy happens before loop
        cudaErrchk(cudaMemcpyAsync(matrix_diagblk_d[1],
                    reinterpret_cast<const complex_d*>(batch_diagblk_h[1] + batch*blocksize*blocksize),
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
        cudaErrchk(cudaMemcpyAsync(matrix_upperblk_d[1],
                    reinterpret_cast<const complex_d*>(batch_upperblk_h[0] + batch*blocksize*blocksize),
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
        cudaErrchk(cudaMemcpyAsync(matrix_lowerblk_d[1],
                    reinterpret_cast<const complex_d*>(batch_lowerblk_h[0] + batch*blocksize*blocksize),
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));



        // // 1. Forward substitution (performed left to right)
        for (unsigned int i = 1; i < n_blocks; ++i) {


            int stream_memload = (i+1) % 2;
            int stream_compute = i % 2;
            int stream_memunload = 2;


            if(i < n_blocks-1){
                // load the blocks for the next iteration
                cudaErrchk(cudaMemcpyAsync(matrix_diagblk_d[stream_memload],
                            reinterpret_cast<const complex_d*>(batch_diagblk_h[i+1] + batch*blocksize*blocksize),
                            blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
                cudaErrchk(cudaMemcpyAsync(matrix_upperblk_d[stream_memload],
                            reinterpret_cast<const complex_d*>(batch_upperblk_h[i] + batch*blocksize*blocksize),
                            blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
                cudaErrchk(cudaMemcpyAsync(matrix_lowerblk_d[stream_memload],
                            reinterpret_cast<const complex_d*>(batch_lowerblk_h[i] + batch*blocksize*blocksize),
                            blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            }

            //wait for the schur inverse from the previous iteration
            cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], schur_inverted[i-1]));

            // MatMul tmp = eig_lowerblk[i-1] * eig_inv_diagblk[i-1]
            // use the inv_diagblk_d from last iteration
            // use inv_lowerblk_d as tmp
            alpha = make_cuDoubleComplex(1.0, 0.0);
            beta = make_cuDoubleComplex(0.0, 0.0);

            cublasErrchk(cublasZgemm(
                cublas_handle[stream_compute],
                CUBLAS_OP_N, CUBLAS_OP_N,
                blocksize, blocksize, blocksize,
                &alpha,
                matrix_lowerblk_d[stream_compute], blocksize,
                inv_diagblk_d, blocksize,
                &beta,
                inv_lowerblk_d, blocksize));
            //MatMul schur complement = eig_diagblk[i] - tmp * eig_upperblk[i-1]
            alpha = make_cuDoubleComplex(-1.0, 0.0);
            beta = make_cuDoubleComplex(1.0, 0.0);

            cublasErrchk(cublasZgemm(
                cublas_handle[stream_compute],
                CUBLAS_OP_N, CUBLAS_OP_N,
                blocksize, blocksize, blocksize,
                &alpha,
                inv_lowerblk_d, blocksize,
                matrix_upperblk_d[stream_compute], blocksize,
                &beta,
                matrix_diagblk_d[stream_compute], blocksize));

            // wait to not overwrite block to unload
            cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], unload[i-1]));
            //copy identity
            cudaErrchk(cudaMemcpyAsync(inv_diagblk_d, identity_d,
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
            // inverse schur complement
            cusolverErrchk(cusolverDnZgetrf(cusolver_handle[stream_compute], blocksize, blocksize,
                                        matrix_diagblk_d[stream_compute],
                                        blocksize, buffer, ipiv_d, info_d));
            

            //back substitution
            cusolverErrchk(cusolverDnZgetrs(cusolver_handle[stream_compute], CUBLAS_OP_N, blocksize,
                                            blocksize,
                                            matrix_diagblk_d[stream_compute],
                                            blocksize, ipiv_d,
                                            inv_diagblk_d, blocksize, info_d));
            // record finishing of computation in step i
            cudaErrchk(cudaEventRecord(schur_inverted[i], stream[stream_compute]));

            // wait to unload for the finish of computations
            cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[i]));
            cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_h[i] + batch*blocksize*blocksize,
                        inv_diagblk_d,
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
            // unloading finished
            cudaErrchk(cudaEventRecord(unload[i], stream[stream_memunload]));

        }
        int stream_memload_before = (n_blocks) % 2;
        int stream_compute_before = (n_blocks-1) % 2;

        cudaErrchk(cudaMemcpyAsync(matrix_upperblk_d[stream_memload_before],
                    reinterpret_cast<const complex_d*>(batch_upperblk_h[n_blocks-2] + batch*blocksize*blocksize),
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
        cudaErrchk(cudaMemcpyAsync(matrix_lowerblk_d[stream_memload_before],
                    reinterpret_cast<const complex_d*>(batch_lowerblk_h[n_blocks-2] + batch*blocksize*blocksize),
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
        // possible race condition with unloading of previous loop
        // not sure
        cudaErrchk(cudaStreamWaitEvent(stream[stream_memload_before], unload[n_blocks-2]));
        cudaErrchk(cudaMemcpyAsync(inv_diagblk_small_d[stream_memload_before],
                    reinterpret_cast<const complex_d*>(batch_inv_diagblk_h[n_blocks-2] + batch*blocksize*blocksize),
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    

        // TODO possible to save memory by allocating and freeing
        // memory which is not needed anymore (to reduce max memory consumption at one point)


        // 2. Backward substitution (performed right to left)
        for(int i = n_blocks-2; i >= 0; --i){

            // fix stream compute to be the stream which loaded
            // blocks before the loop
            stream_memload = (stream_compute_before + (i - n_blocks + 2) ) % 2;
            stream_compute = (stream_memload_before + (i - n_blocks + 2) ) % 2;
            stream_memunload = 2;

            if(i > 0){
                cudaErrchk(cudaMemcpyAsync(matrix_upperblk_d[stream_memload],
                            reinterpret_cast<const complex_d*>(batch_upperblk_h[i-1] + batch*blocksize*blocksize),
                            blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
                cudaErrchk(cudaMemcpyAsync(matrix_lowerblk_d[stream_memload],
                            reinterpret_cast<const complex_d*>(batch_lowerblk_h[i-1] + batch*blocksize*blocksize),
                            blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
                cudaErrchk(cudaMemcpyAsync(inv_diagblk_small_d[stream_memload],
                            reinterpret_cast<const complex_d*>(batch_inv_diagblk_h[i-1] + batch*blocksize*blocksize),
                            blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));    
            }
        

            // wait for the block of the last iteration
            cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], schur_inverted[i+1]));

            //tmp = eig_inv_diagblk[i+1] * eig_lowerblk[i]
            // use identity_cpy_d as tmp
            // reuse inv_diagblk_d from last iteration
            // which is the last true inverse block
            alpha = make_cuDoubleComplex(1.0, 0.0);
            beta = make_cuDoubleComplex(0.0, 0.0);
            cublasErrchk(cublasZgemm(
                cublas_handle[stream_compute],
                CUBLAS_OP_N, CUBLAS_OP_N,
                blocksize, blocksize, blocksize,
                &alpha,
                inv_diagblk_d, blocksize,
                matrix_lowerblk_d[stream_compute], blocksize,
                &beta,
                identity_cpy_d, blocksize));

            // eig_inv_lowerblk[i] = -tmp*eig_inv_diagblk[i];
            alpha = make_cuDoubleComplex(-1.0, 0.0);
            beta = make_cuDoubleComplex(0.0, 0.0);
            

            // use temporary buffer for inv_lowerblk_d
            cublasErrchk(cublasZgemm(
                cublas_handle[stream_compute],
                CUBLAS_OP_N, CUBLAS_OP_N,
                blocksize, blocksize, blocksize,
                &alpha,
                identity_cpy_d, blocksize,
                inv_diagblk_small_d[stream_compute], blocksize,
                &beta,
                matrix_diagblk_d[1], blocksize));

            // tmp = eig_inv_diagblk[i] * eig_upperblk[i]
            alpha = make_cuDoubleComplex(1.0, 0.0);
            beta = make_cuDoubleComplex(0.0, 0.0);

            cublasErrchk(cublasZgemm(
                cublas_handle[stream_compute],
                CUBLAS_OP_N, CUBLAS_OP_N,
                blocksize, blocksize, blocksize,
                &alpha,
                inv_diagblk_small_d[stream_compute], blocksize,
                matrix_upperblk_d[stream_compute], blocksize,
                &beta,
                identity_cpy_d, blocksize));

            //eig_inv_upperblk[i] = -tmp * eig_inv_diagblk[i+1];
            alpha = make_cuDoubleComplex(-1.0, 0.0);
            beta = make_cuDoubleComplex(0.0, 0.0);

            // use temporary buffer for inv_upperblk_d
            cublasErrchk(cublasZgemm(
                cublas_handle[stream_compute],
                CUBLAS_OP_N, CUBLAS_OP_N,
                blocksize, blocksize, blocksize,
                &alpha,
                identity_cpy_d, blocksize,
                inv_diagblk_d, blocksize,
                &beta,
                matrix_diagblk_d[0], blocksize));


            //eig_inv_diagblk[i] -= tmp * eig_inv_lowerblk[i];
            alpha = make_cuDoubleComplex(-1.0, 0.0);
            beta = make_cuDoubleComplex(1.0, 0.0);

            cublasErrchk(cublasZgemm(
                cublas_handle[stream_compute],
                CUBLAS_OP_N, CUBLAS_OP_N,
                blocksize, blocksize, blocksize,
                &alpha,
                identity_cpy_d, blocksize,
                matrix_diagblk_d[1], blocksize,
                &beta,
                inv_diagblk_small_d[stream_compute], blocksize));
    

            // wait to not overwrite blocks to unload
            cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], unload[i+1]));

            // use allocated buffers for inv
            // since host2host is cheap
            // matrix_diagblk_d is only used in forward pass
            cudaErrchk(cudaMemcpyAsync(inv_diagblk_d,
                        inv_diagblk_small_d[stream_compute],
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
            cudaErrchk(cudaMemcpyAsync(inv_upperblk_d,
                        matrix_diagblk_d[0],
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
            cudaErrchk(cudaMemcpyAsync(inv_lowerblk_d,
                        matrix_diagblk_d[1],
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
            cudaErrchk(cudaEventRecord(schur_inverted[i], stream[stream_compute]));

            // wait to unload for the finish of computations
            cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[i]));

            cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_h[i] + batch*blocksize*blocksize,
                        inv_diagblk_d,
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
            cudaErrchk(cudaMemcpyAsync(batch_inv_upperblk_h[i] + batch*blocksize*blocksize,
                        inv_upperblk_d,
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
            cudaErrchk(cudaMemcpyAsync(batch_inv_lowerblk_h[i] + batch*blocksize*blocksize,
                        inv_lowerblk_d,
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
            // unloading finished
            cudaErrchk(cudaEventRecord(unload[i], stream[stream_memunload]));
        }
        // synchronize all the streams
        for(int j = 0; j < number_streams; j++){
            cudaErrchk(cudaStreamSynchronize(stream[j]));
        }
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
        if(matrix_diagblk_d[i]) {
            cudaErrchk(cudaFree(matrix_diagblk_d[i]));
        }
        if(matrix_upperblk_d[i]) {
            cudaErrchk(cudaFree(matrix_upperblk_d[i]));
        }
        if(matrix_lowerblk_d[i]) {
            cudaErrchk(cudaFree(matrix_lowerblk_d[i]));
        }
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
    }
    for(unsigned int i = 0; i < n_blocks; i++){
        if(unload[i]){
            cudaErrchk(cudaEventDestroy(unload[i]));
        }
    }

}

void rgf_retarded_batched(
    unsigned int blocksize,
    unsigned int matrix_size,
    unsigned int batch_size,
    complex_h **batch_diagblk_h,
    complex_h **batch_upperblk_h,
    complex_h **batch_lowerblk_h,
    complex_h **batch_inv_diagblk_h,
    complex_h **batch_inv_upperblk_h,
    complex_h **batch_inv_lowerblk_h)
{


    if(matrix_size % blocksize != 0){
        printf("Error: matrix_size is not a multiple of blocksize\n");
        return;
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
    complex_d* batch_diagblk_d[2];
    complex_d* batch_upperblk_d[2];
    complex_d* batch_lowerblk_d[2];

    complex_d* batch_diagblk_ptr_h[2][batch_size];
    complex_d* batch_upperblk_ptr_h[2][batch_size];
    complex_d* batch_lowerblk_ptr_h[2][batch_size];
    complex_d** batch_diagblk_ptr_d[2];
    complex_d** batch_upperblk_ptr_d[2];
    complex_d** batch_lowerblk_ptr_d[2];


    // allocate single blocks of the matrix
    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&batch_diagblk_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&batch_upperblk_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&batch_lowerblk_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
        for(unsigned int j = 0; j < batch_size; j++){
            batch_diagblk_ptr_h[i][j] = batch_diagblk_d[i] + j * blocksize * blocksize;
            batch_upperblk_ptr_h[i][j] = batch_upperblk_d[i] + j * blocksize * blocksize;
            batch_lowerblk_ptr_h[i][j] = batch_lowerblk_d[i] + j * blocksize * blocksize;
        }
    
    }
    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&batch_diagblk_ptr_d[i], batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMalloc((void**)&batch_upperblk_ptr_d[i], batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMalloc((void**)&batch_lowerblk_ptr_d[i], batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMemcpy(batch_diagblk_ptr_d[i], batch_diagblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(batch_upperblk_ptr_d[i], batch_upperblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(batch_lowerblk_ptr_d[i], batch_lowerblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    }



    // allocate memory for the inverse
    complex_d* batch_inv_diagblk_d = NULL;
    complex_d* batch_inv_upperblk_d = NULL;
    complex_d* batch_inv_lowerblk_d = NULL;
    complex_d* intermediate_d = NULL;

    complex_d* batch_inv_diagblk_ptr_h[batch_size];
    complex_d* batch_inv_lowerblk_ptr_h[batch_size];
    complex_d* intermediate_ptr_h[batch_size];

    complex_d** batch_inv_diagblk_ptr_d;
    complex_d** batch_inv_lowerblk_ptr_d;
    complex_d** intermediate_ptr_d;

    // used to for the small g in the forward pass
    complex_d* batch_inv_diagblk_small_d[2];
    complex_d* batch_inv_diagblk_small_ptr_h[2][batch_size];
    complex_d** batch_inv_diagblk_small_ptr_d[2];


    cudaErrchk(cudaMalloc((void**)&batch_inv_diagblk_d, batch_size * blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&batch_inv_upperblk_d, batch_size * blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&batch_inv_lowerblk_d, batch_size * blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&intermediate_d, batch_size * blocksize * blocksize * sizeof(complex_d)));

    for(unsigned int i = 0; i < batch_size; i++){
        batch_inv_diagblk_ptr_h[i] = batch_inv_diagblk_d + i * blocksize * blocksize;
        batch_inv_lowerblk_ptr_h[i] = batch_inv_lowerblk_d + i * blocksize * blocksize;
        intermediate_ptr_h[i] = intermediate_d + i * blocksize * blocksize;
    }
    cudaErrchk(cudaMalloc((void**)&batch_inv_diagblk_ptr_d, batch_size * sizeof(complex_d*)));
    cudaErrchk(cudaMalloc((void**)&batch_inv_lowerblk_ptr_d, batch_size * sizeof(complex_d*)));
    cudaErrchk(cudaMalloc((void**)&intermediate_ptr_d, batch_size * sizeof(complex_d*)));
    cudaErrchk(cudaMemcpy(batch_inv_diagblk_ptr_d, batch_inv_diagblk_ptr_h, batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(batch_inv_lowerblk_ptr_d, batch_inv_lowerblk_ptr_h, batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(intermediate_ptr_d, intermediate_ptr_h, batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));



    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&batch_inv_diagblk_small_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
        for(unsigned int j = 0; j < batch_size; j++){
            batch_inv_diagblk_small_ptr_h[i][j] = batch_inv_diagblk_small_d[i] + j * blocksize * blocksize;
        }
    }
    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&batch_inv_diagblk_small_ptr_d[i], batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMemcpy(batch_inv_diagblk_small_ptr_d[i], batch_inv_diagblk_small_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    }

    //memory for pivoting

    int *info_d = NULL;
    cudaErrchk(cudaMalloc((void**)&info_d, batch_size * sizeof(int)))

    int *ipiv_d = NULL;
    cudaErrchk(cudaMalloc((void**)&ipiv_d, batch_size * blocksize * sizeof(int)));

    // ----- END OF INIT SECTION -----


    cudaErrchk(cudaMemcpyAsync(batch_diagblk_d[stream_compute], reinterpret_cast<const complex_d*>(batch_diagblk_h[0]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_compute]));


    cublasErrchk(cublasZgetrfBatched(
            cublas_handle[stream_compute],
            blocksize,
            batch_diagblk_ptr_d[stream_compute], blocksize, ipiv_d,
            info_d, batch_size));


    // inversion
    cublasErrchk(cublasZgetriBatched(
                                cublas_handle[stream_compute],
                                blocksize,
                                batch_diagblk_ptr_d[stream_compute],
                                blocksize,
                                ipiv_d,
                                batch_inv_diagblk_ptr_d,
                                blocksize,
                                info_d,
                                batch_size));


    // record finishing the inverse of the first block
    cudaErrchk(cudaEventRecord(schur_inverted[0], stream[stream_compute]));


    //wait for the inverse of the first block
    cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[0]));
    // 0. Inverse of the first block
    cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_h[0], batch_inv_diagblk_d,
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
    // unloading finished
    cudaErrchk(cudaEventRecord(unload[0], stream[stream_memunload]));


    // first memcpy happens before loop
    cudaErrchk(cudaMemcpyAsync(batch_diagblk_d[1],
                reinterpret_cast<const complex_d*>(batch_diagblk_h[1]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(batch_upperblk_d[1],
                reinterpret_cast<const complex_d*>(batch_upperblk_h[0]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(batch_lowerblk_d[1],
                reinterpret_cast<const complex_d*>(batch_lowerblk_h[0]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));



    // // 1. Forward substitution (performed left to right)
    for (unsigned int i = 1; i < n_blocks; ++i) {


        int stream_memload = (i+1) % 2;
        int stream_compute = i % 2;
        int stream_memunload = 2;


        if(i < n_blocks-1){
            // load the blocks for the next iteration
            cudaErrchk(cudaMemcpyAsync(batch_diagblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(batch_diagblk_h[i+1]),
                        batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(batch_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(batch_upperblk_h[i]),
                        batch_size *  blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(batch_lowerblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(batch_lowerblk_h[i]),
                        batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
        }

        //wait for the schur inverse from the previous iteration
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], schur_inverted[i-1]));

        // MatMul tmp = eig_lowerblk[i-1] * eig_inv_diagblk[i-1]
        // use the batch_inv_diagblk_d from last iteration
        // use batch_inv_lowerblk_d as tmp
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            batch_lowerblk_ptr_d[stream_compute], blocksize,
            batch_inv_diagblk_ptr_d, blocksize,
            &beta,
            batch_inv_lowerblk_ptr_d, blocksize, batch_size));
        //MatMul schur complement = eig_diagblk[i] - tmp * eig_upperblk[i-1]
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);

        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            batch_inv_lowerblk_ptr_d, blocksize,
            batch_upperblk_ptr_d[stream_compute], blocksize,
            &beta,
            batch_diagblk_ptr_d[stream_compute], blocksize, batch_size));



        cublasErrchk(cublasZgetrfBatched(
                cublas_handle[stream_compute],
                blocksize,
                batch_diagblk_ptr_d[stream_compute], blocksize, ipiv_d,
                info_d, batch_size));


        // inversion
        cublasErrchk(cublasZgetriBatched(
                                    cublas_handle[stream_compute],
                                    blocksize,
                                    batch_diagblk_ptr_d[stream_compute],
                                    blocksize,
                                    ipiv_d,
                                    batch_inv_diagblk_ptr_d,
                                    blocksize,
                                    info_d,
                                    batch_size));



        // record finishing of computation in step i
        cudaErrchk(cudaEventRecord(schur_inverted[i], stream[stream_compute]));

        // wait to unload for the finish of computations
        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[i]));
        cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_h[i],
                    batch_inv_diagblk_d,
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        // unloading finished
        cudaErrchk(cudaEventRecord(unload[i], stream[stream_memunload]));

    }


    int stream_memload_before = (n_blocks) % 2;
    int stream_compute_before = (n_blocks-1) % 2;

    cudaErrchk(cudaMemcpyAsync(batch_upperblk_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(batch_upperblk_h[n_blocks-2]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    cudaErrchk(cudaMemcpyAsync(batch_lowerblk_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(batch_lowerblk_h[n_blocks-2]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    // possible race condition with unloading of previous loop
    // not sure
    cudaErrchk(cudaStreamWaitEvent(stream[stream_memload_before], unload[n_blocks-2]));
    cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_small_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(batch_inv_diagblk_h[n_blocks-2]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
  

    // TODO possible to save memory by allocating and freeing
    // memory which is not needed anymore (to reduce max memory consumption at one point)


    // 2. Backward substitution (performed right to left)
    for(int i = n_blocks-2; i >= 0; --i){

        // fix stream compute to be the stream which loaded
        // blocks before the loop
        stream_memload = (stream_compute_before + (i - n_blocks + 2) ) % 2;
        stream_compute = (stream_memload_before + (i - n_blocks + 2) ) % 2;
        stream_memunload = 2;

        if(i > 0){
            cudaErrchk(cudaMemcpyAsync(batch_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(batch_upperblk_h[i-1]),
                        batch_size *blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(batch_lowerblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(batch_lowerblk_h[i-1]),
                        batch_size *blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_small_d[stream_memload],
                        reinterpret_cast<const complex_d*>(batch_inv_diagblk_h[i-1]),
                        batch_size *blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));    
        }
    

        // wait for the block of the last iteration
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], schur_inverted[i+1]));

        //tmp = eig_inv_diagblk[i+1] * eig_lowerblk[i]
        // use intermediate_ptr_d as tmp
        // reuse batch_inv_diagblk_d from last iteration
        // which is the last true inverse block
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            batch_inv_diagblk_ptr_d, blocksize,
            batch_lowerblk_ptr_d[stream_compute], blocksize,
            &beta,
            intermediate_ptr_d, blocksize, batch_size));

        // eig_inv_lowerblk[i] = -tmp*eig_inv_diagblk[i];
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        

        // use temporary buffer for batch_inv_lowerblk_d
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            intermediate_ptr_d, blocksize,
            batch_inv_diagblk_small_ptr_d[stream_compute], blocksize,
            &beta,
            batch_diagblk_ptr_d[1], blocksize, batch_size));

        // tmp = eig_inv_diagblk[i] * eig_upperblk[i]
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            batch_inv_diagblk_small_ptr_d[stream_compute], blocksize,
            batch_upperblk_ptr_d[stream_compute], blocksize,
            &beta,
            intermediate_ptr_d, blocksize, batch_size));

        //eig_inv_upperblk[i] = -tmp * eig_inv_diagblk[i+1];
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        // use temporary buffer for batch_inv_upperblk_d
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            intermediate_ptr_d, blocksize,
            batch_inv_diagblk_ptr_d, blocksize,
            &beta,
            batch_diagblk_ptr_d[0], blocksize, batch_size));


        //eig_inv_diagblk[i] -= tmp * eig_inv_lowerblk[i];
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);

        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            intermediate_ptr_d, blocksize,
            batch_diagblk_ptr_d[1], blocksize,
            &beta,
            batch_inv_diagblk_small_ptr_d[stream_compute], blocksize, batch_size));
 

        // wait to not overwrite blocks to unload
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], unload[i+1]));

        // use allocated buffers for inv
        // since host2host is cheap
        // batch_diagblk_d is only used in forward pass
        cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_d,
                    batch_inv_diagblk_small_d[stream_compute],
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        cudaErrchk(cudaMemcpyAsync(batch_inv_upperblk_d,
                    batch_diagblk_d[0],
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        cudaErrchk(cudaMemcpyAsync(batch_inv_lowerblk_d,
                    batch_diagblk_d[1],
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        cudaErrchk(cudaEventRecord(schur_inverted[i], stream[stream_compute]));

        // wait to unload for the finish of computations
        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[i]));

        cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_h[i],
                    batch_inv_diagblk_d,
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        cudaErrchk(cudaMemcpyAsync(batch_inv_upperblk_h[i],
                    batch_inv_upperblk_d,
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        cudaErrchk(cudaMemcpyAsync(batch_inv_lowerblk_h[i],
                    batch_inv_lowerblk_d,
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
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
        if(cusolver_handle[i]) {
            cusolverErrchk(cusolverDnDestroy(cusolver_handle[i]));
        }
    }
    for(int i = 0; i < 2; i++){
        if(batch_diagblk_d[i]) {
            cudaErrchk(cudaFree(batch_diagblk_d[i]));
        }
        if(batch_upperblk_d[i]) {
            cudaErrchk(cudaFree(batch_upperblk_d[i]));
        }
        if(batch_lowerblk_d[i]) {
            cudaErrchk(cudaFree(batch_lowerblk_d[i]));
        }
        if(batch_diagblk_ptr_d[i]) {
            cudaErrchk(cudaFree(batch_diagblk_ptr_d[i]));
        }
        if(batch_upperblk_ptr_d[i]) {
            cudaErrchk(cudaFree(batch_upperblk_ptr_d[i]));
        }
        if(batch_lowerblk_ptr_d[i]) {
            cudaErrchk(cudaFree(batch_lowerblk_ptr_d[i]));
        }

    }
    if(batch_inv_diagblk_d) {
        cudaErrchk(cudaFree(batch_inv_diagblk_d));
    }
    if(batch_inv_upperblk_d) {
        cudaErrchk(cudaFree(batch_inv_upperblk_d));
    }
    if(batch_inv_lowerblk_d) {
        cudaErrchk(cudaFree(batch_inv_lowerblk_d));
    }
    if(intermediate_d){
        cudaErrchk(cudaFree(intermediate_d));
    }


    if(batch_inv_diagblk_ptr_d){
        cudaErrchk(cudaFree(batch_inv_diagblk_ptr_d));
    }
    if(batch_inv_lowerblk_ptr_d){
        cudaErrchk(cudaFree(batch_inv_lowerblk_ptr_d));
    }
    if(intermediate_ptr_d){
        cudaErrchk(cudaFree(intermediate_ptr_d));
    }

    for(int i = 0; i < 2; i++){
        if(batch_inv_diagblk_small_d[i]){
            cudaErrchk(cudaFree(batch_inv_diagblk_small_d[i]));
        }    
        if(batch_inv_diagblk_small_ptr_d[i]){
            cudaErrchk(cudaFree(batch_inv_diagblk_small_ptr_d[i]));
        }

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

}


void rgf_retarded_batched_strided(
    unsigned int blocksize,
    unsigned int matrix_size,
    unsigned int batch_size,
    complex_h **batch_diagblk_h,
    complex_h **batch_upperblk_h,
    complex_h **batch_lowerblk_h,
    complex_h **batch_inv_diagblk_h,
    complex_h **batch_inv_upperblk_h,
    complex_h **batch_inv_lowerblk_h)
{


    if(matrix_size % blocksize != 0){
        printf("Error: matrix_size is not a multiple of blocksize\n");
        return;
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
    complex_d* batch_diagblk_d[2];
    complex_d* batch_upperblk_d[2];
    complex_d* batch_lowerblk_d[2];

    complex_d* batch_diagblk_ptr_h[2][batch_size];
    complex_d* batch_upperblk_ptr_h[2][batch_size];
    complex_d* batch_lowerblk_ptr_h[2][batch_size];
    complex_d** batch_diagblk_ptr_d[2];
    complex_d** batch_upperblk_ptr_d[2];
    complex_d** batch_lowerblk_ptr_d[2];


    // allocate single blocks of the matrix
    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&batch_diagblk_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&batch_upperblk_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&batch_lowerblk_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
        for(unsigned int j = 0; j < batch_size; j++){
            batch_diagblk_ptr_h[i][j] = batch_diagblk_d[i] + j * blocksize * blocksize;
            batch_upperblk_ptr_h[i][j] = batch_upperblk_d[i] + j * blocksize * blocksize;
            batch_lowerblk_ptr_h[i][j] = batch_lowerblk_d[i] + j * blocksize * blocksize;
        }
    
    }
    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&batch_diagblk_ptr_d[i], batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMalloc((void**)&batch_upperblk_ptr_d[i], batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMalloc((void**)&batch_lowerblk_ptr_d[i], batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMemcpy(batch_diagblk_ptr_d[i], batch_diagblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(batch_upperblk_ptr_d[i], batch_upperblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(batch_lowerblk_ptr_d[i], batch_lowerblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    }



    // allocate memory for the inverse
    complex_d* batch_inv_diagblk_d = NULL;
    complex_d* batch_inv_upperblk_d = NULL;
    complex_d* batch_inv_lowerblk_d = NULL;
    complex_d* intermediate_d = NULL;

    complex_d* batch_inv_diagblk_ptr_h[batch_size];
    complex_d* batch_inv_lowerblk_ptr_h[batch_size];
    complex_d* intermediate_ptr_h[batch_size];

    complex_d** batch_inv_diagblk_ptr_d;
    complex_d** batch_inv_lowerblk_ptr_d;
    complex_d** intermediate_ptr_d;

    // used to for the small g in the forward pass
    complex_d* batch_inv_diagblk_small_d[2];
    complex_d* batch_inv_diagblk_small_ptr_h[2][batch_size];
    complex_d** batch_inv_diagblk_small_ptr_d[2];


    cudaErrchk(cudaMalloc((void**)&batch_inv_diagblk_d, batch_size * blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&batch_inv_upperblk_d, batch_size * blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&batch_inv_lowerblk_d, batch_size * blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&intermediate_d, batch_size * blocksize * blocksize * sizeof(complex_d)));

    for(unsigned int i = 0; i < batch_size; i++){
        batch_inv_diagblk_ptr_h[i] = batch_inv_diagblk_d + i * blocksize * blocksize;
        batch_inv_lowerblk_ptr_h[i] = batch_inv_lowerblk_d + i * blocksize * blocksize;
        intermediate_ptr_h[i] = intermediate_d + i * blocksize * blocksize;
    }
    cudaErrchk(cudaMalloc((void**)&batch_inv_diagblk_ptr_d, batch_size * sizeof(complex_d*)));
    cudaErrchk(cudaMalloc((void**)&batch_inv_lowerblk_ptr_d, batch_size * sizeof(complex_d*)));
    cudaErrchk(cudaMalloc((void**)&intermediate_ptr_d, batch_size * sizeof(complex_d*)));
    cudaErrchk(cudaMemcpy(batch_inv_diagblk_ptr_d, batch_inv_diagblk_ptr_h, batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(batch_inv_lowerblk_ptr_d, batch_inv_lowerblk_ptr_h, batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(intermediate_ptr_d, intermediate_ptr_h, batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));



    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&batch_inv_diagblk_small_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
        for(unsigned int j = 0; j < batch_size; j++){
            batch_inv_diagblk_small_ptr_h[i][j] = batch_inv_diagblk_small_d[i] + j * blocksize * blocksize;
        }
    }
    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&batch_inv_diagblk_small_ptr_d[i], batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMemcpy(batch_inv_diagblk_small_ptr_d[i], batch_inv_diagblk_small_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    }

    //memory for pivoting

    int *info_d = NULL;
    cudaErrchk(cudaMalloc((void**)&info_d, batch_size * sizeof(int)))

    int *ipiv_d = NULL;
    cudaErrchk(cudaMalloc((void**)&ipiv_d, batch_size * blocksize * sizeof(int)));

    // ----- END OF INIT SECTION -----


    cudaErrchk(cudaMemcpyAsync(batch_diagblk_d[stream_compute], reinterpret_cast<const complex_d*>(batch_diagblk_h[0]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_compute]));


    cublasErrchk(cublasZgetrfBatched(
            cublas_handle[stream_compute],
            blocksize,
            batch_diagblk_ptr_d[stream_compute], blocksize, ipiv_d,
            info_d, batch_size));


    // inversion
    cublasErrchk(cublasZgetriBatched(
                                cublas_handle[stream_compute],
                                blocksize,
                                batch_diagblk_ptr_d[stream_compute],
                                blocksize,
                                ipiv_d,
                                batch_inv_diagblk_ptr_d,
                                blocksize,
                                info_d,
                                batch_size));


    // record finishing the inverse of the first block
    cudaErrchk(cudaEventRecord(schur_inverted[0], stream[stream_compute]));


    //wait for the inverse of the first block
    cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[0]));
    // 0. Inverse of the first block
    cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_h[0], batch_inv_diagblk_d,
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
    // unloading finished
    cudaErrchk(cudaEventRecord(unload[0], stream[stream_memunload]));


    // first memcpy happens before loop
    cudaErrchk(cudaMemcpyAsync(batch_diagblk_d[1],
                reinterpret_cast<const complex_d*>(batch_diagblk_h[1]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(batch_upperblk_d[1],
                reinterpret_cast<const complex_d*>(batch_upperblk_h[0]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(batch_lowerblk_d[1],
                reinterpret_cast<const complex_d*>(batch_lowerblk_h[0]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));



    // // 1. Forward substitution (performed left to right)
    for (unsigned int i = 1; i < n_blocks; ++i) {


        int stream_memload = (i+1) % 2;
        int stream_compute = i % 2;
        int stream_memunload = 2;


        if(i < n_blocks-1){
            // load the blocks for the next iteration
            cudaErrchk(cudaMemcpyAsync(batch_diagblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(batch_diagblk_h[i+1]),
                        batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(batch_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(batch_upperblk_h[i]),
                        batch_size *  blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(batch_lowerblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(batch_lowerblk_h[i]),
                        batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
        }

        //wait for the schur inverse from the previous iteration
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], schur_inverted[i-1]));

        // MatMul tmp = eig_lowerblk[i-1] * eig_inv_diagblk[i-1]
        // use the batch_inv_diagblk_d from last iteration
        // use batch_inv_lowerblk_d as tmp
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        cublasErrchk(cublasZgemmStridedBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            batch_lowerblk_d[stream_compute], blocksize,
            blocksize * blocksize,
            batch_inv_diagblk_d, blocksize,
            blocksize * blocksize,
            &beta,
            batch_inv_lowerblk_d, blocksize,
            blocksize * blocksize,
            batch_size));
        //MatMul schur complement = eig_diagblk[i] - tmp * eig_upperblk[i-1]
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);

        cublasErrchk(cublasZgemmStridedBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            batch_inv_lowerblk_d, blocksize,
            blocksize * blocksize,
            batch_upperblk_d[stream_compute], blocksize,
            blocksize * blocksize,
            &beta,
            batch_diagblk_d[stream_compute], blocksize,
            blocksize * blocksize,
            batch_size));



        cublasErrchk(cublasZgetrfBatched(
                cublas_handle[stream_compute],
                blocksize,
                batch_diagblk_ptr_d[stream_compute], blocksize, ipiv_d,
                info_d, batch_size));


        // inversion
        cublasErrchk(cublasZgetriBatched(
                                    cublas_handle[stream_compute],
                                    blocksize,
                                    batch_diagblk_ptr_d[stream_compute],
                                    blocksize,
                                    ipiv_d,
                                    batch_inv_diagblk_ptr_d,
                                    blocksize,
                                    info_d,
                                    batch_size));



        // record finishing of computation in step i
        cudaErrchk(cudaEventRecord(schur_inverted[i], stream[stream_compute]));

        // wait to unload for the finish of computations
        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[i]));
        cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_h[i],
                    batch_inv_diagblk_d,
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        // unloading finished
        cudaErrchk(cudaEventRecord(unload[i], stream[stream_memunload]));

    }


    int stream_memload_before = (n_blocks) % 2;
    int stream_compute_before = (n_blocks-1) % 2;

    cudaErrchk(cudaMemcpyAsync(batch_upperblk_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(batch_upperblk_h[n_blocks-2]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    cudaErrchk(cudaMemcpyAsync(batch_lowerblk_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(batch_lowerblk_h[n_blocks-2]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    // possible race condition with unloading of previous loop
    // not sure
    cudaErrchk(cudaStreamWaitEvent(stream[stream_memload_before], unload[n_blocks-2]));
    cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_small_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(batch_inv_diagblk_h[n_blocks-2]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
  

    // TODO possible to save memory by allocating and freeing
    // memory which is not needed anymore (to reduce max memory consumption at one point)


    // 2. Backward substitution (performed right to left)
    for(int i = n_blocks-2; i >= 0; --i){

        // fix stream compute to be the stream which loaded
        // blocks before the loop
        stream_memload = (stream_compute_before + (i - n_blocks + 2) ) % 2;
        stream_compute = (stream_memload_before + (i - n_blocks + 2) ) % 2;
        stream_memunload = 2;

        if(i > 0){
            cudaErrchk(cudaMemcpyAsync(batch_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(batch_upperblk_h[i-1]),
                        batch_size *blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(batch_lowerblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(batch_lowerblk_h[i-1]),
                        batch_size *blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_small_d[stream_memload],
                        reinterpret_cast<const complex_d*>(batch_inv_diagblk_h[i-1]),
                        batch_size *blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));    
        }
    

        // wait for the block of the last iteration
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], schur_inverted[i+1]));

        //tmp = eig_inv_diagblk[i+1] * eig_lowerblk[i]
        // use intermediate_ptr_d as tmp
        // reuse batch_inv_diagblk_d from last iteration
        // which is the last true inverse block
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemmStridedBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            batch_inv_diagblk_d, blocksize,
            blocksize * blocksize,
            batch_lowerblk_d[stream_compute], blocksize,
            blocksize * blocksize,
            &beta,
            intermediate_d, blocksize,
            blocksize * blocksize,
            batch_size));

        // eig_inv_lowerblk[i] = -tmp*eig_inv_diagblk[i];
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        

        // use temporary buffer for batch_inv_lowerblk_d
        cublasErrchk(cublasZgemmStridedBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            intermediate_d, blocksize,
            blocksize * blocksize,
            batch_inv_diagblk_small_d[stream_compute], blocksize,
            blocksize * blocksize,
            &beta,
            batch_diagblk_d[1], blocksize,
            blocksize * blocksize,
            batch_size));

        // tmp = eig_inv_diagblk[i] * eig_upperblk[i]
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        cublasErrchk(cublasZgemmStridedBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            batch_inv_diagblk_small_d[stream_compute], blocksize,
            blocksize * blocksize,
            batch_upperblk_d[stream_compute], blocksize,
            blocksize * blocksize,
            &beta,
            intermediate_d, blocksize,
            blocksize * blocksize,
            batch_size));

        //eig_inv_upperblk[i] = -tmp * eig_inv_diagblk[i+1];
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        // use temporary buffer for batch_inv_upperblk_d
        cublasErrchk(cublasZgemmStridedBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            intermediate_d, blocksize,
            blocksize * blocksize,
            batch_inv_diagblk_d, blocksize,
            blocksize * blocksize,
            &beta,
            batch_diagblk_d[0], blocksize,
            blocksize * blocksize,
            batch_size));


        //eig_inv_diagblk[i] -= tmp * eig_inv_lowerblk[i];
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);

        cublasErrchk(cublasZgemmStridedBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            intermediate_d, blocksize,
            blocksize * blocksize,
            batch_diagblk_d[1], blocksize,
            blocksize * blocksize,
            &beta,
            batch_inv_diagblk_small_d[stream_compute], blocksize,
            blocksize * blocksize,
            batch_size));
 

        // wait to not overwrite blocks to unload
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], unload[i+1]));

        // use allocated buffers for inv
        // since host2host is cheap
        // batch_diagblk_d is only used in forward pass
        cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_d,
                    batch_inv_diagblk_small_d[stream_compute],
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        cudaErrchk(cudaMemcpyAsync(batch_inv_upperblk_d,
                    batch_diagblk_d[0],
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        cudaErrchk(cudaMemcpyAsync(batch_inv_lowerblk_d,
                    batch_diagblk_d[1],
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        cudaErrchk(cudaEventRecord(schur_inverted[i], stream[stream_compute]));

        // wait to unload for the finish of computations
        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[i]));

        cudaErrchk(cudaMemcpyAsync(batch_inv_diagblk_h[i],
                    batch_inv_diagblk_d,
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        cudaErrchk(cudaMemcpyAsync(batch_inv_upperblk_h[i],
                    batch_inv_upperblk_d,
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        cudaErrchk(cudaMemcpyAsync(batch_inv_lowerblk_h[i],
                    batch_inv_lowerblk_d,
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
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
        if(cusolver_handle[i]) {
            cusolverErrchk(cusolverDnDestroy(cusolver_handle[i]));
        }
    }
    for(int i = 0; i < 2; i++){
        if(batch_diagblk_d[i]) {
            cudaErrchk(cudaFree(batch_diagblk_d[i]));
        }
        if(batch_upperblk_d[i]) {
            cudaErrchk(cudaFree(batch_upperblk_d[i]));
        }
        if(batch_lowerblk_d[i]) {
            cudaErrchk(cudaFree(batch_lowerblk_d[i]));
        }
        if(batch_diagblk_ptr_d[i]) {
            cudaErrchk(cudaFree(batch_diagblk_ptr_d[i]));
        }
        if(batch_upperblk_ptr_d[i]) {
            cudaErrchk(cudaFree(batch_upperblk_ptr_d[i]));
        }
        if(batch_lowerblk_ptr_d[i]) {
            cudaErrchk(cudaFree(batch_lowerblk_ptr_d[i]));
        }

    }
    if(batch_inv_diagblk_d) {
        cudaErrchk(cudaFree(batch_inv_diagblk_d));
    }
    if(batch_inv_upperblk_d) {
        cudaErrchk(cudaFree(batch_inv_upperblk_d));
    }
    if(batch_inv_lowerblk_d) {
        cudaErrchk(cudaFree(batch_inv_lowerblk_d));
    }
    if(intermediate_d){
        cudaErrchk(cudaFree(intermediate_d));
    }


    if(batch_inv_diagblk_ptr_d){
        cudaErrchk(cudaFree(batch_inv_diagblk_ptr_d));
    }
    if(batch_inv_lowerblk_ptr_d){
        cudaErrchk(cudaFree(batch_inv_lowerblk_ptr_d));
    }
    if(intermediate_ptr_d){
        cudaErrchk(cudaFree(intermediate_ptr_d));
    }

    for(int i = 0; i < 2; i++){
        if(batch_inv_diagblk_small_d[i]){
            cudaErrchk(cudaFree(batch_inv_diagblk_small_d[i]));
        }    
        if(batch_inv_diagblk_small_ptr_d[i]){
            cudaErrchk(cudaFree(batch_inv_diagblk_small_ptr_d[i]));
        }

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

}
