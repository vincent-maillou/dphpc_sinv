#include "PSR.h"

void myFunction(Eigen::MatrixXcd& A) {
    // Your function logic here
    std::cout << A << std::endl;
}

void load_matrix(
    std::string filename, 
    std::complex<double> *matrix, 
    int rows, 
    int cols)
{
    // Open the binary file for reading
    std::ifstream input(filename, std::ios::binary);
    if (input.is_open()) {
        // Read the binary data into the std::complex<double> array
        input.read(reinterpret_cast<char*>(matrix), sizeof(std::complex<double>) * rows * cols);

        // Check if the read operation was successful
        if (!input) {
            std::cerr << "Read operation failed or reached the end of the file." << std::endl;
        } 
        // Close the input file
        input.close();
    } else {
        std::cerr << "Failed to open the binary file for reading." << std::endl;
    }
}




void rgf_for_subsystem(
    unsigned int blocksize,
    unsigned int matrix_size,
    complex_h *input_contiguous_h,
    complex_h *output_contiguous_h)
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
    complex_d* inv_diagblk_small_d;
    complex_d* inv_upperblk_d = NULL;
    complex_d* inv_lowerblk_d = NULL;

    cudaErrchk(cudaMalloc((void**)&inv_diagblk_small_d, blocksize * matrix_size * sizeof(complex_d)));

    
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


    //figure out extra amount of memory needed
    complex_d *buffer = NULL;
    int bufferSize = 0;
    cusolverErrchk(cusolverDnZgetrf_bufferSize(cusolver_handle[stream_compute], blocksize, blocksize,
                                            (complex_d *)matrix_diagblk_d[stream_compute],
                                              blocksize, &bufferSize));
    cudaErrchk(cudaMalloc((void**)&buffer, sizeof(complex_d) * bufferSize));

    // ----- END OF INIT SECTION -----

    // init right hand side identity matrix on device for backsub
    cudaErrchk(cudaMemcpyAsync(identity_d, reinterpret_cast<const complex_d*>(identity_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_compute]));
    cudaErrchk(cudaMemcpyAsync(inv_diagblk_small_d, identity_d,
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
    

    cudaErrchk(cudaMemcpyAsync(matrix_diagblk_d[stream_compute], reinterpret_cast<const complex_d*>(input_contiguous_h),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_compute]));


    cusolverErrchk(cusolverDnZgetrf(cusolver_handle[stream_compute], blocksize, blocksize,
                                matrix_diagblk_d[stream_compute], blocksize, buffer, ipiv_d, info_d));
    

    //back substitution
    cusolverErrchk(cusolverDnZgetrs(cusolver_handle[stream_compute], CUBLAS_OP_N, blocksize,
                                    blocksize, matrix_diagblk_d[stream_compute], blocksize, ipiv_d,
                                    inv_diagblk_small_d, blocksize, info_d));

    // record finishing the inverse of the first block
    cudaErrchk(cudaEventRecord(schur_inverted[0], stream[stream_compute]));




    // first memcpy happens before loop
    cudaErrchk(cudaMemcpyAsync(matrix_diagblk_d[1],
                reinterpret_cast<const complex_d*>(input_contiguous_h + 3*blocksize*blocksize),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(matrix_upperblk_d[1],
                reinterpret_cast<const complex_d*>(input_contiguous_h + blocksize*blocksize),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(matrix_lowerblk_d[1],
                reinterpret_cast<const complex_d*>(input_contiguous_h + 2*blocksize*blocksize),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));




    // // 1. Forward substitution (performed left to right)
    for (unsigned int i = 1; i < n_blocks; ++i) {


        int stream_memload = (i+1) % 2;
        int stream_compute = i % 2;


        if(i < n_blocks-1){
            // load the blocks for the next iteration
            cudaErrchk(cudaMemcpyAsync(matrix_diagblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(input_contiguous_h + 3*(i+1)*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(matrix_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(input_contiguous_h + 3*(i)*blocksize*blocksize+blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(matrix_lowerblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(input_contiguous_h  + 3*(i)*blocksize*blocksize+2*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
        }

        //wait for the schur inverse from the previous iteration
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], schur_inverted[i-1]));

        // MatMul tmp = eig_lowerblk[i-1] * eig_inv_diagblk[i-1]
        // use the inv_diagblk_small_d from last iteration
        // use inv_lowerblk_d as tmp
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            matrix_lowerblk_d[stream_compute], blocksize,
            inv_diagblk_small_d + (i-1)*blocksize*blocksize, blocksize,
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

        //copy identity
        cudaErrchk(cudaMemcpyAsync(inv_diagblk_small_d + i*blocksize*blocksize, identity_d,
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
                                        inv_diagblk_small_d + i*blocksize*blocksize, blocksize, info_d));
        
        
        // record finishing of computation in step i
        cudaErrchk(cudaEventRecord(schur_inverted[i], stream[stream_compute]));


    }

    // last small inverse is true inverse
    cudaErrchk(cudaStreamWaitEvent(stream[2], schur_inverted[n_blocks-1]));
    cudaErrchk(cudaMemcpyAsync(output_contiguous_h + 3*(n_blocks-1)*blocksize*blocksize,
                inv_diagblk_small_d + (n_blocks-1)*blocksize*blocksize,
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[2]));


    int stream_memload_before = (n_blocks) % 2;
    int stream_compute_before = (n_blocks-1) % 2;

    cudaErrchk(cudaMemcpyAsync(matrix_upperblk_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(input_contiguous_h + 3*(n_blocks-2)*blocksize*blocksize+blocksize*blocksize),
                blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    cudaErrchk(cudaMemcpyAsync(matrix_lowerblk_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(input_contiguous_h + 3*(n_blocks-2)*blocksize*blocksize+2*blocksize*blocksize),
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
                        reinterpret_cast<const complex_d*>(input_contiguous_h + 3*(i-1)*blocksize*blocksize+blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(matrix_lowerblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(input_contiguous_h + 3*(i-1)*blocksize*blocksize+2*blocksize*blocksize),
                        blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));   
        }
    

        // wait for the block of the last iteration
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], schur_inverted[i+1]));

        //tmp = eig_inv_diagblk[i+1] * eig_lowerblk[i]
        // use identity_cpy_d as tmp
        // reuse inv_diagblk_small_d from last iteration
        // which is the last true inverse block
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemm(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            inv_diagblk_small_d + (i+1)*blocksize*blocksize, blocksize,
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
            inv_diagblk_small_d + (i)*blocksize*blocksize, blocksize,
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
            inv_diagblk_small_d + (i)*blocksize*blocksize, blocksize,
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
            inv_diagblk_small_d + (i+1)*blocksize*blocksize, blocksize,
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
            inv_diagblk_small_d + (i)*blocksize*blocksize, blocksize));
 

        // wait to not overwrite blocks to unload
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], unload[i+1]));

        // use allocated buffers for inv
        // since host2host is cheap
        // matrix_diagblk_d is only used in forward pass
        cudaErrchk(cudaMemcpyAsync(inv_upperblk_d,
                    matrix_diagblk_d[0],
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        cudaErrchk(cudaMemcpyAsync(inv_lowerblk_d,
                    matrix_diagblk_d[1],
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        cudaErrchk(cudaEventRecord(schur_inverted[i], stream[stream_compute]));

        // wait to unload for the finish of computations
        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[i]));

        cudaErrchk(cudaMemcpyAsync(output_contiguous_h + 3*i*blocksize*blocksize,
                    inv_diagblk_small_d + (i)*blocksize*blocksize,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        cudaErrchk(cudaMemcpyAsync(output_contiguous_h + 3*i*blocksize*blocksize + blocksize*blocksize,
                    inv_upperblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        cudaErrchk(cudaMemcpyAsync(output_contiguous_h + 3*i*blocksize*blocksize + 2*blocksize*blocksize,
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


    if(inv_diagblk_small_d){
        cudaErrchk(cudaFree(inv_diagblk_small_d));
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


void reduce_schur_topleftcorner_gpu(
	int partition_blocksize,
   	int blocksize,
   	cudaStream_t stream,
	cusolverDnHandle_t cusolver_handle,
    	cublasHandle_t cublas_handle,
    	cuDoubleComplex* A_gpu,
    	cuDoubleComplex* L_gpu,
    	cuDoubleComplex* U_gpu,
    	cuDoubleComplex* identity_d,
    	int l_dim
) {

    // init right hand side identity matrix on device for inversion 
    cuDoubleComplex* identity_cpy_d = NULL;
    cuDoubleComplex* diag_buff_d = NULL;

    cudaErrchk(cudaMalloc((void**)&identity_cpy_d, blocksize * blocksize * sizeof(cuDoubleComplex)));
    cudaErrchk(cudaMalloc((void**)&diag_buff_d, blocksize * blocksize * sizeof(cuDoubleComplex)));

    cudaErrchk(cudaMemcpy(identity_cpy_d, identity_d,
                blocksize * blocksize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

    

    // Allocate buffer needed for LU-Decompositions of the to-be inverted Matrices + buffers for pivots and info flags
    int info_h = 0;
    int *ipiv_d = NULL;
    int *info_d = NULL;
    cudaErrchk(cudaMalloc((void**)&info_d, sizeof(int)))
    cudaErrchk(cudaMalloc((void**)&ipiv_d, blocksize*sizeof(int)));

    cuDoubleComplex *buffer = NULL;
    int bufferSize = 0;
    cusolverErrchk(cusolverDnZgetrf_bufferSize(cusolver_handle, blocksize, blocksize,
                                               diag_buff_d, blocksize,
					       &bufferSize));
    cudaErrchk(cudaMalloc((void**)&buffer, sizeof(cuDoubleComplex) * bufferSize));

    cuDoubleComplex alpha;
    cuDoubleComplex beta;

    // Corner elimination downward
    for (int i_blockrow = 1; i_blockrow < partition_blocksize; ++i_blockrow) {
	
        int colsSkip = (i_blockrow-1) * l_dim * blocksize;
        int rowOffset = (i_blockrow-1) * blocksize;
        
        cudaErrchk(cudaStreamSynchronize(stream));

        extract_subblock_from_GPU(diag_buff_d, A_gpu, blocksize, l_dim, i_blockrow-1, i_blockrow-1);

        invert_GPU_matrix(diag_buff_d, identity_cpy_d, blocksize, cusolver_handle, buffer, info_d, info_h, ipiv_d);

        // Computation of Block L_(i+1)_i = A_(i+1)_i * (A_i_i)^(-1)
        alpha = make_cuDoubleComplex(1.0,0.0);
        beta = make_cuDoubleComplex(0.0,0.0);
        cublasErrchk(cublasZgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    blocksize, blocksize, blocksize,
                    &alpha,
                    A_gpu+colsSkip+rowOffset+blocksize, l_dim,
                    identity_cpy_d, blocksize,
                    &beta,
                    L_gpu+colsSkip+rowOffset+blocksize, l_dim));

        // Computation of Block U_i_(i+1) = (A_i_i)^(-1) * A_i_(i+1)
        cublasErrchk(cublasZgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    blocksize, blocksize, blocksize,
                    &alpha,
                    identity_cpy_d, blocksize,
                    A_gpu+colsSkip+rowOffset+l_dim*blocksize, l_dim,
                    &beta,
                    U_gpu+colsSkip+rowOffset+l_dim*blocksize, l_dim));

        // Computation of Block A_(i+1)_(i+1) -= L_(i+1)_i * A_i_(i+1)
        alpha = make_cuDoubleComplex(-1.0,0.0);
        beta = make_cuDoubleComplex(1.0,0.0);
        cublasErrchk(cublasZgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    blocksize, blocksize, blocksize,
                    &alpha,
                    L_gpu+colsSkip+rowOffset+blocksize, l_dim,
                    A_gpu+colsSkip+rowOffset+l_dim*blocksize, l_dim,
                    &beta,
                    A_gpu+colsSkip+rowOffset+l_dim*blocksize+blocksize, l_dim));

        // Reset identity_cpy_d to the identity Matrix of size blocksize
        cudaErrchk(cudaMemcpy(identity_cpy_d, identity_d, blocksize * blocksize * sizeof(std::complex<double>),  cudaMemcpyDeviceToDevice));

    }

    cudaErrchk(cudaStreamSynchronize(stream));

    if(identity_cpy_d) {
        cudaErrchk(cudaFree(identity_cpy_d));
    }
    if(diag_buff_d) {
        cudaErrchk(cudaFree(diag_buff_d));
    }
    if(buffer) {
	cudaErrchk(cudaFree(buffer));
    }
    if(ipiv_d) {
        cudaErrchk(cudaFree(ipiv_d));
    }
    if(info_d) {
        cudaErrchk(cudaFree(info_d));
    }
}


void reduce_schur_central_gpu(
    	int partition_blocksize,
    	int blocksize,
    	cudaStream_t stream,
    	cusolverDnHandle_t cusolver_handle,
    	cublasHandle_t cublas_handle,
    	cuDoubleComplex* A_gpu,
    	cuDoubleComplex* L_gpu,
    	cuDoubleComplex* U_gpu,
    	cuDoubleComplex* identity_d,
    	int l_dim
) {
    // init right hand side identity matrix on device for inversion 
    cuDoubleComplex* identity_cpy_d = NULL;
    cuDoubleComplex* diag_buff_d = NULL;

    cudaErrchk(cudaMalloc((void**)&identity_cpy_d, blocksize * blocksize * sizeof(cuDoubleComplex)));
    cudaErrchk(cudaMalloc((void**)&diag_buff_d, blocksize * blocksize * sizeof(cuDoubleComplex)));

    cudaErrchk(cudaMemcpy(identity_cpy_d, identity_d,
                blocksize * blocksize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

    // Allocate buffer needed for LU-Decompositions of the to-be inverted Matrices + buffers for pivots and info flags
    int info_h = 0;
    int *ipiv_d = NULL;
    int *info_d = NULL;
    cudaErrchk(cudaMalloc((void**)&info_d, sizeof(int)))
    cudaErrchk(cudaMalloc((void**)&ipiv_d, blocksize*sizeof(int)));

    cuDoubleComplex *buffer = NULL;
    int bufferSize = 0;
    cusolverErrchk(cusolverDnZgetrf_bufferSize(cusolver_handle, blocksize, blocksize,
                                               diag_buff_d, blocksize,
					       &bufferSize));
    cudaErrchk(cudaMalloc((void**)&buffer, sizeof(cuDoubleComplex) * bufferSize));

    cuDoubleComplex alpha;
    cuDoubleComplex beta;

    // Center elimination downward
    for (int i_blockrow = 2; i_blockrow < partition_blocksize; ++i_blockrow) {
	int colsSkip_i = i_blockrow * l_dim * blocksize;
	int rowOffset_i = i_blockrow * blocksize;
	int rowOffset_im1 = rowOffset_i - blocksize;
	int colsSkip_ip1 = colsSkip_i + l_dim * blocksize;

	int toprowOffset = 0;
	int topcolsSkip = l_dim * blocksize;

	cudaErrchk(cudaStreamSynchronize(stream));

	// Copy the diag block to be inversed into the buff, in this case it is A_im1_i
	// for(int j = 0; j < blocksize; ++j) {
	// 	cudaErrchk(cudaMemcpy(diag_buff_d+j*blocksize, A_gpu+colsSkip_i+rowOffset_im1+j*l_dim, blocksize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));	
	// }
    extract_subblock_from_GPU(diag_buff_d, A_gpu, blocksize, l_dim, i_blockrow-1, i_blockrow);

    invert_GPU_matrix(diag_buff_d, identity_cpy_d, blocksize, cusolver_handle, buffer, info_d, info_h, ipiv_d);

	// Computation of Block L_i_i = A_i_i * (A_im1_i)^(-1)
	alpha = make_cuDoubleComplex(1.0,0.0);
	beta = make_cuDoubleComplex(0.0,0.0);
	cublasErrchk(cublasZgemm(
	            cublas_handle,
	            CUBLAS_OP_N, CUBLAS_OP_N,
	            blocksize, blocksize, blocksize,
	            &alpha,
	            A_gpu+colsSkip_i+rowOffset_i, l_dim,
	            identity_cpy_d, blocksize,
	            &beta,
	            L_gpu+colsSkip_i+rowOffset_i, l_dim));

	// Computation of Block L_topR_i = A_topR_i * (A_im1_i)^(-1),  topR stands for the logical top blockrow which is 0 here
	cublasErrchk(cublasZgemm(
	            cublas_handle,
	            CUBLAS_OP_N, CUBLAS_OP_N,
	            blocksize, blocksize, blocksize,
	            &alpha,
	            A_gpu+colsSkip_i+toprowOffset, l_dim,
	            identity_cpy_d, blocksize,
	            &beta,
	            L_gpu+colsSkip_i+toprowOffset, l_dim));

	// Computation of Block U_im1_ip1 = (A_im1_i)^(-1) * A_im1_ip1
	cublasErrchk(cublasZgemm(
	            cublas_handle,
	            CUBLAS_OP_N, CUBLAS_OP_N,
	            blocksize, blocksize, blocksize,
	            &alpha,
	            identity_cpy_d, blocksize,
	            A_gpu+colsSkip_ip1+rowOffset_im1, l_dim,
	            &beta,
	            U_gpu+colsSkip_ip1+rowOffset_im1, l_dim));

	// Computation of Block U_im1_topC = (A_im1_i)^(-1) * A_im1_topC,  topC stands for the logical top blockcolumn which is l_dim * blocksize here
	cublasErrchk(cublasZgemm(
	            cublas_handle,
	            CUBLAS_OP_N, CUBLAS_OP_N,
	            blocksize, blocksize, blocksize,
	            &alpha,
	            identity_cpy_d, blocksize,
	            A_gpu+topcolsSkip+rowOffset_im1, l_dim,
	            &beta,
	            U_gpu+topcolsSkip+rowOffset_im1, l_dim));

	// Computation of Block A_i_ip1 -= L_i_i * A_im1_ip1, A_i_ip1 is the succesive diagonal block of the original matrix
	alpha = make_cuDoubleComplex(-1.0,0.0);
	beta = make_cuDoubleComplex(1.0,0.0);
	cublasErrchk(cublasZgemm(
	            cublas_handle,
	            CUBLAS_OP_N, CUBLAS_OP_N,
	            blocksize, blocksize, blocksize,
	            &alpha,
	            L_gpu+colsSkip_i+rowOffset_i, l_dim,
	            A_gpu+colsSkip_ip1+rowOffset_im1, l_dim,
	            &beta,
		    A_gpu+colsSkip_ip1+rowOffset_i, l_dim));

	// Computation of Block A_topR_topC -= L_topR_i * A_im1_topC
	cublasErrchk(cublasZgemm(
	            cublas_handle,
	            CUBLAS_OP_N, CUBLAS_OP_N,
	            blocksize, blocksize, blocksize,
	            &alpha,
	            L_gpu+colsSkip_i+toprowOffset, l_dim,
	            A_gpu+topcolsSkip+rowOffset_im1, l_dim,
	            &beta,
	            A_gpu+topcolsSkip+toprowOffset, l_dim));

	// Computation of Block A_i_topC -= L_i_i * A_im1_topC
	cublasErrchk(cublasZgemm(
	            cublas_handle,
	            CUBLAS_OP_N, CUBLAS_OP_N,
	            blocksize, blocksize, blocksize,
	            &alpha,
	            L_gpu+colsSkip_i+rowOffset_i, l_dim,
	            A_gpu+topcolsSkip+rowOffset_im1, l_dim,
	            &beta,
	            A_gpu+topcolsSkip+rowOffset_i, l_dim));

	// Computation of Block A_topR_ip1 -= L_topR_i * A_im1_ip1
	cublasErrchk(cublasZgemm(
	            cublas_handle,
	            CUBLAS_OP_N, CUBLAS_OP_N,
	            blocksize, blocksize, blocksize,
	            &alpha,
	            L_gpu+colsSkip_i+toprowOffset, l_dim,
	            A_gpu+colsSkip_ip1+rowOffset_im1, l_dim,
	            &beta,
	            A_gpu+colsSkip_ip1+toprowOffset, l_dim));


/*
	// Testing correctness
	Eigen::MatrixXcd A_inv_gpu = Eigen::MatrixXcd::Zero(blocksize, blocksize);

	cudaErrchk(cudaMemcpy(A_inv_gpu.data(), reinterpret_cast<std::complex<double>*>(identity_cpy_d), blocksize * blocksize * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

	if (A_inv_gpu.isApprox(A_inv_im1_im1)) {
		std::cout << "Inv success\n";
	} else {
		std::cout << "Inv fail\n";
	}

	Eigen::MatrixXcd A_host = Eigen::MatrixXcd::Zero(A.rows(), A.cols());

	cudaErrchk(cudaMemcpy(A_host.data(), reinterpret_cast<std::complex<double>*>(A_gpu), blocksize * blocksize * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

	if (A_inv_gpu.isApprox(A_inv_im1_im1)) {
		std::cout << "Inv success\n";
	} else {
		std::cout << "Inv fail\n";
	}
*/
	// Reset identity_cpy_d to the identity Matrix of size blocksize
	cudaErrchk(cudaMemcpy(identity_cpy_d, identity_d, blocksize * blocksize * sizeof(std::complex<double>),  cudaMemcpyDeviceToDevice));
    }

    cudaErrchk(cudaStreamSynchronize(stream));


    if(identity_cpy_d) {
        cudaErrchk(cudaFree(identity_cpy_d));
    }
    if(diag_buff_d) {
        cudaErrchk(cudaFree(diag_buff_d));
    }
    if(buffer) {
	cudaErrchk(cudaFree(buffer));
    }
    if(ipiv_d) {
        cudaErrchk(cudaFree(ipiv_d));
    }
    if(info_d) {
        cudaErrchk(cudaFree(info_d));
    }

}




void reduce_schur_bottomrightcorner_gpu(
    	int partition_blocksize,
    	int blocksize,
    	cudaStream_t stream,
    	cusolverDnHandle_t cusolver_handle,
    	cublasHandle_t cublas_handle,
    	cuDoubleComplex* A_gpu,
    	cuDoubleComplex* L_gpu,
    	cuDoubleComplex* U_gpu,
    	cuDoubleComplex* identity_d,
    	int l_dim
) {

    // init right hand side identity matrix on device for inversion 
    cuDoubleComplex* identity_cpy_d = NULL;
    cuDoubleComplex* diag_buff_d = NULL;

    cudaErrchk(cudaMalloc((void**)&identity_cpy_d, blocksize * blocksize * sizeof(cuDoubleComplex)));
    cudaErrchk(cudaMalloc((void**)&diag_buff_d, blocksize * blocksize * sizeof(cuDoubleComplex)));

    cudaErrchk(cudaMemcpy(identity_cpy_d, identity_d,
                blocksize * blocksize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

    // Allocate buffer needed for LU-Decompositions of the to-be inverted Matrices + buffers for pivots and info flags
    int info_h = 0;
    int *ipiv_d = NULL;
    int *info_d = NULL;
    cudaErrchk(cudaMalloc((void**)&info_d, sizeof(int)))
    cudaErrchk(cudaMalloc((void**)&ipiv_d, blocksize*sizeof(int)));

    cuDoubleComplex *buffer = NULL;
    int bufferSize = 0;
    cusolverErrchk(cusolverDnZgetrf_bufferSize(cusolver_handle, blocksize, blocksize,
                                               diag_buff_d, blocksize,
					       &bufferSize));
    cudaErrchk(cudaMalloc((void**)&buffer, sizeof(cuDoubleComplex) * bufferSize));

    cuDoubleComplex alpha;
    cuDoubleComplex beta;

    // Corner elimination upward
    for (int i_blockrow = partition_blocksize - 1; i_blockrow >= 1; --i_blockrow) {
	int colsSkip_i = i_blockrow * l_dim * blocksize;
	int rowOffset_i = i_blockrow * blocksize;
	int colsSkip_ip1 = colsSkip_i + l_dim * blocksize;
	int rowOffset_im1 = rowOffset_i - blocksize;

	cudaErrchk(cudaStreamSynchronize(stream));

	// // Copy the diag block to be inversed into the buff, in this case it is A_i_ip1
	// for(int j = 0; j < blocksize; ++j) {
	// 	cudaErrchk(cudaMemcpy(diag_buff_d+j*blocksize, A_gpu+colsSkip_ip1+rowOffset_i+j*l_dim, blocksize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));	
	// }
    extract_subblock_from_GPU(diag_buff_d, A_gpu, blocksize, l_dim, i_blockrow, i_blockrow + 1);

	// // Inversion of Block A_i_ip1 into buffer identity_cpy_d
	// // This block is original a diagonal block of the original matrix
	// cusolverErrchk(cusolverDnZgetrf(cusolver_handle, blocksize, blocksize,
	// 				diag_buff_d, blocksize, 
	// 				buffer, ipiv_d, info_d));

	// cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));

	// if (info_h != 0) {
	// 	std::cout << "Error: LU factorization failed" << std::endl;
	// 	std::cout << "info_h = " << info_h << std::endl;
	// }

	// cusolverErrchk(cusolverDnZgetrs(cusolver_handle, CUBLAS_OP_N, blocksize, blocksize,
	// 				diag_buff_d, blocksize, ipiv_d,
	// 				identity_cpy_d, blocksize, info_d));

	// cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));

	// if (info_h != 0) {
	// 	std::printf("Error: Inversion failed\n");
	// 	std::printf("info_h = %d\n", info_h);
    //     } 

    invert_GPU_matrix(diag_buff_d, identity_cpy_d, blocksize, cusolver_handle, buffer, info_d, info_h, ipiv_d);

	// Computation of Block L_im1_ip1 = A_im1_ip1 * (A_i_ip1)^(-1)
	alpha = make_cuDoubleComplex(1.0,0.0);
	beta = make_cuDoubleComplex(0.0,0.0);
	cublasErrchk(cublasZgemm(
	            cublas_handle,
	            CUBLAS_OP_N, CUBLAS_OP_N,
	            blocksize, blocksize, blocksize,
	            &alpha,
	            A_gpu+colsSkip_ip1+rowOffset_im1, l_dim,
	            identity_cpy_d, blocksize,
	            &beta,
	            L_gpu+colsSkip_ip1+rowOffset_im1, l_dim));

	// Computation of Block U_i_i = (A_i_ip1)^(-1) * A_i_i
	cublasErrchk(cublasZgemm(
	            cublas_handle,
	            CUBLAS_OP_N, CUBLAS_OP_N,
	            blocksize, blocksize, blocksize,
	            &alpha,
	            identity_cpy_d, blocksize,
	            A_gpu+colsSkip_i+rowOffset_i, l_dim,
	            &beta,
	            U_gpu+colsSkip_i+rowOffset_i, l_dim));

	// Computation of Block A_im1_i -= L_im1_ip1 * A_i_i
	alpha = make_cuDoubleComplex(-1.0,0.0);
	beta = make_cuDoubleComplex(1.0,0.0);
	cublasErrchk(cublasZgemm(
	            cublas_handle,
	            CUBLAS_OP_N, CUBLAS_OP_N,
	            blocksize, blocksize, blocksize,
	            &alpha,
	            L_gpu+colsSkip_ip1+rowOffset_im1, l_dim,
	            A_gpu+colsSkip_i+rowOffset_i, l_dim,
	            &beta,
	            A_gpu+colsSkip_i+rowOffset_im1, l_dim));


	// Reset identity_cpy_d to the identity Matrix of size blocksize
	cudaErrchk(cudaMemcpy(identity_cpy_d, identity_d, blocksize * blocksize * sizeof(std::complex<double>),  cudaMemcpyDeviceToDevice));
    }

    cudaErrchk(cudaStreamSynchronize(stream));

    if(identity_cpy_d) {
        cudaErrchk(cudaFree(identity_cpy_d));
    }
    if(diag_buff_d) {
        cudaErrchk(cudaFree(diag_buff_d));
    }
    if(buffer) {
	cudaErrchk(cudaFree(buffer));
    }
    if(ipiv_d) {
        cudaErrchk(cudaFree(ipiv_d));
    }
    if(info_d) {
        cudaErrchk(cudaFree(info_d));
    }

}

void produceSchurTopLeftCorner_gpu(
	int partition_blocksize,
   	int blocksize,
   	cudaStream_t stream,
	cusolverDnHandle_t cusolver_handle,
    	cublasHandle_t cublas_handle,
        cuDoubleComplex* A_gpu,
    	cuDoubleComplex* G_gpu,
    	cuDoubleComplex* L_gpu,
    	cuDoubleComplex* U_gpu,
    	cuDoubleComplex* identity_d,
    	int l_dim
) {

    // init right hand side identity matrix on device for inversion 
    cuDoubleComplex* identity_cpy_d = NULL;
    cuDoubleComplex* diag_buff_d = NULL;

    cudaErrchk(cudaMalloc((void**)&identity_cpy_d, blocksize * blocksize * sizeof(cuDoubleComplex)));
    cudaErrchk(cudaMalloc((void**)&diag_buff_d, blocksize * blocksize * sizeof(cuDoubleComplex)));

    cudaErrchk(cudaMemcpy(identity_cpy_d, identity_d,
                blocksize * blocksize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

    

    // Allocate buffer needed for LU-Decompositions of the to-be inverted Matrices + buffers for pivots and info flags
    int info_h = 0;
    int *ipiv_d = NULL;
    int *info_d = NULL;
    cudaErrchk(cudaMalloc((void**)&info_d, sizeof(int)))
    cudaErrchk(cudaMalloc((void**)&ipiv_d, blocksize*sizeof(int)));

    cuDoubleComplex *buffer = NULL;
    int bufferSize = 0;
    cusolverErrchk(cusolverDnZgetrf_bufferSize(cusolver_handle, blocksize, blocksize,
                                               diag_buff_d, blocksize,
					       &bufferSize));
    cudaErrchk(cudaMalloc((void**)&buffer, sizeof(cuDoubleComplex) * bufferSize));

    cuDoubleComplex alpha;
    cuDoubleComplex beta;

    // Upper left corner produced upwards
    for (int i_blockrow = partition_blocksize - 1; i_blockrow > 0; --i_blockrow) {
	
        int colsSkip = (i_blockrow-1) * l_dim * blocksize;
        int colsSkip_plus = i_blockrow * l_dim * blocksize;
        int rowOffset = (i_blockrow-1) * blocksize;
        int rowOffset_plus = i_blockrow * blocksize;
        
        cudaErrchk(cudaStreamSynchronize(stream));

        extract_subblock_from_GPU(diag_buff_d, A_gpu, blocksize, l_dim, i_blockrow-1, i_blockrow-1);

        invert_GPU_matrix(diag_buff_d, identity_cpy_d, blocksize, cusolver_handle, buffer, info_d, info_h, ipiv_d);

        // Computation of Block G_i_(i-1) = - G_(i)_(i) * L_i_(i-1)
        alpha = make_cuDoubleComplex(-1.0,0.0);
        beta = make_cuDoubleComplex(0.0,0.0);
        cublasErrchk(cublasZgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    blocksize, blocksize, blocksize,
                    &alpha,
                    G_gpu+colsSkip_plus+rowOffset_plus, l_dim,
                    L_gpu+colsSkip + rowOffset_plus, l_dim,
                    &beta,
                    G_gpu+colsSkip+rowOffset_plus, l_dim));

        // Computation of Block G_(i-1)_i = - U_(i-1)_(i) * G_(i)_(i) 
        cublasErrchk(cublasZgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    blocksize, blocksize, blocksize,
                    &alpha,
                    U_gpu+colsSkip_plus + rowOffset, l_dim,
                    G_gpu+colsSkip_plus+rowOffset_plus, l_dim,
                    &beta,
                    G_gpu+colsSkip_plus+rowOffset, l_dim));

        // Computation of Block G_(i-1)_(i-1) = (A_(i-1)_(i-1))^-1) -  U_(i-1)_(i) * G_i_(i-1)
        alpha = make_cuDoubleComplex(-1.0,0.0);
        beta = make_cuDoubleComplex(0.0,0.0);
        cublasErrchk(cublasZgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    blocksize, blocksize, blocksize,
                    &alpha,
                    U_gpu+colsSkip_plus + rowOffset, l_dim,
                    G_gpu+colsSkip+rowOffset_plus, l_dim,
                    &beta,
                    G_gpu+colsSkip+rowOffset, l_dim));

        alpha = make_cuDoubleComplex(1.0,0.0);
        beta = make_cuDoubleComplex(1.0,0.0);
        cublasErrchk(cublasZgeam(
                        cublas_handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        blocksize, blocksize,
                        &alpha,
                        identity_cpy_d, blocksize,
                        &beta,
                        G_gpu+colsSkip+rowOffset, l_dim,
                        G_gpu+colsSkip+rowOffset, l_dim));

        // Reset identity_cpy_d to the identity Matrix of size blocksize
        cudaErrchk(cudaMemcpy(identity_cpy_d, identity_d, blocksize * blocksize * sizeof(std::complex<double>),  cudaMemcpyDeviceToDevice));

    }

    cudaErrchk(cudaStreamSynchronize(stream));

    if(identity_cpy_d) {
        cudaErrchk(cudaFree(identity_cpy_d));
    }
    if(diag_buff_d) {
        cudaErrchk(cudaFree(diag_buff_d));
    }
    if(buffer) {
	cudaErrchk(cudaFree(buffer));
    }
    if(ipiv_d) {
        cudaErrchk(cudaFree(ipiv_d));
    }
    if(info_d) {
        cudaErrchk(cudaFree(info_d));
    }
}


void produceSchurBottomRightCorner_gpu(
	int partition_blocksize,
   	int blocksize,
   	cudaStream_t stream,
	cusolverDnHandle_t cusolver_handle,
    	cublasHandle_t cublas_handle,
        cuDoubleComplex* A_gpu,
    	cuDoubleComplex* G_gpu,
    	cuDoubleComplex* L_gpu,
    	cuDoubleComplex* U_gpu,
    	cuDoubleComplex* identity_d,
    	int l_dim
) {

    // init right hand side identity matrix on device for inversion 
    cuDoubleComplex* identity_cpy_d = NULL;
    cuDoubleComplex* diag_buff_d = NULL;

    cudaErrchk(cudaMalloc((void**)&identity_cpy_d, blocksize * blocksize * sizeof(cuDoubleComplex)));
    cudaErrchk(cudaMalloc((void**)&diag_buff_d, blocksize * blocksize * sizeof(cuDoubleComplex)));

    cudaErrchk(cudaMemcpy(identity_cpy_d, identity_d,
                blocksize * blocksize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

    

    // Allocate buffer needed for LU-Decompositions of the to-be inverted Matrices + buffers for pivots and info flags
    int info_h = 0;
    int *ipiv_d = NULL;
    int *info_d = NULL;
    cudaErrchk(cudaMalloc((void**)&info_d, sizeof(int)))
    cudaErrchk(cudaMalloc((void**)&ipiv_d, blocksize*sizeof(int)));

    cuDoubleComplex *buffer = NULL;
    int bufferSize = 0;
    cusolverErrchk(cusolverDnZgetrf_bufferSize(cusolver_handle, blocksize, blocksize,
                                               diag_buff_d, blocksize,
					       &bufferSize));
    cudaErrchk(cudaMalloc((void**)&buffer, sizeof(cuDoubleComplex) * bufferSize));

    cuDoubleComplex alpha;
    cuDoubleComplex beta;

    // Upper left corner produced upwards
    for (int i_blockrow = 0; i_blockrow < partition_blocksize - 1; ++i_blockrow) {
	
        int colsSkip = (i_blockrow + 1) * l_dim * blocksize;
        int colsSkip_plus = (i_blockrow + 2) * l_dim * blocksize;
        int rowOffset = (i_blockrow) * blocksize;
        int rowOffset_plus = (i_blockrow + 1) * blocksize;
        
        cudaErrchk(cudaStreamSynchronize(stream));

        extract_subblock_from_GPU(diag_buff_d, A_gpu, blocksize, l_dim, i_blockrow + 1, i_blockrow + 2);

        invert_GPU_matrix(diag_buff_d, identity_cpy_d, blocksize, cusolver_handle, buffer, info_d, info_h, ipiv_d);

        // Computation of Block G_i_(i+1) = - G_(i)_(i) * L_i_(i+1)
        alpha = make_cuDoubleComplex(-1.0,0.0);
        beta = make_cuDoubleComplex(0.0,0.0);
        cublasErrchk(cublasZgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    blocksize, blocksize, blocksize,
                    &alpha,
                    G_gpu+ colsSkip+ rowOffset, l_dim,
                    L_gpu+ colsSkip_plus + rowOffset, l_dim,
                    &beta,
                    G_gpu+colsSkip_plus+rowOffset, l_dim));

        // Computation of Block G_(i+1)_i = - U_(i+1)_(i) * G_(i)_(i) 
        cublasErrchk(cublasZgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    blocksize, blocksize, blocksize,
                    &alpha,
                    U_gpu+colsSkip + rowOffset_plus, l_dim,
                    G_gpu+colsSkip+rowOffset, l_dim,
                    &beta,
                    G_gpu+colsSkip+rowOffset_plus, l_dim));

        // Computation of Block G_(i+1)_(i+1) = (A_(i+1)_(i+1))^-1) -  U_(i+1)_(i) * G_i_(i+1)
        alpha = make_cuDoubleComplex(-1.0,0.0);
        beta = make_cuDoubleComplex(0.0,0.0);
        cublasErrchk(cublasZgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    blocksize, blocksize, blocksize,
                    &alpha,
                    U_gpu+colsSkip + rowOffset_plus, l_dim,
                    G_gpu+colsSkip_plus+rowOffset, l_dim,
                    &beta,
                    G_gpu+colsSkip_plus+rowOffset_plus, l_dim));

        alpha = make_cuDoubleComplex(1.0,0.0);
        beta = make_cuDoubleComplex(1.0,0.0);
        cublasErrchk(cublasZgeam(
                        cublas_handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        blocksize, blocksize,
                        &alpha,
                        identity_cpy_d, blocksize,
                        &beta,
                        G_gpu+colsSkip_plus+rowOffset_plus, l_dim,
                        G_gpu+colsSkip_plus+rowOffset_plus, l_dim));

        // Reset identity_cpy_d to the identity Matrix of size blocksize
        cudaErrchk(cudaMemcpy(identity_cpy_d, identity_d, blocksize * blocksize * sizeof(std::complex<double>),  cudaMemcpyDeviceToDevice));

    }

    cudaErrchk(cudaStreamSynchronize(stream));

    if(identity_cpy_d) {
        cudaErrchk(cudaFree(identity_cpy_d));
    }
    if(diag_buff_d) {
        cudaErrchk(cudaFree(diag_buff_d));
    }
    if(buffer) {
	cudaErrchk(cudaFree(buffer));
    }
    if(ipiv_d) {
        cudaErrchk(cudaFree(ipiv_d));
    }
    if(info_d) {
        cudaErrchk(cudaFree(info_d));
    }
}



void produceSchurcentral_gpu(
	int partition_blocksize,
   	int blocksize,
   	cudaStream_t stream,
	cusolverDnHandle_t cusolver_handle,
    	cublasHandle_t cublas_handle,
        cuDoubleComplex* A_gpu,
    	cuDoubleComplex* G_gpu,
    	cuDoubleComplex* L_gpu,
    	cuDoubleComplex* U_gpu,
    	cuDoubleComplex* identity_d,
    	int l_dim
) {

    // init right hand side identity matrix on device for inversion 
    cuDoubleComplex* identity_cpy_d = NULL;
    cuDoubleComplex* diag_buff_d = NULL;

    cudaErrchk(cudaMalloc((void**)&identity_cpy_d, blocksize * blocksize * sizeof(cuDoubleComplex)));
    cudaErrchk(cudaMalloc((void**)&diag_buff_d, blocksize * blocksize * sizeof(cuDoubleComplex)));

    cudaErrchk(cudaMemcpy(identity_cpy_d, identity_d,
                blocksize * blocksize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

    

    // Allocate buffer needed for LU-Decompositions of the to-be inverted Matrices + buffers for pivots and info flags
    int info_h = 0;
    int *ipiv_d = NULL;
    int *info_d = NULL;
    cudaErrchk(cudaMalloc((void**)&info_d, sizeof(int)))
    cudaErrchk(cudaMalloc((void**)&ipiv_d, blocksize*sizeof(int)));

    cuDoubleComplex *buffer = NULL;
    int bufferSize = 0;
    cusolverErrchk(cusolverDnZgetrf_bufferSize(cusolver_handle, blocksize, blocksize,
                                               diag_buff_d, blocksize,
					       &bufferSize));
    cudaErrchk(cudaMalloc((void**)&buffer, sizeof(cuDoubleComplex) * bufferSize));

    int colsSkipbtm_minus = (partition_blocksize - 1) * l_dim * blocksize;
    int colsSkipbtm = (partition_blocksize) * l_dim * blocksize;
    int rowOffsetbtm_minus = (partition_blocksize - 2) * blocksize;
    int rowOffsetbtm = (partition_blocksize -1) * blocksize;

    int colsSkiptop = l_dim * blocksize;
    int colsSkiptop_plus = 2 * l_dim * blocksize;
    int colsSkiptop_plusplus = 3 * l_dim * blocksize;

    int rowOffsettop = 0;
    int rowOffsettop_plus = blocksize;
    int rowOffsettop_plusplus = 2 * blocksize;


    cuDoubleComplex alpha;
    cuDoubleComplex beta;
    // Computation of Block G_(n)_(n-1) = - G_(n)_(0) * L_0_(n-1) + G_(n)_(n) * L_n_(n-1)
    alpha = make_cuDoubleComplex(-1.0,0.0);
    beta = make_cuDoubleComplex(0.0,0.0);
    cublasErrchk(cublasZgemm(
                cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                blocksize, blocksize, blocksize,
                &alpha,
                G_gpu + colsSkiptop + rowOffsetbtm, l_dim,
                L_gpu + rowOffsettop + colsSkipbtm_minus, l_dim,
                &beta,
                G_gpu+colsSkipbtm_minus+rowOffsetbtm, l_dim));

    alpha = make_cuDoubleComplex(-1.0,0.0);
    beta = make_cuDoubleComplex(1.0,0.0);
    cublasErrchk(cublasZgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    blocksize, blocksize, blocksize,
                    &alpha,
                    G_gpu+colsSkipbtm+rowOffsetbtm, l_dim,
                    L_gpu+colsSkipbtm_minus+rowOffsetbtm, l_dim,
                    &beta,
                    G_gpu+colsSkipbtm_minus+rowOffsetbtm, l_dim));


    // Computation of Block G_(n-1)_(n) = - U_(n_1)_(n) * G_n_(n) + U_(n-1)_(0) * G_0_(n)
    alpha = make_cuDoubleComplex(-1.0,0.0);
    beta = make_cuDoubleComplex(0.0,0.0);
    cublasErrchk(cublasZgemm(
                cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                blocksize, blocksize, blocksize,
                &alpha,
                U_gpu + colsSkipbtm + rowOffsetbtm_minus, l_dim,
                G_gpu + colsSkipbtm + rowOffsetbtm, l_dim,
                &beta,
                G_gpu+colsSkipbtm+rowOffsetbtm_minus, l_dim));
    
    alpha = make_cuDoubleComplex(-1.0,0.0);
    beta = make_cuDoubleComplex(1.0,0.0);
    cublasErrchk(cublasZgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    blocksize, blocksize, blocksize,
                    &alpha,
                    U_gpu + colsSkiptop +rowOffsetbtm_minus, l_dim,
                    G_gpu + colsSkipbtm +rowOffsettop, l_dim,
                    &beta,
                    G_gpu+colsSkipbtm+rowOffsetbtm_minus, l_dim));

    

    // Upper left corner produced upwards
    for (int i_blockrow = partition_blocksize - 2; i_blockrow > 0; --i_blockrow) {
	
        int colsSkip = (i_blockrow + 1) * l_dim * blocksize;
        int colsSkip_plus = (i_blockrow + 2) * l_dim * blocksize;
        int rowOffset = (i_blockrow) * blocksize;
        int rowOffset_plus = (i_blockrow + 1) * blocksize;
        
        cudaErrchk(cudaStreamSynchronize(stream));

        // Computation of Block G_0_(i) = - G_(0)_(0) * L_(0)_(i) + G_(0)_(i + 1) * L_(i+1)_(i)
        alpha = make_cuDoubleComplex(-1.0,0.0);
        beta = make_cuDoubleComplex(0.0,0.0);
        cublasErrchk(cublasZgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    blocksize, blocksize, blocksize,
                    &alpha,
                    G_gpu+ colsSkiptop + rowOffsettop, l_dim,
                    L_gpu+ colsSkip + rowOffsettop, l_dim,
                    &beta,
                    G_gpu + rowOffsettop + colsSkip, l_dim));
        alpha = make_cuDoubleComplex(-1.0,0.0);
        beta = make_cuDoubleComplex(1.0,0.0);
        cublasErrchk(cublasZgemm(
                        cublas_handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        blocksize, blocksize, blocksize,
                        &alpha,
                        G_gpu+ colsSkip_plus + rowOffsettop, l_dim,
                        L_gpu+ colsSkip + rowOffset_plus, l_dim,
                        &beta,
                        G_gpu + rowOffsettop + colsSkip, l_dim));

        // Computation bottom_blockrowof Block G_(i)_0 = - U_(i)_(i+1) * G_(i+1)_(0)  + U_(i)_(0) * G_(0)_(0)
        cublasErrchk(cublasZgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    blocksize, blocksize, blocksize,
                    &alpha,
                    U_gpu+colsSkip_plus + rowOffset, l_dim,
                    G_gpu+colsSkiptop+rowOffset_plus, l_dim,
                    &beta,
                    G_gpu+colsSkiptop+rowOffset, l_dim));
        alpha = make_cuDoubleComplex(-1.0,0.0);
        beta = make_cuDoubleComplex(1.0,0.0);
        cublasErrchk(cublasZgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    blocksize, blocksize, blocksize,
                    &alpha,
                    U_gpu+colsSkiptop + rowOffset, l_dim,
                    G_gpu+colsSkiptop+rowOffsettop, l_dim,
                    &beta,
                    G_gpu+colsSkiptop+rowOffset, l_dim));
    }

    for (int i_blockrow = partition_blocksize - 2; i_blockrow > 1; --i_blockrow){
        int colsSkip_minus = (i_blockrow) * l_dim * blocksize;
        int colsSkip = (i_blockrow + 1) * l_dim * blocksize;
        int colsSkip_plus = (i_blockrow + 2) * l_dim * blocksize;
        int rowOffset_minus = (i_blockrow - 1) * blocksize;
        int rowOffset = (i_blockrow) * blocksize;
        int rowOffset_plus = (i_blockrow + 1) * blocksize;

        cudaErrchk(cudaStreamSynchronize(stream));

        extract_subblock_from_GPU(diag_buff_d, A_gpu, blocksize, l_dim, i_blockrow, i_blockrow + 1);
        invert_GPU_matrix(diag_buff_d, identity_cpy_d, blocksize, cusolver_handle, buffer, info_d, info_h, ipiv_d);
        //Computation of Block G_(i)_(i) = (A_(i)_(i))^-1) -  U_(i)_(0) * G_0_(i) + U(i)_(i+1) * G_(i+1)_(i)
        alpha = make_cuDoubleComplex(-1.0,0.0);
        beta = make_cuDoubleComplex(0.0,0.0);
        cublasErrchk(cublasZgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    blocksize, blocksize, blocksize,
                    &alpha,
                    U_gpu+colsSkiptop + rowOffset, l_dim,
                    G_gpu+colsSkip+rowOffsettop, l_dim,
                    &beta,
                    G_gpu+colsSkip+rowOffset, l_dim));
        alpha = make_cuDoubleComplex(-1.0,0.0);
        beta = make_cuDoubleComplex(1.0,0.0);
        cublasErrchk(cublasZgemm(
                        cublas_handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        blocksize, blocksize, blocksize,
                        &alpha,
                        U_gpu+colsSkip_plus + rowOffset, l_dim,
                        G_gpu+colsSkip + rowOffset_plus, l_dim,
                        &beta,
                        G_gpu+colsSkip + rowOffset, l_dim));
        alpha = make_cuDoubleComplex(1.0,0.0);
        beta = make_cuDoubleComplex(1.0,0.0);
        cublasErrchk(cublasZgeam(
                        cublas_handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        blocksize, blocksize,
                        &alpha,
                        identity_cpy_d, blocksize,
                        &beta,
                        G_gpu+colsSkip+rowOffset, l_dim,
                        G_gpu+colsSkip+rowOffset, l_dim));

        //Computation of Block G_(i-1)_(i) = - U_(i-1)_(0) * G_(0)_(i) + U_(i-1)_(i) * G_(i)_(i)
        alpha = make_cuDoubleComplex(-1.0,0.0);
        beta = make_cuDoubleComplex(0.0,0.0);
        cublasErrchk(cublasZgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    blocksize, blocksize, blocksize,
                    &alpha,
                    U_gpu+colsSkiptop + rowOffset_minus, l_dim,
                    G_gpu+colsSkip+rowOffsettop, l_dim,
                    &beta,
                    G_gpu+colsSkip+rowOffset_minus, l_dim));
        alpha = make_cuDoubleComplex(-1.0,0.0);
        beta = make_cuDoubleComplex(1.0,0.0);
        cublasErrchk(cublasZgemm(
                        cublas_handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        blocksize, blocksize, blocksize,
                        &alpha,
                        U_gpu+colsSkip + rowOffset_minus, l_dim,
                        G_gpu+colsSkip+rowOffset, l_dim,
                        &beta,
                        G_gpu+colsSkip+rowOffset_minus, l_dim));
        
        // Computation of BLock G_(i)_(i-1) = - G_(i)_(0) * L_(0)_(i-1) + G_(i)_(i) * L_(i)_(i-1)
        alpha = make_cuDoubleComplex(-1.0,0.0);
        beta = make_cuDoubleComplex(0.0,0.0);
        cublasErrchk(cublasZgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    blocksize, blocksize, blocksize,
                    &alpha,
                    G_gpu+ colsSkiptop + rowOffset, l_dim,
                    L_gpu+ colsSkip_minus + rowOffsettop, l_dim,
                    &beta,
                    G_gpu + rowOffset + colsSkip_minus, l_dim));
        alpha = make_cuDoubleComplex(-1.0,0.0);
        beta = make_cuDoubleComplex(1.0,0.0);
        cublasErrchk(cublasZgemm(
                        cublas_handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        blocksize, blocksize, blocksize,
                        &alpha,
                        G_gpu+ colsSkip + rowOffset, l_dim,
                        L_gpu+ colsSkip_minus + rowOffset, l_dim,
                        &beta,
                        G_gpu + rowOffset + colsSkip_minus, l_dim));



        // Reset identity_cpy_d to the identity Matrix of size blocksize
        cudaErrchk(cudaMemcpy(identity_cpy_d, identity_d, blocksize * blocksize * sizeof(std::complex<double>),  cudaMemcpyDeviceToDevice));

    }
    extract_subblock_from_GPU(diag_buff_d, A_gpu, blocksize, l_dim, 1, 2);
    invert_GPU_matrix(diag_buff_d, identity_cpy_d, blocksize, cusolver_handle, buffer, info_d, info_h, ipiv_d);
    //Computation of Block G_(1)_(1) = (A_(1)_(1))^-1) -  U_(1)_(0) * G_(0)_(1) + U_(1)_(2) * G_(2)_(1)
    alpha = make_cuDoubleComplex(-1.0,0.0);
    beta = make_cuDoubleComplex(0.0,0.0);
    cublasErrchk(cublasZgemm(
                cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                blocksize, blocksize, blocksize,
                &alpha,
                U_gpu+colsSkiptop + rowOffsettop_plus, l_dim,
                G_gpu+colsSkiptop_plus +rowOffsettop, l_dim,
                &beta,
                G_gpu+colsSkiptop_plus+rowOffsettop_plus, l_dim));
    alpha = make_cuDoubleComplex(-1.0,0.0);
    beta = make_cuDoubleComplex(1.0,0.0);
    cublasErrchk(cublasZgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    blocksize, blocksize, blocksize,
                    &alpha,
                    U_gpu+colsSkiptop_plusplus + rowOffsettop_plus, l_dim,
                    G_gpu+colsSkiptop_plus + rowOffsettop_plusplus, l_dim,
                    &beta,
                    G_gpu+colsSkiptop_plus + rowOffsettop_plus, l_dim));
    alpha = make_cuDoubleComplex(1.0,0.0);
    beta = make_cuDoubleComplex(1.0,0.0);
    cublasErrchk(cublasZgeam(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    blocksize, blocksize,
                    &alpha,
                    identity_cpy_d, blocksize,
                    &beta,
                    G_gpu+colsSkiptop_plus+rowOffsettop_plus, l_dim,
                    G_gpu+colsSkiptop_plus+rowOffsettop_plus, l_dim));



    // Reset identity_cpy_d to the identity Matrix of size blocksize
    //cudaErrchk(cudaMemcpy(identity_cpy_d, identity_d, blocksize * blocksize * sizeof(std::complex<double>),  cudaMemcpyDeviceToDevice));

    cudaErrchk(cudaStreamSynchronize(stream));

    if(identity_cpy_d) {
        cudaErrchk(cudaFree(identity_cpy_d));
    }
    if(diag_buff_d) {
        cudaErrchk(cudaFree(diag_buff_d));
    }
    if(buffer) {
	cudaErrchk(cudaFree(buffer));
    }
    if(ipiv_d) {
        cudaErrchk(cudaFree(ipiv_d));
    }
    if(info_d) {
        cudaErrchk(cudaFree(info_d));
    }
}


// End of additions for cuda impl





void reduce_schur_sequentially(Eigen::MatrixXcd** eigenA,
                             Eigen::MatrixXcd** G_matrices,
                             Eigen::MatrixXcd** L_matrices,
                             Eigen::MatrixXcd** U_matrices,
                             int partitions,
                             int partition_blocksize,
                             int blocksize,
                             int rank) {
   // Generate the G, L and U matrices for each process
    for (int i = 0; i < partitions; ++i) {
        int start_blockrow = i * partition_blocksize;
        
        G_matrices[i] = new Eigen::MatrixXcd(eigenA[i]->rows(), eigenA[i]->cols());
        L_matrices[i] = new Eigen::MatrixXcd(eigenA[i]->rows(), eigenA[i]->cols());
        U_matrices[i] = new Eigen::MatrixXcd(eigenA[i]->rows(), eigenA[i]->cols());

        G_matrices[i]->setZero();
        L_matrices[i]->setZero();
        U_matrices[i]->setZero();

        std::cout << "Process " << rank << " is reducing blockrows " << start_blockrow << " to " << start_blockrow + partition_blocksize - 1 << std::endl;
        

        if (i == 0){
            auto result = reduce_schur_topleftcorner(*eigenA[i], start_blockrow, partition_blocksize, blocksize);
            *L_matrices[i] = std::get<0>(result);
            *U_matrices[i] = std::get<1>(result);
        }

        if (i > 0 && i < partitions - 1){
            auto result = reduce_schur_central(*eigenA[i], start_blockrow, partition_blocksize, blocksize);
            *L_matrices[i] = std::get<0>(result);
            *U_matrices[i] = std::get<1>(result);
        }

        if (i == partitions - 1){
            auto result = reduce_schur_bottomrightcorner(*eigenA[i], start_blockrow, partition_blocksize, blocksize);
            *L_matrices[i] = std::get<0>(result);
            *U_matrices[i] = std::get<1>(result);
        }

    }
}


std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd> reduce_schur_topleftcorner(
    Eigen::MatrixXcd& A,
    int start_blockrow,
    int partition_blocksize,
    int blocksize
) {
    Eigen::MatrixXcd L = Eigen::MatrixXcd::Zero(A.rows(), A.cols());
    Eigen::MatrixXcd U = Eigen::MatrixXcd::Zero(A.rows(), A.cols());

    // Corner elimination downward
    for (int i_blockrow = start_blockrow + 1; i_blockrow < start_blockrow + partition_blocksize; ++i_blockrow) {
        int im1_rowindice = (i_blockrow - 1) * blocksize;
        int i_rowindice = i_blockrow * blocksize;
        

        Eigen::MatrixXcd A_inv_im1_im1 = A.block(im1_rowindice, im1_rowindice, blocksize, blocksize).inverse();

        L.block(i_rowindice, im1_rowindice, blocksize, blocksize) =
            A.block(i_rowindice, im1_rowindice, blocksize, blocksize) * A_inv_im1_im1;

        U.block(im1_rowindice, i_rowindice, blocksize, blocksize) =
            A_inv_im1_im1 * A.block(im1_rowindice, i_rowindice, blocksize, blocksize);

        A.block(i_rowindice, i_rowindice, blocksize, blocksize) -=
            L.block(i_rowindice, im1_rowindice, blocksize, blocksize) *
            A.block(im1_rowindice, i_rowindice, blocksize, blocksize);
    }

    return std::make_tuple(L, U);
}


std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd> reduce_schur_bottomrightcorner(
    Eigen::MatrixXcd& A,
    int start_blockrow,
    int partition_blocksize,
    int blocksize
) {
    Eigen::MatrixXcd L = Eigen::MatrixXcd::Zero(A.rows(), A.cols());
    Eigen::MatrixXcd U = Eigen::MatrixXcd::Zero(A.rows(), A.cols());

    // Corner elimination upward
    for (int i_blockrow = start_blockrow + partition_blocksize - 2; i_blockrow >= start_blockrow; --i_blockrow) {
        int i_rowindice = i_blockrow * blocksize;
        int ip1_rowindice = (i_blockrow + 1) * blocksize;

        Eigen::MatrixXcd A_inv_ip1_ip1 = A.block(ip1_rowindice, ip1_rowindice, blocksize, blocksize).inverse();

        L.block(i_rowindice, ip1_rowindice, blocksize, blocksize) =
            A.block(i_rowindice, ip1_rowindice, blocksize, blocksize) * A_inv_ip1_ip1;

        U.block(ip1_rowindice, i_rowindice, blocksize, blocksize) =
            A_inv_ip1_ip1 * A.block(ip1_rowindice, i_rowindice, blocksize, blocksize);

        A.block(i_rowindice, i_rowindice, blocksize, blocksize) -=
            L.block(i_rowindice, ip1_rowindice, blocksize, blocksize) *
            A.block(ip1_rowindice, i_rowindice, blocksize, blocksize);
    }

    return std::make_tuple(L, U);
}

std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd> reduce_schur_bottomrightcorner_2(
    Eigen::MatrixXcd& A,
    int partition_blocksize,
    int blocksize
) {
    Eigen::MatrixXcd L = Eigen::MatrixXcd::Zero(A.rows(), A.cols());
    Eigen::MatrixXcd U = Eigen::MatrixXcd::Zero(A.rows(), A.cols());

    // Corner elimination upward
    for (int i_blockrow = partition_blocksize - 1; i_blockrow >= 1; --i_blockrow) {
        int i_rowindice = i_blockrow * blocksize;
	int il1_rowindice = i_rowindice - blocksize;    
        int ip1_rowindice = i_rowindice + blocksize;

        Eigen::MatrixXcd A_inv_ip1_ip1 = A.block(i_rowindice, ip1_rowindice, blocksize, blocksize).inverse();

        L.block(il1_rowindice, ip1_rowindice, blocksize, blocksize) =
            A.block(il1_rowindice, ip1_rowindice, blocksize, blocksize) * A_inv_ip1_ip1;

        U.block(i_rowindice, i_rowindice, blocksize, blocksize) =
            A_inv_ip1_ip1 * A.block(i_rowindice, i_rowindice, blocksize, blocksize);

        A.block(il1_rowindice, i_rowindice, blocksize, blocksize) -=
            L.block(il1_rowindice, ip1_rowindice, blocksize, blocksize) *
            A.block(i_rowindice, i_rowindice, blocksize, blocksize);
    }

    return std::make_tuple(L, U);
}

std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd> reduce_schur_central(
    Eigen::MatrixXcd& A,
    int start_blockrow,
    int partition_blocksize,
    int blocksize
) {
    Eigen::MatrixXcd L = Eigen::MatrixXcd::Zero(A.rows(), A.cols());
    Eigen::MatrixXcd U = Eigen::MatrixXcd::Zero(A.rows(), A.cols());
    
    // Center elimination downward
    for (int i_blockrow = start_blockrow + 2; i_blockrow < start_blockrow + partition_blocksize; ++i_blockrow) {
        int im1_rowindice = (i_blockrow - 1) * blocksize;
        int i_rowindice = i_blockrow * blocksize;

        int top_rowindice = start_blockrow * blocksize;

        Eigen::MatrixXcd A_inv_im1_im1 = A.block(im1_rowindice, im1_rowindice, blocksize, blocksize).inverse();

        L.block(i_rowindice, im1_rowindice, blocksize, blocksize) =
            A.block(i_rowindice, im1_rowindice, blocksize, blocksize) * A_inv_im1_im1;

        L.block(top_rowindice, im1_rowindice, blocksize, blocksize) =
            A.block(top_rowindice, im1_rowindice, blocksize, blocksize) * A_inv_im1_im1;

        U.block(im1_rowindice, i_rowindice, blocksize, blocksize) =
            A_inv_im1_im1 * A.block(im1_rowindice, i_rowindice, blocksize, blocksize);

        U.block(im1_rowindice, top_rowindice, blocksize, blocksize) =
            A_inv_im1_im1 * A.block(im1_rowindice, top_rowindice, blocksize, blocksize);

        A.block(i_rowindice, i_rowindice, blocksize, blocksize) -=
            L.block(i_rowindice, im1_rowindice, blocksize, blocksize) *
            A.block(im1_rowindice, i_rowindice, blocksize, blocksize);

        A.block(top_rowindice, top_rowindice, blocksize, blocksize) -=
            L.block(top_rowindice, im1_rowindice, blocksize, blocksize) *
            A.block(im1_rowindice, top_rowindice, blocksize, blocksize);

        A.block(i_rowindice, top_rowindice, blocksize, blocksize) -=
            L.block(i_rowindice, im1_rowindice, blocksize, blocksize) *
            A.block(im1_rowindice, top_rowindice, blocksize, blocksize);

        A.block(top_rowindice, i_rowindice, blocksize, blocksize) -=
            L.block(top_rowindice, im1_rowindice, blocksize, blocksize) *
            A.block(im1_rowindice, i_rowindice, blocksize, blocksize);
    }

    return std::make_tuple(L, U);
}


std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd> reduce_schur_central_2(
    Eigen::MatrixXcd& A,
    int partition_blocksize,
    int blocksize
) {
    Eigen::MatrixXcd L = Eigen::MatrixXcd::Zero(A.rows(), A.cols());
    Eigen::MatrixXcd U = Eigen::MatrixXcd::Zero(A.rows(), A.cols());

    // Center elimination downward
    for (int i_blockrow = 2; i_blockrow < partition_blocksize; ++i_blockrow) {
        int i_rowindice = i_blockrow * blocksize;
        int im1_rowindice = i_rowindice - blocksize;
	int ip1_rowindice = i_rowindice + blocksize;

        int top_rowindice = 0;
	int top_rowindice_col = blocksize;

        Eigen::MatrixXcd A_inv_im1_im1 = A.block(im1_rowindice, i_rowindice, blocksize, blocksize).inverse();

        L.block(i_rowindice, i_rowindice, blocksize, blocksize) =
            A.block(i_rowindice, i_rowindice, blocksize, blocksize) * A_inv_im1_im1;

        L.block(top_rowindice, i_rowindice, blocksize, blocksize) =
            A.block(top_rowindice, i_rowindice, blocksize, blocksize) * A_inv_im1_im1;

        U.block(im1_rowindice, ip1_rowindice, blocksize, blocksize) =
            A_inv_im1_im1 * A.block(im1_rowindice, ip1_rowindice, blocksize, blocksize);

        U.block(im1_rowindice, top_rowindice_col, blocksize, blocksize) =
            A_inv_im1_im1 * A.block(im1_rowindice, top_rowindice_col, blocksize, blocksize);

        A.block(i_rowindice, ip1_rowindice, blocksize, blocksize) -=
            L.block(i_rowindice, i_rowindice, blocksize, blocksize) *
            A.block(im1_rowindice, ip1_rowindice, blocksize, blocksize);

        A.block(top_rowindice, top_rowindice_col, blocksize, blocksize) -=
            L.block(top_rowindice, i_rowindice, blocksize, blocksize) *
            A.block(im1_rowindice, top_rowindice_col, blocksize, blocksize);

        A.block(i_rowindice, top_rowindice_col, blocksize, blocksize) -=
            L.block(i_rowindice, i_rowindice, blocksize, blocksize) *
            A.block(im1_rowindice, top_rowindice_col, blocksize, blocksize);

        A.block(top_rowindice, ip1_rowindice, blocksize, blocksize) -=
            L.block(top_rowindice, i_rowindice, blocksize, blocksize) *
            A.block(im1_rowindice, ip1_rowindice, blocksize, blocksize);
    }

    return std::make_tuple(L, U);
}


void aggregate_reduced_system_locally(
    Eigen::MatrixXcd& A_schur,
    Eigen::MatrixXcd** A_schur_processes,
    int nblocks_schur_system,
    int partition_blocksize,
    int blocksize,
    int partitions
)
{
    // A_schur will first take as the first row the (local) reduced row of the root process.
    int start_rowindice = 0;

    int start_rowindice_remote = (0 + partition_blocksize - 1) * blocksize;

    int start_colindice = 0;

    int start_colindice_remote = (partition_blocksize - 1) * blocksize;
    
    A_schur.block(0, 0, blocksize, 2 * blocksize) =
        A_schur_processes[0]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize);

    
    // Then, A_schur will aggregate the Schur complement rows of the central processes.
    // Each central process sends 2 rows (4 distinct blocks that have been locally aggregated
    // by the sending process) to the root.
    for (int process_i = 1; process_i < partitions - 1; ++process_i) {
        // Assuming comm.recv is equivalent to direct assignment
        // Upper left double block of process-local A_schur
        start_rowindice = blocksize + (process_i - 1) * 2 * blocksize;

        start_rowindice_remote = (process_i  * partition_blocksize) * blocksize;
        
        start_colindice = 2 * (process_i - 1) * blocksize;

        start_colindice_remote = (process_i * partition_blocksize - 1) * blocksize;
        
        A_schur.block(start_rowindice, start_colindice, blocksize, 2 * blocksize) =
            A_schur_processes[process_i]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize);

        // Upper right single block of process-local A_schur
        start_colindice += 2 * blocksize;

        start_colindice_remote += partition_blocksize * blocksize;

        A_schur.block(start_rowindice, start_colindice, blocksize, blocksize) =
            A_schur_processes[process_i]->block(start_rowindice_remote, start_colindice_remote, blocksize, blocksize);

        // Lower left single block of process-local A_schur
        start_rowindice += blocksize;
        start_colindice -= blocksize;

        start_rowindice_remote = ((process_i + 1) * partition_blocksize - 1) * blocksize;
        start_colindice_remote = (process_i * partition_blocksize) * blocksize;

        A_schur.block(start_rowindice, start_colindice, blocksize, blocksize) =
            A_schur_processes[process_i]->block(start_rowindice_remote, start_colindice_remote, blocksize, blocksize);

        // Lower right double block of process-local A_schur
        start_colindice += blocksize;
        start_colindice_remote += (partition_blocksize -1 ) * blocksize;

        A_schur.block(start_rowindice, start_colindice, blocksize, 2 * blocksize) =
            A_schur_processes[process_i]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize);


    }
    
    // Finally, A_schur will aggregate the Schur complement row of the last process.
    //start_rowindice_remote = 80;

    start_rowindice_remote = (partitions - 1) * partition_blocksize * blocksize;

    start_rowindice = (nblocks_schur_system - 1) * blocksize;


    start_colindice_remote = (partitions - 1) * partition_blocksize * blocksize - blocksize;

    start_colindice = (nblocks_schur_system - 2) * blocksize;
    
    // Assuming comm.recv is equivalent to direct assignment
    A_schur.block(start_rowindice, start_colindice, blocksize, 2 * blocksize) =
        A_schur_processes[partitions-1]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize);
}


void writeback_inverted_system_locally(
    Eigen::MatrixXcd G,
    Eigen::MatrixXcd** G_schur_processes,
    int nblocks_schur_system,
    int partition_blocksize,
    int blocksize,
    int partitions
) 
{
    // full G_BCR will be spread across all processes, the first process will take the upper left double block of G.
    int start_rowindice = 0;

    int start_rowindice_remote = (0 + partition_blocksize - 1) * blocksize;

    int start_colindice = 0;

    int start_colindice_remote = (partition_blocksize - 1) * blocksize;
    
    G_schur_processes[0]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize) =
        G.block(0, 0, blocksize, 2 * blocksize);

    // Then, G will be scattered to the schur complement rows of the central processes.
    // Each central process receives 2 rows (4 distinct blocks that have been locally aggregated
    // by the sending process) from the root.
    for (int process_i = 1; process_i < partitions - 1; ++process_i) {
        // Assuming comm.recv is equivalent to direct assignment
        // Upper left double block of process-local A_schur
        start_rowindice = blocksize + (process_i - 1) * 2 * blocksize;

        start_rowindice_remote = (process_i  * partition_blocksize) * blocksize;
        
        start_colindice = 2 * (process_i - 1) * blocksize;

        start_colindice_remote = (process_i * partition_blocksize - 1) * blocksize;
        
        G_schur_processes[process_i]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize) =
            G.block(start_rowindice, start_colindice, blocksize, 2 * blocksize);
        
        // Upper right single block of process-local G
        start_colindice += 2 * blocksize;

        start_colindice_remote += partition_blocksize * blocksize;

        G_schur_processes[process_i]->block(start_rowindice_remote, start_colindice_remote, blocksize, blocksize) =
            G.block(start_rowindice, start_colindice, blocksize, blocksize);

        // Lower left single block of process-local G
        start_rowindice += blocksize;
        start_colindice -= blocksize;

        start_rowindice_remote = ((process_i + 1) * partition_blocksize - 1) * blocksize;
        start_colindice_remote = (process_i * partition_blocksize) * blocksize;

        G_schur_processes[process_i]->block(start_rowindice_remote, start_colindice_remote, blocksize, blocksize) =
            G.block(start_rowindice, start_colindice, blocksize, blocksize);

        // Lower right double block of process-local G
        start_colindice += blocksize;
        start_colindice_remote += (partition_blocksize -1 ) * blocksize;

        G_schur_processes[process_i]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize) =
           G.block(start_rowindice, start_colindice, blocksize, 2 * blocksize);

    }
    // Finally, G_BCR will scatter the Schur complement row to the last process.

    start_rowindice_remote = (partitions - 1) * partition_blocksize * blocksize;

    start_rowindice = (nblocks_schur_system - 1) * blocksize;


    start_colindice_remote = (partitions - 1) * partition_blocksize * blocksize - blocksize;

    start_colindice = (nblocks_schur_system - 2) * blocksize;
    
    // Assuming comm.recv is equivalent to direct assignment
    G_schur_processes[partitions - 1]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize) =
        G.block(start_rowindice, start_colindice, blocksize, 2 * blocksize);

}

void produce_schur_sequentially(Eigen::MatrixXcd** eigenA,
                             Eigen::MatrixXcd** G_matrices,
                             Eigen::MatrixXcd** L_matrices,
                             Eigen::MatrixXcd** U_matrices,
                             int partitions,
                             int partition_blocksize,
                             int blocksize,
                             int rank) {
   
    for (int i = 0; i < partitions; ++i) {
        int start_blockrow = i * partition_blocksize;

        std::cout << "Process " << rank << " is producing blockrows " << start_blockrow << " to " << start_blockrow + partition_blocksize - 1 << std::endl;


        if (i == 0){
            produceSchurTopLeftCorner(*eigenA[i], *L_matrices[i], *U_matrices[i], *G_matrices[i], start_blockrow, partition_blocksize, blocksize);
        }

        if (i > 0 && i < partitions -1){
            produceSchurCentral(*eigenA[i], *L_matrices[i], *U_matrices[i], *G_matrices[i], start_blockrow, partition_blocksize, blocksize);
        }

        if (i == partitions - 1){
            produceSchurBottomRightCorner(*eigenA[i], *L_matrices[i], *U_matrices[i], *G_matrices[i], start_blockrow, partition_blocksize, blocksize);
        }

    }
}

void produceSchurTopLeftCorner(Eigen::MatrixXcd A,
                               Eigen::MatrixXcd L,
                               Eigen::MatrixXcd U,
                               Eigen::MatrixXcd& G,
                               int start_blockrow,
                               int partition_blocksize,
                               int blocksize) {
    int top_blockrow = start_blockrow;
    int bottom_blockrow = start_blockrow + partition_blocksize;
    
    // Upper left corner produced upwards
    for (int i = bottom_blockrow - 1; i > top_blockrow; --i) {

        int im1_rowindice = (i - 1) * blocksize;
        int i_rowindice = i * blocksize;

        G.block(i_rowindice, im1_rowindice, blocksize, blocksize) =
            -G.block(i_rowindice, i_rowindice, blocksize, blocksize) *
            L.block(i_rowindice, im1_rowindice, blocksize, blocksize);
        
        G.block(im1_rowindice, i_rowindice, blocksize, blocksize) =
            -U.block(im1_rowindice, i_rowindice, blocksize, blocksize) *
            G.block(i_rowindice, i_rowindice, blocksize, blocksize);
        
        G.block(im1_rowindice, im1_rowindice, blocksize, blocksize) =
            (A.block(im1_rowindice, im1_rowindice, blocksize, blocksize).inverse()) -
            U.block(im1_rowindice, i_rowindice, blocksize, blocksize) *
            G.block(i_rowindice, im1_rowindice, blocksize, blocksize);
    }
}


void produceSchurBottomRightCorner(Eigen::MatrixXcd A,
                                   Eigen::MatrixXcd L,
                                   Eigen::MatrixXcd U,
                                   Eigen::MatrixXcd& G,
                                   int start_blockrow,
                                   int partition_blocksize,
                                   int blocksize) {
    int top_blockrow = start_blockrow;
    int bottom_blockrow = start_blockrow + partition_blocksize;
    
    // Lower right corner produced downwards
    for (int i = top_blockrow; i < bottom_blockrow - 1; ++i) {
        int i_rowindice = i * blocksize;
        int ip1_rowindice = (i + 1) * blocksize;

        G.block(i_rowindice, ip1_rowindice, blocksize, blocksize) =
            -G.block(i_rowindice, i_rowindice, blocksize, blocksize) *
            L.block(i_rowindice, ip1_rowindice, blocksize, blocksize);
        
        G.block(ip1_rowindice, i_rowindice, blocksize, blocksize) =
            -U.block(ip1_rowindice, i_rowindice, blocksize, blocksize) *
            G.block(i_rowindice, i_rowindice, blocksize, blocksize);
        
        G.block(ip1_rowindice, ip1_rowindice, blocksize, blocksize) =
            (A.block(ip1_rowindice, ip1_rowindice, blocksize, blocksize).inverse()) -
            U.block(ip1_rowindice, i_rowindice, blocksize, blocksize) *
            G.block(i_rowindice, ip1_rowindice, blocksize, blocksize);
    }
}

void produceSchurBottomRightCorner_2(Eigen::MatrixXcd A,
                                   Eigen::MatrixXcd L,
                                   Eigen::MatrixXcd U,
                                   Eigen::MatrixXcd& G,
                                   int partition_blocksize,
                                   int blocksize) {
    int top_blockrow = 0;
    int bottom_blockrow = partition_blocksize;
    
    // Lower right corner produced downwards
    for (int i = top_blockrow; i < bottom_blockrow - 1; ++i) {
        int i_rowindice = i * blocksize;
        int ip1_rowindice = (i + 1) * blocksize;
        int i_rowindiceCol = i_rowindice + blocksize;
        int ip1_rowindiceCol = ip1_rowindice + blocksize;

        G.block(i_rowindice, ip1_rowindiceCol, blocksize, blocksize) =
            -G.block(i_rowindice, i_rowindiceCol, blocksize, blocksize) *
            L.block(i_rowindice, ip1_rowindiceCol, blocksize, blocksize);
        
        G.block(ip1_rowindice, i_rowindiceCol, blocksize, blocksize) =
            -U.block(ip1_rowindice, i_rowindiceCol, blocksize, blocksize) *
            G.block(i_rowindice, i_rowindiceCol, blocksize, blocksize);
        
        G.block(ip1_rowindice, ip1_rowindiceCol, blocksize, blocksize) =
            (A.block(ip1_rowindice, ip1_rowindiceCol, blocksize, blocksize).inverse()) -
            U.block(ip1_rowindice, i_rowindiceCol, blocksize, blocksize) *
            G.block(i_rowindice, ip1_rowindiceCol, blocksize, blocksize);
    }
}

void produceSchurCentral(Eigen::MatrixXcd A,
                         Eigen::MatrixXcd L,
                         Eigen::MatrixXcd U,
                         Eigen::MatrixXcd& G,
                         int start_blockrow,
                         int partition_blocksize,
                         int blocksize) {
    int top_blockrow = start_blockrow;
    int bottom_blockrow = start_blockrow + partition_blocksize;

    int top_rowindice = top_blockrow * blocksize;
    int topp1_rowindice = (top_blockrow + 1) * blocksize;
    int topp2_rowindice = (top_blockrow + 2) * blocksize;

    int botm1_rowindice = (bottom_blockrow - 2) * blocksize;
    int bot_rowindice = (bottom_blockrow - 1) * blocksize;

    G.block(bot_rowindice, botm1_rowindice, blocksize, blocksize) =
        -1 * (G.block(bot_rowindice, top_rowindice, blocksize, blocksize) *
              L.block(top_rowindice, botm1_rowindice, blocksize, blocksize) +
              G.block(bot_rowindice, bot_rowindice, blocksize, blocksize) *
              L.block(bot_rowindice, botm1_rowindice, blocksize, blocksize));

    G.block(botm1_rowindice, bot_rowindice, blocksize, blocksize) =
        -1 * (U.block(botm1_rowindice, bot_rowindice, blocksize, blocksize) *
              G.block(bot_rowindice, bot_rowindice, blocksize, blocksize) +
              U.block(botm1_rowindice, top_rowindice, blocksize, blocksize) *
              G.block(top_rowindice, bot_rowindice, blocksize, blocksize));

    for (int i = bottom_blockrow - 2; i > top_blockrow; --i) {
        int i_rowindice = i * blocksize;
        int ip1_rowindice = (i + 1) * blocksize;

        G.block(top_rowindice, i_rowindice, blocksize, blocksize) =
            -1 * (G.block(top_rowindice, top_rowindice, blocksize, blocksize) *
                  L.block(top_rowindice, i_rowindice, blocksize, blocksize) +
                  G.block(top_rowindice, ip1_rowindice, blocksize, blocksize) *
                  L.block(ip1_rowindice, i_rowindice, blocksize, blocksize));

        G.block(i_rowindice, top_rowindice, blocksize, blocksize) =
            -1 * (U.block(i_rowindice, ip1_rowindice, blocksize, blocksize) *
                  G.block(ip1_rowindice, top_rowindice, blocksize, blocksize) +
                  U.block(i_rowindice, top_rowindice, blocksize, blocksize) *
                  G.block(top_rowindice, top_rowindice, blocksize, blocksize));
    }

    for (int i = bottom_blockrow - 2; i > top_blockrow + 1; --i) {
        int im1_rowindice = (i - 1) * blocksize;
        int i_rowindice = i * blocksize;
        int ip1_rowindice = (i + 1) * blocksize;

        // Compute the inverse block
        Eigen::MatrixXcd invBlock = A.block(i_rowindice, i_rowindice, blocksize, blocksize).inverse();
        G.block(i_rowindice, i_rowindice, blocksize, blocksize) = invBlock -
            U.block(i_rowindice, top_rowindice, blocksize, blocksize) *
            G.block(top_rowindice, i_rowindice, blocksize, blocksize) -
            U.block(i_rowindice, ip1_rowindice, blocksize, blocksize) *
            G.block(ip1_rowindice, i_rowindice, blocksize, blocksize);

        G.block(im1_rowindice, i_rowindice, blocksize, blocksize) =
            -1 * (U.block(im1_rowindice, top_rowindice, blocksize, blocksize) *
                  G.block(top_rowindice, i_rowindice, blocksize, blocksize) +
                  U.block(im1_rowindice, i_rowindice, blocksize, blocksize) *
                  G.block(i_rowindice, i_rowindice, blocksize, blocksize));

        G.block(i_rowindice, im1_rowindice, blocksize, blocksize) =
            -1 * (G.block(i_rowindice, top_rowindice, blocksize, blocksize) *
                  L.block(top_rowindice, im1_rowindice, blocksize, blocksize) +
                  G.block(i_rowindice, i_rowindice, blocksize, blocksize) *
                  L.block(i_rowindice, im1_rowindice, blocksize, blocksize));
    }

    G.block(topp1_rowindice, topp1_rowindice, blocksize, blocksize) =
        (A.block(topp1_rowindice, topp1_rowindice, blocksize, blocksize).inverse()) -
        U.block(topp1_rowindice, top_rowindice, blocksize, blocksize) *
        G.block(top_rowindice, topp1_rowindice, blocksize, blocksize) -
        U.block(topp1_rowindice, topp2_rowindice, blocksize, blocksize) *
        G.block(topp2_rowindice, topp1_rowindice, blocksize, blocksize);
}


void produceSchurCentral_2(Eigen::MatrixXcd A,
                         Eigen::MatrixXcd L,
                         Eigen::MatrixXcd U,
                         Eigen::MatrixXcd& G,
                         int partition_blocksize,
                         int blocksize) {
    int top_blockrow = 0;
    int bottom_blockrow = partition_blocksize;

    int top_rowindice = 0;
    int topp1_rowindice = blocksize;
    int topp2_rowindice = blocksize << 1;

    int botm1_rowindice = (bottom_blockrow - 2) * blocksize;
    int bot_rowindice = (bottom_blockrow - 1) * blocksize;

    int top_rowindiceCol = blocksize;
    int topp1_rowindiceCol = blocksize << 1;
    int topp2_rowindiceCol = (blocksize << 1) + blocksize;

    int botm1_rowindiceCol = (bottom_blockrow - 1) * blocksize;
    int bot_rowindiceCol = bottom_blockrow * blocksize;

    G.block(bot_rowindice, botm1_rowindiceCol, blocksize, blocksize) =
        -1 * (G.block(bot_rowindice, top_rowindiceCol, blocksize, blocksize) *
              L.block(top_rowindice, botm1_rowindiceCol, blocksize, blocksize) +
              G.block(bot_rowindice, bot_rowindiceCol, blocksize, blocksize) *
              L.block(bot_rowindice, botm1_rowindiceCol, blocksize, blocksize));

    // G.block(botm1_rowindice, bot_rowindiceCol, blocksize, blocksize) =
    //     -1 * (U.block(botm1_rowindice, bot_rowindiceCol, blocksize, blocksize) *
    //           G.block(bot_rowindice, bot_rowindiceCol, blocksize, blocksize) +
    //           U.block(botm1_rowindice, top_rowindiceCol, blocksize, blocksize) *
    //           G.block(top_rowindice, bot_rowindiceCol, blocksize, blocksize));

    for (int i = bottom_blockrow - 2; i > top_blockrow; --i) {
        int i_rowindice = i * blocksize;
        int ip1_rowindice = (i + 1) * blocksize;
	int i_rowindiceCol = i_rowindice + blocksize;
	int ip1_rowindiceCol = ip1_rowindice + blocksize;

        G.block(top_rowindice, i_rowindiceCol, blocksize, blocksize) =
            -1 * (G.block(top_rowindice, top_rowindiceCol, blocksize, blocksize) *
                  L.block(top_rowindice, i_rowindiceCol, blocksize, blocksize) +
                  G.block(top_rowindice, ip1_rowindiceCol, blocksize, blocksize) *
                  L.block(ip1_rowindice, i_rowindiceCol, blocksize, blocksize));

        G.block(i_rowindice, top_rowindiceCol, blocksize, blocksize) =
            -1 * (U.block(i_rowindice, ip1_rowindiceCol, blocksize, blocksize) *
                  G.block(ip1_rowindice, top_rowindiceCol, blocksize, blocksize) +
                  U.block(i_rowindice, top_rowindiceCol, blocksize, blocksize) *
                  G.block(top_rowindice, top_rowindiceCol, blocksize, blocksize));
    }

    for (int i = bottom_blockrow - 2; i > top_blockrow + 1; --i) {
        int im1_rowindice = (i - 1) * blocksize;
        int i_rowindice = i * blocksize;
        int ip1_rowindice = (i + 1) * blocksize;
	int im1_rowindiceCol = im1_rowindice + blocksize;
	int i_rowindiceCol = i_rowindice + blocksize;
	int ip1_rowindiceCol = ip1_rowindice + blocksize;

        // Compute the inverse block
        Eigen::MatrixXcd invBlock = A.block(i_rowindice, i_rowindiceCol, blocksize, blocksize).inverse();
        G.block(i_rowindice, i_rowindiceCol, blocksize, blocksize) = invBlock -
            U.block(i_rowindice, top_rowindiceCol, blocksize, blocksize) *
            G.block(top_rowindice, i_rowindiceCol, blocksize, blocksize) -
            U.block(i_rowindice, ip1_rowindiceCol, blocksize, blocksize) *
            G.block(ip1_rowindice, i_rowindiceCol, blocksize, blocksize);

        G.block(im1_rowindice, i_rowindiceCol, blocksize, blocksize) =
            -1 * (U.block(im1_rowindice, top_rowindiceCol, blocksize, blocksize) *
                  G.block(top_rowindice, i_rowindiceCol, blocksize, blocksize) +
                  U.block(im1_rowindice, i_rowindiceCol, blocksize, blocksize) *
                  G.block(i_rowindice, i_rowindiceCol, blocksize, blocksize));

        G.block(i_rowindice, im1_rowindiceCol, blocksize, blocksize) =
            -1 * (G.block(i_rowindice, top_rowindiceCol, blocksize, blocksize) *
                  L.block(top_rowindice, im1_rowindiceCol, blocksize, blocksize) +
                  G.block(i_rowindice, i_rowindiceCol, blocksize, blocksize) *
                  L.block(i_rowindice, im1_rowindiceCol, blocksize, blocksize));
    }

    G.block(topp1_rowindice, topp1_rowindiceCol, blocksize, blocksize) =
        (A.block(topp1_rowindice, topp1_rowindiceCol, blocksize, blocksize).inverse()) -
        U.block(topp1_rowindice, top_rowindiceCol, blocksize, blocksize) *
        G.block(top_rowindice, topp1_rowindiceCol, blocksize, blocksize) -
        U.block(topp1_rowindice, topp2_rowindiceCol, blocksize, blocksize) *
        G.block(topp2_rowindice, topp1_rowindiceCol, blocksize, blocksize);
}

void aggregate_Gblocks_tofinalinverse_sequentially(int partitions,
                                       int partition_blocksize,
                                       int blocksize,
                                       Eigen::MatrixXcd** G_matrices,
                                       Eigen::MatrixXcd& G_final
)
{       
    for (int i = 0; i < partitions; ++i) {
        int start_blockrow = i * partition_blocksize;

        if (i == 0){
            for (int j = 0; j < partition_blocksize; ++j){
                G_final.block(start_blockrow * blocksize + j * blocksize, start_blockrow * blocksize + j * blocksize, blocksize, blocksize) =\
                    (*G_matrices[i]).block(start_blockrow * blocksize + j * blocksize, start_blockrow * blocksize + j * blocksize, blocksize, blocksize);
                G_final.block(start_blockrow * blocksize + j * blocksize, (start_blockrow + 1) * blocksize + j * blocksize, blocksize, blocksize) =\
                    (*G_matrices[i]).block(start_blockrow * blocksize + j * blocksize, (start_blockrow + 1) * blocksize + j * blocksize, blocksize, blocksize);
                if (j < partition_blocksize - 1){
                    G_final.block((start_blockrow + 1) * blocksize + j * blocksize, start_blockrow * blocksize + j * blocksize, blocksize, blocksize) =\
                        (*G_matrices[i]).block((start_blockrow + 1) * blocksize + j * blocksize, start_blockrow * blocksize + j * blocksize, blocksize, blocksize);
                }
            }
            
        }

        if (i > 0 && i < partitions - 1){
            for (int j = 0; j < partition_blocksize; ++j){
                G_final.block(start_blockrow * blocksize + j * blocksize, start_blockrow * blocksize + j * blocksize, blocksize, blocksize) =\
                (*G_matrices[i]).block(start_blockrow * blocksize + j * blocksize, start_blockrow * blocksize + j * blocksize, blocksize, blocksize);

                G_final.block(start_blockrow * blocksize + j * blocksize, (start_blockrow + 1) * blocksize + j * blocksize, blocksize, blocksize) =\
                (*G_matrices[i]).block(start_blockrow * blocksize + j * blocksize, (start_blockrow + 1) * blocksize + j * blocksize, blocksize, blocksize);

                G_final.block(start_blockrow * blocksize + j * blocksize, (start_blockrow - 1) * blocksize + j * blocksize, blocksize, blocksize) =\
                (*G_matrices[i]).block(start_blockrow * blocksize + j * blocksize, (start_blockrow - 1) * blocksize + j * blocksize, blocksize, blocksize);
            }
        }

        if (i == partitions - 1){
            for (int j = 0; j < partition_blocksize; ++j){
                G_final.block(start_blockrow * blocksize + j * blocksize, start_blockrow * blocksize + j * blocksize, blocksize, blocksize) =\
                (*G_matrices[i]).block(start_blockrow * blocksize + j * blocksize, start_blockrow * blocksize + j * blocksize, blocksize, blocksize);

                G_final.block(start_blockrow * blocksize + j * blocksize, (start_blockrow - 1) * blocksize + j * blocksize, blocksize, blocksize) =\
                (*G_matrices[i]).block(start_blockrow * blocksize + j * blocksize, (start_blockrow - 1) * blocksize + j * blocksize, blocksize, blocksize);

                if(j < partition_blocksize -1){
                    G_final.block(start_blockrow * blocksize + j * blocksize, (start_blockrow + 1) * blocksize + j * blocksize, blocksize, blocksize) =\
                    (*G_matrices[i]).block(start_blockrow * blocksize + j * blocksize, (start_blockrow + 1) * blocksize + j * blocksize, blocksize, blocksize);
                }
            }
        }

    }

}


void fill_buffer(Eigen::MatrixXcd& inMatrix, Eigen::MatrixXcd** eigenA, int partition_blocksize, int blocksize, int rank, int partitions) {

    if(rank == 0) {
	int start_rowindice_remote = (0 + partition_blocksize - 1) * blocksize;

	int start_colindice_remote = (partition_blocksize - 1) * blocksize;

	inMatrix.block(0, 0, blocksize, 2*blocksize) = (eigenA[0]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize));
    } else if (rank == partitions-1) {
	int start_rowindice_remote = (partitions - 1) * partition_blocksize * blocksize;

	int start_colindice_remote = (partitions - 1) * partition_blocksize * blocksize - blocksize;

	// Assuming comm.recv is equivalent to direct assignment
	inMatrix.block(0, 0, blocksize, 2*blocksize) = (eigenA[partitions-1]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize));
    } else {
	// Upper right double block of process-local A_schur
        int start_rowindice_remote = (rank  * partition_blocksize) * blocksize;
        
        int start_colindice_remote = (rank * partition_blocksize - 1) * blocksize;
        
        inMatrix.block(0, 0, blocksize, 2*blocksize) =  (eigenA[rank]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize));

        // Upper right single block of process-local A_schur
        start_colindice_remote += partition_blocksize * blocksize;

        inMatrix.block(0, 2*blocksize, blocksize, blocksize) = (eigenA[rank]->block(start_rowindice_remote, start_colindice_remote, blocksize, blocksize));

        // Lower left single block of process-local A_schur
        start_rowindice_remote = ((rank + 1) * partition_blocksize - 1) * blocksize;
        start_colindice_remote = (rank * partition_blocksize) * blocksize;

        inMatrix.block(0, 3*blocksize, blocksize, blocksize) = (eigenA[rank]->block(start_rowindice_remote, start_colindice_remote, blocksize, blocksize));

        // Lower right double block of process-local A_schur
        start_colindice_remote += (partition_blocksize -1 ) * blocksize;

        inMatrix.block(0, 4*blocksize, blocksize, 2*blocksize) = (eigenA[rank]->block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize));
    }
}

void fill_buffer_2(Eigen::MatrixXcd& inMatrix, Eigen::MatrixXcd processA, int partition_blocksize, int blocksize, int rank, int partitions) {

    if(rank == 0) {
	int start_rowindice_remote = (0 + partition_blocksize - 1) * blocksize;

	int start_colindice_remote = (partition_blocksize - 1) * blocksize;

	inMatrix.block(0, 0, blocksize, 2*blocksize) = (processA.block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize));
    } else if (rank == partitions-1) {
	// Assuming comm.recv is equivalent to direct assignment
	inMatrix.block(0, 0, blocksize, 2*blocksize) = (processA.block(0, 0, blocksize, 2 * blocksize));
    } else {
	// Upper right double block of process-local A_schur
        int start_rowindice_remote = (rank  * partition_blocksize) * blocksize;
        
        int start_colindice_remote = (rank * partition_blocksize - 1) * blocksize;
        
        inMatrix.block(0, 0, blocksize, 2*blocksize) =  (processA.block(0, 0, blocksize, 2 * blocksize));

        // Upper right single block of process-local A_schur
        start_colindice_remote = partition_blocksize * blocksize;

        inMatrix.block(0, 2*blocksize, blocksize, blocksize) = (processA.block(0, start_colindice_remote, blocksize, blocksize));

        // Lower left single block of process-local A_schur
        start_rowindice_remote = (partition_blocksize - 1) * blocksize;

        inMatrix.block(0, 3*blocksize, blocksize, blocksize) = (processA.block(start_rowindice_remote, blocksize, blocksize, blocksize));

        // Lower right double block of process-local A_schur
        start_colindice_remote = partition_blocksize * blocksize;

        inMatrix.block(0, 4*blocksize, blocksize, 2*blocksize) = (processA.block(start_rowindice_remote, start_colindice_remote, blocksize, 2 * blocksize));
    }
}



void fill_reduced_schur_matrix(Eigen::MatrixXcd& A_schur, double* comm_buf, int in_buf_size, int blocksize, int partitions) {

    const int rowSize = blocksize;
    const int colSizeCorner = blocksize << 1;
    const int colSizeMiddle = colSizeCorner + blocksize;
    const int half_buf_size = 6*blocksize*blocksize;
   
    // Fill in the 2 blocks from the top process (i.e. rank 0)
    A_schur.block( 0, 0, rowSize, colSizeCorner) = 
       	    Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
	    ( (std::complex<double>*) comm_buf, rowSize, colSizeCorner);
    
    //Fill in the 2 blocks from the bottom process (i.e rank (partitions-1))
    A_schur.block( (2*partitions-3)*blocksize, (2*partitions-4)*blocksize, rowSize, colSizeCorner) = 
            Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
	    ( (std::complex<double>*) (comm_buf + (partitions-1)*in_buf_size), rowSize, colSizeCorner);

    //Fill in the the 6 blocks over two rows from the processes in the middle (i.e. rank > 0 && rank < (partitions - 1))
    for(int i = 1; i < partitions-1; ++i) {
	A_schur.block( (2*i-1)*blocksize, (2*i-2)*blocksize, rowSize, colSizeMiddle) = 
		Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
		( (std::complex<double>*) (comm_buf + i*in_buf_size), rowSize, colSizeMiddle);

	A_schur.block( (2*i)*blocksize, (2*i-1)*blocksize, rowSize, colSizeMiddle) = 
		Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
		( (std::complex<double>*) (comm_buf + i*in_buf_size + half_buf_size), rowSize, colSizeMiddle);
    }

}


void fill_reduced_schur_matrix_gpu(cuDoubleComplex* A_schur_gpu, std::complex<double>* comm_buf_custom, int blocksize, int partitions, int l_dim) {

    // Change this aterward maybe
    l_dim = (2*partitions - 2) * blocksize;
    int rowBlocks = 2;
    int rowBlock = 0;
    int colBlock = 0;
    int buffBlock = 0;

    // Fill in the 2 blocks from the top process (i.e. rank 0)
    // leading dimension of comm_buff_custom is blocksize since the comm_buf_custom is conceptualized as a blocksize x (6*blocksize *(paritions-2) + 4 * blocksize) matrix
    // for(int j = 0; j < 2*blocksize; ++j) {
	// cudaErrchk(cudaMemcpy(A_schur_gpu+j*l_dim, reinterpret_cast<cuDoubleComplex*>(comm_buf_custom+j*blocksize), blocksize * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    // }
    copy_rowblocks_buffer2GPU(A_schur_gpu, reinterpret_cast<cuDoubleComplex*>(comm_buf_custom), blocksize,  l_dim, rowBlocks, rowBlock, colBlock, buffBlock);

    //Fill in the 2 blocks from the bottom process (i.e rank (partitions-1))
    // int colsSkip = (2*partitions-4) * blocksize * l_dim;
    // int rowOffset = (2*partitions-3) * blocksize;
    // int buffOffset = (partitions-2)*6*blocksize*blocksize + 2*blocksize*blocksize;
    // for(int j = 0; j < 2*blocksize; ++j) {
	// cudaErrchk(cudaMemcpy(A_schur_gpu+colsSkip+rowOffset+j*l_dim, reinterpret_cast<cuDoubleComplex*>(comm_buf_custom+buffOffset+j*blocksize), blocksize * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    // }

    rowBlocks = 2;
    rowBlock = (2*partitions - 3);
    colBlock = (2*partitions - 4);
    buffBlock = (partitions - 2) * 6  + 2;
    copy_rowblocks_buffer2GPU(A_schur_gpu, reinterpret_cast<cuDoubleComplex*>(comm_buf_custom), blocksize,  l_dim, rowBlocks, rowBlock, colBlock, buffBlock);

    //Fill in the the 6 blocks over two rows from the processes in the middle (i.e. rank > 0 && rank < (partitions - 1))
    for(int i = 1; i < partitions-1; ++i) {
        // First Blockrow
        // colsSkip = (2*i-2) * blocksize * l_dim;
        // rowOffset = (2*i-1) * blocksize;
        // buffOffset = (i-1) * blocksize * blocksize * 6 + 2 * blocksize * blocksize;

        rowBlocks = 3;
        rowBlock = (2*i - 1);
        colBlock = (2*i - 2);
        buffBlock = (i - 1) * 6 + 2;

        // for(int j = 0; j < 3*blocksize; ++j) {
        //     cudaErrchk(cudaMemcpy(A_schur_gpu+colsSkip+rowOffset+j*l_dim, reinterpret_cast<cuDoubleComplex*>(comm_buf_custom+buffOffset+j*blocksize), blocksize * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        // }
        copy_rowblocks_buffer2GPU(A_schur_gpu, reinterpret_cast<cuDoubleComplex*>(comm_buf_custom), blocksize,  l_dim, rowBlocks, rowBlock, colBlock, buffBlock);

        // Second Blockrow
        // colsSkip = (2*i-1) * blocksize * l_dim;
        // rowOffset = 2*i * blocksize;
        // buffOffset = (i-1) * blocksize * blocksize * 6 + 2 * blocksize * blocksize + 3 * blocksize * blocksize;

        rowBlocks = 3;
        rowBlock = (2*i);
        colBlock = (2*i - 1);
        buffBlock = (i - 1) * 6 + 2 + 3;

        // for(int j = 0; j < 3*blocksize; ++j) {
        //     cudaErrchk(cudaMemcpy(A_schur_gpu+colsSkip+rowOffset+j*l_dim, reinterpret_cast<cuDoubleComplex*>(comm_buf_custom+buffOffset+j*blocksize), blocksize * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        // }
        copy_rowblocks_buffer2GPU(A_schur_gpu, reinterpret_cast<cuDoubleComplex*>(comm_buf_custom), blocksize,  l_dim, rowBlocks, rowBlock, colBlock, buffBlock);
    }
}


void fill_reduced_schur_matrix_cd(Eigen::MatrixXcd& A_schur, std::complex<double>* comm_buf_custom, int in_buf_size, int blocksize, int partitions, int rank) {

    const int rowSize = blocksize;
    const int colSizeCorner = blocksize << 1;
    const int colSizeMiddle = colSizeCorner + blocksize;
    const int half_buf_size = 6*blocksize*blocksize;

    // Eigen::MatrixXcd A_schur_cd(A_schur.rows(), A_schur.cols());
    // // A_schur_full is a debug object to check if all the elements have been correctly gathered on all nodes.
    // Eigen::MatrixXcd A_schur_full = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(comm_buf_custom, rowSize, blocksize *  ((partitions - 2) * 6 + 2 * 2));
   
    // Fill in the 2 blocks from the top process (i.e. rank 0)
    // A_schur.block( 0, 0, rowSize, colSizeCorner) = 
    //    	    Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
	//     ( (std::complex<double>*) comm_buf, rowSize, colSizeCorner);

    A_schur.block( 0, 0, rowSize, colSizeCorner) = 
       	    Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
        ( comm_buf_custom, rowSize, colSizeCorner);

    
    // if (!(A_schur.block( 0, 0, rowSize, colSizeCorner).isApprox(A_schur_cd.block( 0, 0, rowSize, colSizeCorner)))) {
    //     std::cout << "Warning: first block is not equal on rank: " << rank << std::endl;
    // } 
    
    //Fill in the 2 blocks from the bottom process (i.e rank (partitions-1))
    // A_schur.block( (2*partitions-3)*blocksize, (2*partitions-4)*blocksize, rowSize, colSizeCorner) = 
    //         Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
	//     ( (std::complex<double>*) (comm_buf + (partitions-1)*in_buf_size), rowSize, colSizeCorner);

    A_schur.block(  (2*partitions-3)*blocksize, (2*partitions-4)*blocksize, rowSize, colSizeCorner) = 
       	    Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
        ( comm_buf_custom + (partitions - 2) * in_buf_size / 2 + in_buf_size / 6, rowSize, colSizeCorner);

    // if (!(A_schur.block( (2*partitions-3)*blocksize, (2*partitions-4)*blocksize, rowSize, colSizeCorner).isApprox(A_schur_cd.block( (2*partitions-3)*blocksize, (2*partitions-4)*blocksize, rowSize, colSizeCorner)))) {
    //     std::cout << "Warning: second last block is not equal on rank: " << rank << std::endl;
    // }

    //Fill in the the 6 blocks over two rows from the processes in the middle (i.e. rank > 0 && rank < (partitions - 1))
    for(int i = 1; i < partitions-1; ++i) {
        // A_schur.block( (2*i-1)*blocksize, (2*i-2)*blocksize, rowSize, colSizeMiddle) = 
        //     Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
        //     ( (std::complex<double>*) (comm_buf + i*in_buf_size), rowSize, colSizeMiddle);

        A_schur.block( (2*i-1)*blocksize, (2*i-2)*blocksize, rowSize, colSizeMiddle) = 
            Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
            (comm_buf_custom + (i-1)*in_buf_size/2 + in_buf_size / 6, rowSize, colSizeMiddle);

        // if (!(A_schur.block( (2*i-1)*blocksize, (2*i-2)*blocksize, rowSize, colSizeMiddle).isApprox(A_schur_cd.block( (2*i-1)*blocksize, (2*i-2)*blocksize, rowSize, colSizeMiddle)))) {
        //     std::cout << "Warning: middle upper section block: " << i <<  " is not equal on rank: " << rank << std::endl;
        // }

        // A_schur.block( (2*i)*blocksize, (2*i-1)*blocksize, rowSize, colSizeMiddle) = 
        //     Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
        //     ( (std::complex<double>*) (comm_buf + i*in_buf_size + half_buf_size), rowSize, colSizeMiddle);

        A_schur.block( (2*i)*blocksize, (2*i-1)*blocksize, rowSize, colSizeMiddle) = 
            Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
            (comm_buf_custom + (i-1)*in_buf_size / 2 + in_buf_size / 6 + in_buf_size / 4 , rowSize, colSizeMiddle);

        // if (!(A_schur.block( (2*i)*blocksize, (2*i-1)*blocksize, rowSize, colSizeMiddle).isApprox(A_schur_cd.block( (2*i)*blocksize, (2*i-1)*blocksize, rowSize, colSizeMiddle)))) {
        //     std::cout << "Warning: middle lower section block: " << i <<  " is not equal on rank: " << rank << std::endl;
        // }

    }



    // if (rank == 0)
    // {   std::cout << "A_schur norm: " << A_schur.norm() << std::endl;
    //     std::cout << "A_schur_full norm: " << A_schur_full.norm() << std::endl; }

    // if (rank == 0) {

    //     // std::cout << "A_full_schur.block( 0, 0, rowSize, colSizeCorner) = " << A_schur_full.block( 0, 0, rowSize, colSizeCorner) << std::endl;
    //     // //std::cout << "A_schur_cd.block( 0, 0, rowSize, colSizeCorner) = " << A_schur_cd.block( 0, 0, rowSize, colSizeCorner) << std::endl;
    //     // std::cout << "A_schur.block( 0, 0, rowSize, colSizeCorner) = " << A_schur.block( 0, 0, rowSize, colSizeCorner) << std::endl;
    //     // std::cout << "A_schur first partition block: " << A_schur.block( blocksize, 0, blocksize, blocksize) << std::endl;
    //     // std::cout << "A_schur second partition block: " << A_schur.block( blocksize, blocksize, blocksize, blocksize) << std::endl;
    //     // std::cout << "A_schur second partition block: " << A_schur.block( blocksize, blocksize, blocksize, blocksize) << std::endl;
    //     // std::cout << "A_schur second last partition block: " << A_schur.block( blocksize + (partitions - 2 ) * 2 * blocksize, (partitions - 2 ) * 2 * blocksize, blocksize, blocksize) << std::endl;
    //     // std::cout << "A_schur last partition block: " << A_schur.block( blocksize + (partitions - 2 ) * 2 * blocksize, blocksize + (partitions - 2 ) * 2 * blocksize, blocksize, blocksize) << std::endl;
    // }

}



Eigen::MatrixXcd psr_seqsolve(int N,
                             int blocksize,
                             int n_blocks,
                             int partitions,
                             int partition_blocksize,
                             int rank,
                             int n_blocks_schursystem,
                             Eigen::MatrixXcd& eigenA_read_in,
                             bool compare_reference
){

    Eigen::MatrixXcd** eigenA = new Eigen::MatrixXcd*[partitions];
    for(int i = 0; i < partitions; ++i) {
        eigenA[i] = new Eigen::MatrixXcd(N, N);
        *eigenA[i] = eigenA_read_in;
    }

    // Referece inverse
    Eigen::MatrixXcd full_inverse;
    if(compare_reference){
        full_inverse = eigenA_read_in.inverse();
    }

    // Begin reduce_schur
    Eigen::MatrixXcd** G_matrices = new Eigen::MatrixXcd*[partitions];
    Eigen::MatrixXcd** L_matrices = new Eigen::MatrixXcd*[partitions];
    Eigen::MatrixXcd** U_matrices = new Eigen::MatrixXcd*[partitions];

    reduce_schur_sequentially(eigenA, G_matrices, L_matrices, U_matrices,
                             partitions,
                             partition_blocksize,
                             blocksize,
                             rank);

    // End reduce_schur

    //Define aggregated schur matrix on "process 0"
    Eigen::MatrixXcd A_schur = Eigen::MatrixXcd(blocksize*n_blocks_schursystem, blocksize*n_blocks_schursystem);
    A_schur.setZero();

    aggregate_reduced_system_locally(A_schur, eigenA, n_blocks_schursystem, partition_blocksize, blocksize, partitions);

    auto G_schur = A_schur.inverse();

    writeback_inverted_system_locally(G_schur, G_matrices, n_blocks_schursystem, partition_blocksize, blocksize, partitions);

    produce_schur_sequentially(eigenA, G_matrices, L_matrices, U_matrices,
                             partitions,
                             partition_blocksize,
                             blocksize,
                             rank);


    Eigen::MatrixXcd G_final = Eigen::MatrixXcd(N, N);
    G_final.setZero();

    aggregate_Gblocks_tofinalinverse_sequentially(partitions,
                                       partition_blocksize,
                                       blocksize,
                                       G_matrices,
                                       G_final
    );

    if(compare_reference){
        compareSINV_referenceInverse_byblock(n_blocks,
                                     blocksize,
                                     G_final,
                                     full_inverse,
                                     0
        );
    }

    for(int i = 0; i < partitions; ++i) {
        delete eigenA[i];
        delete G_matrices[i];
        delete L_matrices[i];
        delete U_matrices[i];
    }

    delete[] eigenA;
    delete[] G_matrices;
    delete[] L_matrices;
    delete[] U_matrices;

    return G_final;


}


Eigen::MatrixXcd psr_solve(int N,
                             int blocksize,
                             int n_blocks,
                             int partitions,
                             int partition_blocksize,
                             int rank,
                             int n_blocks_schursystem,
                             Eigen::MatrixXcd& eigenA_read_in,
                             bool compare_reference
){  

    Eigen::MatrixXcd eigenA2 = eigenA_read_in;

    // Referece inverse
    Eigen::MatrixXcd full_inverse;
    if(compare_reference){
        full_inverse = eigenA_read_in.inverse();
    }
	
    //Limit it to the processes partition of A
    int start_blockrow = rank * partition_blocksize;
    int rowSizePartition = partition_blocksize * blocksize;
    int colSizePartition = (partition_blocksize + 2) * blocksize;
    Eigen::MatrixXcd processA;
    Eigen::MatrixXcd G,L,U;

    if (rank == 0) {
	processA = eigenA2.block(0, 0, rowSizePartition, colSizePartition-blocksize);
	G = Eigen::MatrixXcd::Zero(rowSizePartition, colSizePartition-blocksize);
    }

    if (rank > 0 && rank < partitions - 1) {
	int startRowIndex = start_blockrow*blocksize;
	int startColIndex = (start_blockrow-1)*blocksize;
	processA = eigenA2.block(startRowIndex, startColIndex, rowSizePartition, colSizePartition);
	G = Eigen::MatrixXcd::Zero(rowSizePartition, colSizePartition);
    }
    if (rank == partitions - 1) {
	int startRowIndex = start_blockrow*blocksize;
	int startColIndex = (start_blockrow-1)*blocksize;
	processA = eigenA2.block(startRowIndex, startColIndex, rowSizePartition, colSizePartition-blocksize);
	G = Eigen::MatrixXcd::Zero(rowSizePartition, colSizePartition-blocksize);
    }


    // Start reduce_schur
    std::cout << "Process " << rank << " is reducing blockrows " << start_blockrow << " to " << start_blockrow + partition_blocksize - 1 << std::endl;


    if (rank == 0){
        auto result = reduce_schur_topleftcorner(processA, 0, partition_blocksize, blocksize);
        L = std::get<0>(result);
        U = std::get<1>(result);
    }

    if (rank > 0 && rank < partitions - 1){
        auto result = reduce_schur_central_2(processA, partition_blocksize, blocksize);
        L = std::get<0>(result);
        U = std::get<1>(result);
    }
    
    if (rank == partitions - 1){
        auto result = reduce_schur_bottomrightcorner_2(processA, partition_blocksize, blocksize);
        L = std::get<0>(result);
        U = std::get<1>(result);
    }
    // End reduce_schur

    // Start of MPIALLGATHER for reduced_schur_system and inverse of said system
    Eigen::MatrixXcd A_schur = Eigen::MatrixXcd(blocksize*n_blocks_schursystem, blocksize*n_blocks_schursystem);
    A_schur.setZero();

    unsigned long comm_buf_size = (blocksize * blocksize * partitions * 6) << 1; 
    double* comm_buf = new double[comm_buf_size];

    unsigned long in_buf_size = (blocksize * blocksize * 6) << 1;
    Eigen::MatrixXcd inMatrix = Eigen::MatrixXcd::Zero(blocksize, 6*blocksize);
    fill_buffer_2(inMatrix, processA, partition_blocksize, blocksize, rank, partitions);


    double* in_buf = (double*) inMatrix.data();
    MPI_Allgather(in_buf, in_buf_size, MPI_DOUBLE, comm_buf, in_buf_size, MPI_DOUBLE, MPI_COMM_WORLD);

    fill_reduced_schur_matrix(A_schur, comm_buf, in_buf_size, blocksize, partitions);

    delete[] comm_buf;

    auto G_schur = A_schur.inverse();
    // End of MPIALLGATHER for reduced_schur_system and inverse of said system

    // Start of writeback of reduced inverse to full G partitions
    if(rank == 0) {
	int start_rowindice = (partition_blocksize - 1) * blocksize;
	int start_colindice = (partition_blocksize - 1) * blocksize;
	G.block(start_rowindice, start_colindice, blocksize, (blocksize << 1)) = G_schur.block(0, 0, blocksize, (blocksize << 1));	
    }
    if(rank > 0 && rank < partitions - 1) {
	// Upper left double block of process-local G
	int start_rowindice_remote = (1 + ((rank - 1) << 1)) * blocksize; // (rank - 1) * 2 + 1
	int start_colindice_remote = ((rank - 1) << 1) * blocksize; // (rank - 1) * 2
	G.block(0, 0, blocksize, (blocksize << 1)) = G_schur.block(start_rowindice_remote, start_colindice_remote, blocksize, (blocksize << 1));

	// Upper right single block of process-local G
	int start_colindice = partition_blocksize * blocksize;
	start_colindice_remote += (blocksize << 1);
	G.block(0, start_colindice, blocksize, blocksize) = G_schur.block(start_rowindice_remote, start_colindice_remote, blocksize, blocksize);

	// Lower left single block of process-local G
	int start_rowindice = (partition_blocksize - 1) * blocksize;
	start_colindice = blocksize;
	start_rowindice_remote += blocksize;
	start_colindice_remote -= blocksize;
	G.block(start_rowindice, start_colindice, blocksize, blocksize) = G_schur.block(start_rowindice_remote, start_colindice_remote, blocksize, blocksize);

	// Lower right double block of process-local G
	start_colindice = partition_blocksize * blocksize;
	start_colindice_remote += blocksize;
	G.block(start_rowindice, start_colindice, blocksize, (blocksize << 1)) = G_schur.block(start_rowindice_remote, start_colindice_remote, blocksize, (blocksize << 1));
    }
    if(rank == partitions - 1) {
	int start_rowindice_remote = (1 + ((partitions - 2) << 1)) * blocksize;
	int start_colindice_remote = start_rowindice_remote - blocksize;
	G.block(0, 0, blocksize, (blocksize << 1)) = G_schur.block(start_rowindice_remote, start_colindice_remote, blocksize, (blocksize << 1));
    }
    // End of writeback of reduced inverse to full G partitions


    // Start of produce_schur
    //std::cout << "Process " << rank << " is producing blockrows " << start_blockrow << " to " << start_blockrow + partition_blocksize - 1 << std::endl;

    if(rank == 0) {
	produceSchurTopLeftCorner(processA, L, U, G, 0, partition_blocksize, blocksize);
    }
    if(rank > 0 && rank < partitions - 1) {
	produceSchurCentral_2(processA, L, U, G, partition_blocksize, blocksize);
    }
    if(rank == partitions - 1) {
	produceSchurBottomRightCorner_2(processA, L, U, G, partition_blocksize, blocksize);
    }
    // End of produce_schur

    // Start of reconstructing Tridiagonal system of the full inverse via MPIALLGATHER    
    Eigen::MatrixXcd G_final = Eigen::MatrixXcd(N, N);
    G_final.setZero();

    comm_buf_size = (rowSizePartition * colSizePartition * partitions) << 1;
    comm_buf = new double[comm_buf_size];
    in_buf_size = (rowSizePartition * colSizePartition) << 1;
    // !!!!!!
    // Attention I am currently purposefully overshooting the boundaries of G.data() for processes 0 and partitions - 1 
    // in the MPI_Allgather i.e. for those processes G.data() has only size ((rowSizePartition * (colSizePartition - blocksize)) << 1) insted of in_buf_size.
    // But since I am not writing anything back from that overshoot area it doesn't impact correctness currently
    // !!!!!
    in_buf = (double*) G.data();
    MPI_Allgather(in_buf, in_buf_size, MPI_DOUBLE, comm_buf, in_buf_size, MPI_DOUBLE, MPI_COMM_WORLD);

    
    for(int i = 0; i < partitions; ++i) {
	int start_rowindice = i * partition_blocksize * blocksize;
	int start_colindice = 0;
	if(i > 0) {
		start_colindice = start_rowindice - blocksize;
	}
	int rowSize = rowSizePartition;
	int colSize = colSizePartition;
	if(i == 0 || i == partitions - 1) {
		colSize -= blocksize;
	}
	G_final.block(start_rowindice, start_colindice, rowSize, colSize) =
			Eigen::Map<Eigen::MatrixXcd> ( (std::complex<double>*) (comm_buf + (i * in_buf_size)), rowSize, colSize);

	// Setting the off Tridiagonal blocks used in the produceSchurCentral step to 0 
	if(i > 0 && i < partitions - 1) {
		(G_final.block(start_rowindice, start_colindice + 3 * blocksize, blocksize, (partition_blocksize - 1) * blocksize)).setZero();
		(G_final.block(start_rowindice + 2 * blocksize, start_colindice + blocksize, (partition_blocksize - 1) * blocksize, blocksize)).setZero();
	}

    }

    delete[] comm_buf;
    // End of reconstructing Tridiagonal system of the full inverse via MPIALLGATHER    

    if(compare_reference){
        compareSINV_referenceInverse_byblock(n_blocks,
                                     blocksize,
                                     G_final,
                                     full_inverse,
                                     rank
        );
    }
    return G_final;
}


Eigen::MatrixXcd psr_solve_customMPI(int N,
                             int blocksize,
                             int n_blocks,
                             int partitions,
                             int partition_blocksize,
                             int rank,
                             int n_blocks_schursystem,
                             Eigen::MatrixXcd& eigenA_read_in,
                             bool compare_reference
){  

    Eigen::MatrixXcd eigenA2 = eigenA_read_in;

    // Referece inverse
    Eigen::MatrixXcd full_inverse;
    if(compare_reference){
        full_inverse = eigenA_read_in.inverse();
    }
	
    //Limit it to the processes partition of A
    int start_blockrow = rank * partition_blocksize;
    int rowSizePartition = partition_blocksize * blocksize;
    int colSizePartition = (partition_blocksize + 2) * blocksize;
    Eigen::MatrixXcd processA;
    Eigen::MatrixXcd G,L,U;

    if (rank == 0) {
        processA = eigenA2.block(0, 0, rowSizePartition, colSizePartition-blocksize);
        G = Eigen::MatrixXcd::Zero(rowSizePartition, colSizePartition-blocksize);
    }

    if (rank > 0 && rank < partitions - 1) {
        int startRowIndex = start_blockrow*blocksize;
        int startColIndex = (start_blockrow-1)*blocksize;
        processA = eigenA2.block(startRowIndex, startColIndex, rowSizePartition, colSizePartition);
        G = Eigen::MatrixXcd::Zero(rowSizePartition, colSizePartition);
    }
    if (rank == partitions - 1) {
        int startRowIndex = start_blockrow*blocksize;
        int startColIndex = (start_blockrow-1)*blocksize;
        processA = eigenA2.block(startRowIndex, startColIndex, rowSizePartition, colSizePartition-blocksize);
        G = Eigen::MatrixXcd::Zero(rowSizePartition, colSizePartition-blocksize);
    }


    // Start reduce_schur
    //std::cout << "Process " << rank << " is reducing blockrows " << start_blockrow << " to " << start_blockrow + partition_blocksize - 1 << std::endl;


    // ----- Start timing -----
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();



    if (rank == 0){
        auto result = reduce_schur_topleftcorner(processA, 0, partition_blocksize, blocksize);
        L = std::get<0>(result);
        U = std::get<1>(result);
    }

    if (rank > 0 && rank < partitions - 1){
        auto result = reduce_schur_central_2(processA, partition_blocksize, blocksize);
        L = std::get<0>(result);
        U = std::get<1>(result);
    }
    
    if (rank == partitions - 1){
        auto result = reduce_schur_bottomrightcorner_2(processA, partition_blocksize, blocksize);
        L = std::get<0>(result);
        U = std::get<1>(result);
    }
    // End reduce_schur

    // Start of MPIALLGATHER for reduced_schur_system and inverse of said system
    Eigen::MatrixXcd A_schur = Eigen::MatrixXcd(blocksize*n_blocks_schursystem, blocksize*n_blocks_schursystem);
    A_schur.setZero();

    // Start of creating custom MPI datatypes for block sends
    MPI_Datatype subblockType;
    create_subblock_Type(&subblockType, rowSizePartition, blocksize, 1);

    MPI_Datatype subblockType_2;
    create_subblock_Type(&subblockType_2, rowSizePartition, blocksize, 2);

    MPI_Datatype subblock_ReceiveType;
    create_subblock_Type(&subblock_ReceiveType, blocksize, blocksize, 1);

    MPI_Datatype redschur_blockpatternType;
    if (rank == 0){
        create_ul2_redschur_blockpattern_Type(&redschur_blockpatternType, subblockType_2, blocksize, rowSizePartition, partition_blocksize);
    }
    else if (rank == partitions - 1){
        create_br2_redschur_blockpattern_Type(&redschur_blockpatternType, subblockType_2, blocksize, rowSizePartition, partition_blocksize);
    }
    else{
        create_central_redschur_blockpattern_Type(&redschur_blockpatternType, subblockType, subblockType_2,  blocksize, rowSizePartition, partition_blocksize);
    }

    // End of creating custom MPI datatypes for block sends


    // Perform the same Allgather with custom MPI Datatypes
    unsigned long comm_custom_buf_size = (blocksize * blocksize * ((partitions - 2) * 6 + 2 * 2));
    std::complex<double>* comm_custom_buf = new std::complex<double>[comm_custom_buf_size];
    int *receivecounts = new int[partitions];
    int *displs = new int[partitions];


    for(int i = 0; i < partitions; ++i) {
        if (i == 0){
            receivecounts[i] = 2;
            displs[i] = 0;
        }
        else if (i == partitions - 1){
            receivecounts[i] = 2;
            displs[i] = (displs[i-1] + receivecounts[i-1]);
        }
        else{
            receivecounts[i] = 6;
            displs[i] = (displs[i-1] + receivecounts[i-1]);
        }
    }

    MPI_Allgatherv(processA.data(), 1, redschur_blockpatternType, comm_custom_buf, receivecounts, displs,  subblock_ReceiveType, MPI_COMM_WORLD);
    
    unsigned long in_buf_size = (blocksize * blocksize * 6) << 1;
    fill_reduced_schur_matrix_cd(A_schur, comm_custom_buf, in_buf_size, blocksize, partitions, rank);

    delete[] comm_custom_buf;

    auto G_schur = A_schur.inverse();
    // End of MPIALLGATHER for reduced_schur_system and inverse of said system

    // Start of writeback of reduced inverse to full G partitions
    if(rank == 0) {
        int start_rowindice = (partition_blocksize - 1) * blocksize;
        int start_colindice = (partition_blocksize - 1) * blocksize;
        G.block(start_rowindice, start_colindice, blocksize, (blocksize << 1)) = G_schur.block(0, 0, blocksize, (blocksize << 1));	
    }
    if(rank > 0 && rank < partitions - 1) {
	// Upper left double block of process-local G
        int start_rowindice_remote = (1 + ((rank - 1) << 1)) * blocksize; // (rank - 1) * 2 + 1
        int start_colindice_remote = ((rank - 1) << 1) * blocksize; // (rank - 1) * 2
        G.block(0, 0, blocksize, (blocksize << 1)) = G_schur.block(start_rowindice_remote, start_colindice_remote, blocksize, (blocksize << 1));

        // Upper right single block of process-local G
        int start_colindice = partition_blocksize * blocksize;
        start_colindice_remote += (blocksize << 1);
        G.block(0, start_colindice, blocksize, blocksize) = G_schur.block(start_rowindice_remote, start_colindice_remote, blocksize, blocksize);

        // Lower left single block of process-local G
        int start_rowindice = (partition_blocksize - 1) * blocksize;
        start_colindice = blocksize;
        start_rowindice_remote += blocksize;
        start_colindice_remote -= blocksize;
        G.block(start_rowindice, start_colindice, blocksize, blocksize) = G_schur.block(start_rowindice_remote, start_colindice_remote, blocksize, blocksize);

        // Lower right double block of process-local G
        start_colindice = partition_blocksize * blocksize;
        start_colindice_remote += blocksize;
        G.block(start_rowindice, start_colindice, blocksize, (blocksize << 1)) = G_schur.block(start_rowindice_remote, start_colindice_remote, blocksize, (blocksize << 1));
    }
    if(rank == partitions - 1) {
        int start_rowindice_remote = (1 + ((partitions - 2) << 1)) * blocksize;
        int start_colindice_remote = start_rowindice_remote - blocksize;
        G.block(0, 0, blocksize, (blocksize << 1)) = G_schur.block(start_rowindice_remote, start_colindice_remote, blocksize, (blocksize << 1));
    }
    // End of writeback of reduced inverse to full G partitions


    // Start of produce_schur
    //std::cout << "Process " << rank << " is producing blockrows " << start_blockrow << " to " << start_blockrow + partition_blocksize - 1 << std::endl;

    if(rank == 0) {
	    produceSchurTopLeftCorner(processA, L, U, G, 0, partition_blocksize, blocksize);
    }
    if(rank > 0 && rank < partitions - 1) {
	    produceSchurCentral_2(processA, L, U, G, partition_blocksize, blocksize);
    }
    if(rank == partitions - 1) {
	    produceSchurBottomRightCorner_2(processA, L, U, G, partition_blocksize, blocksize);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    // ----- End timing -----

    double elapsed_time = end_time - start_time;

    if(compare_reference == false){
        if(rank == 0) {
            std::cout << " ..took: " << elapsed_time << " s" << std::endl;
        }

        // Write the elapsed time to a file
        if(rank == 0) {
            std::ofstream time_file;

            // Format a string for the name of the file using the blocksize and the number of blocks
            std::string filename = "PSR_CPU_bs" + std::to_string(blocksize) + "_nb" + std::to_string(n_blocks) + "_world" + std::to_string(partitions) + ".txt";

            time_file.open(filename, std::ios::app);
            time_file << "Elapsed time: " << elapsed_time << " s" << std::endl;
            time_file.close();
        }
    }
    

    // End of produce_schur


    if(compare_reference){
        compareSINV_referenceInverse_localprodG_byblock(partitions, blocksize,
                                                         partition_blocksize, G, full_inverse, rank);
    }

    
    // // Start of reconstructing Tridiagonal system of the full inverse via MPIALLGATHER   
    // This is not necessary as G does not need to be completely reconstructed on each process 
    // Eigen::MatrixXcd G_final = Eigen::MatrixXcd(N, N);
    // G_final.setZero();

    // int comm_buf_size = (rowSizePartition * colSizePartition * partitions) << 1;
    // double* comm_buf = new double[comm_buf_size];
    // in_buf_size = (rowSizePartition * colSizePartition) << 1;
    // // !!!!!!
    // // Attention I am currently purposefully overshooting the boundaries of G.data() for processes 0 and partitions - 1 
    // // in the MPI_Allgather i.e. for those processes G.data() has only size ((rowSizePartition * (colSizePartition - blocksize)) << 1) insted of in_buf_size.
    // // But since I am not writing anything back from that overshoot area it doesn't impact correctness currently
    // // !!!!!
    // double* in_buf = (double*) G.data();
    // MPI_Allgather(in_buf, in_buf_size, MPI_DOUBLE, comm_buf, in_buf_size, MPI_DOUBLE, MPI_COMM_WORLD);

    
    // for(int i = 0; i < partitions; ++i) {
    //     int start_rowindice = i * partition_blocksize * blocksize;
    //     int start_colindice = 0;
    //     if(i > 0) {
    //         start_colindice = start_rowindice - blocksize;
    //     }
    //     int rowSize = rowSizePartition;
    //     int colSize = colSizePartition;
    //     if(i == 0 || i == partitions - 1) {
    //         colSize -= blocksize;
    //     }
    //     G_final.block(start_rowindice, start_colindice, rowSize, colSize) =
    //             Eigen::Map<Eigen::MatrixXcd> ( (std::complex<double>*) (comm_buf + (i * in_buf_size)), rowSize, colSize);

    //     // Setting the off Tridiagonal blocks used in the produceSchurCentral step to 0 
    //     if(i > 0 && i < partitions - 1) {
    //         (G_final.block(start_rowindice, start_colindice + 3 * blocksize, blocksize, (partition_blocksize - 1) * blocksize)).setZero();
    //         (G_final.block(start_rowindice + 2 * blocksize, start_colindice + blocksize, (partition_blocksize - 1) * blocksize, blocksize)).setZero();
    //     }

    // }

    // delete[] comm_buf;
    // // End of reconstructing Tridiagonal system of the full inverse via MPIALLGATHER    

    // if(compare_reference){
    //     compareSINV_referenceInverse_byblock(n_blocks,
    //                                  blocksize,
    //                                  G_final,
    //                                  full_inverse,
    //                                  rank
    //     );
    // }


    MPI_Type_free(&subblockType);
    MPI_Type_free(&subblockType_2);
    MPI_Type_free(&subblock_ReceiveType);
    MPI_Type_free(&redschur_blockpatternType);

    return G;
}


Eigen::MatrixXcd psr_solve_customMPI_gpu(int N,
                             int blocksize,
                             int n_blocks,
                             int partitions,
                             int partition_blocksize,
                             int rank,
                             int n_blocks_schursystem,
                             Eigen::MatrixXcd& eigenA_read_in,
                             bool compare_reference
){  
    // Initialize a cuda stream
    cudaStream_t stream = NULL;
    cudaErrchk(cudaStreamCreate(&stream));

    cusolverDnHandle_t cusolver_handle;
    cusolverErrchk(cusolverDnCreate(&cusolver_handle));
    cusolverErrchk(cusolverDnSetStream(cusolver_handle, stream));

    cublasHandle_t cublas_handle;
    cublasErrchk(cublasCreate(&cublas_handle));
    cublasErrchk(cublasSetStream(cublas_handle, stream));


    // Referece inverse
    Eigen::MatrixXcd full_inverse;
    if(compare_reference){
        full_inverse = eigenA_read_in.inverse();
    }
	
    //Limit it to the processes partition of A
    int start_blockrow = rank * partition_blocksize;
    int rowSizePartition = partition_blocksize * blocksize;
    int colSizePartition = (partition_blocksize + 2) * blocksize;
    Eigen::MatrixXcd processA;
    Eigen::MatrixXcd G,L,U;

    if (rank == 0) {
        processA = eigenA_read_in.block(0, 0, rowSizePartition, colSizePartition-blocksize);
        G = Eigen::MatrixXcd::Zero(rowSizePartition, colSizePartition-blocksize);
    }

    if (rank > 0 && rank < partitions - 1) {
        int startRowIndex = start_blockrow*blocksize;
        int startColIndex = (start_blockrow-1)*blocksize;
        processA = eigenA_read_in.block(startRowIndex, startColIndex, rowSizePartition, colSizePartition);
        G = Eigen::MatrixXcd::Zero(rowSizePartition, colSizePartition);
    }
    if (rank == partitions - 1) {
        int startRowIndex = start_blockrow*blocksize;
        int startColIndex = (start_blockrow-1)*blocksize;
        processA = eigenA_read_in.block(startRowIndex, startColIndex, rowSizePartition, colSizePartition-blocksize);
        G = Eigen::MatrixXcd::Zero(rowSizePartition, colSizePartition-blocksize);
    }


    // Loading of Matrices to the GPU
    // Since cuda has only Col-Major and our Eigen Matrices also use Col-Major
    int l_dim = processA.rows();

    // Initialize Idendity Matrix and a Copy of it which is also used as a buffer for the result of inversions.
    // create right hand side identity matrix
    // std::complex<double>* identity_h;
    // cudaErrchk(cudaMallocHost((void**)&identity_h, blocksize * blocksize * sizeof(std::complex<double>)));

    // for(unsigned int i = 0; i < blocksize * blocksize; i++){
    //     identity_h[i] = 0.0;
    //     if(i / blocksize == i % blocksize){
    //         identity_h[i] = 1.0;
    //     }
    // }

    // init right hand side identity matrix on device as a constant to be copied for later inversion 
    cuDoubleComplex* identity_d = NULL;

    cudaErrchk(cudaMalloc((void**)&identity_d, blocksize * blocksize * sizeof(cuDoubleComplex)));
    create_identity_GPU(identity_d, blocksize);

    // Allocate memory for the input Matrix and work Matrix
    cuDoubleComplex* A_gpu = NULL;

    cudaErrchk(cudaMalloc((void**) &A_gpu, processA.rows() * processA.cols() * sizeof(cuDoubleComplex)));

    // Load input Matrix onto GPU
    cudaErrchk(cudaMemcpy(A_gpu, reinterpret_cast<const cuDoubleComplex*>(processA.data()), processA.rows() * processA.cols() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // Allocate memory for the L and U Matrices
    cuDoubleComplex* L_gpu = NULL;
    cuDoubleComplex* U_gpu = NULL;

    cudaErrchk(cudaMalloc((void**) &L_gpu, processA.rows() * processA.cols() * sizeof(cuDoubleComplex)));
    cudaErrchk(cudaMalloc((void**) &U_gpu, processA.rows() * processA.cols() * sizeof(cuDoubleComplex)));


    // Start reduce_schur
    //std::cout << "Process " << rank << " is reducing blockrows " << start_blockrow << " to " << start_blockrow + partition_blocksize - 1 << std::endl;


    // ----- Start timing -----
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();


    L = Eigen::MatrixXcd::Zero(processA.rows(), processA.cols());
    U = Eigen::MatrixXcd::Zero(processA.rows(), processA.cols());

    if (rank == 0){
        reduce_schur_topleftcorner_gpu(partition_blocksize, blocksize, stream, cusolver_handle, cublas_handle, A_gpu, L_gpu, U_gpu, identity_d, l_dim);
    }

    if (rank > 0 && rank < partitions - 1){
        reduce_schur_central_gpu(partition_blocksize, blocksize, stream, cusolver_handle, cublas_handle, A_gpu, L_gpu, U_gpu, identity_d, l_dim);
    }
    
    if (rank == partitions - 1){
        reduce_schur_bottomrightcorner_gpu(partition_blocksize, blocksize, stream, cusolver_handle, cublas_handle, A_gpu, L_gpu, U_gpu, identity_d, l_dim);
    }

    cudaErrchk(cudaMemcpy(processA.data(), reinterpret_cast<std::complex<double>*>(A_gpu), processA.rows() * processA.cols() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));
    // L and U Matrices on the host are currently still needed for process_schur steps
    cudaErrchk(cudaMemcpy(L.data(), reinterpret_cast<std::complex<double>*>(L_gpu), processA.rows() * processA.cols() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));
    cudaErrchk(cudaMemcpy(U.data(), reinterpret_cast<std::complex<double>*>(U_gpu), processA.rows() * processA.cols() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));
    // End reduce_schur

    // Start of MPIALLGATHER for reduced_schur_system and inverse of said system
    Eigen::MatrixXcd A_schur = Eigen::MatrixXcd(blocksize*n_blocks_schursystem, blocksize*n_blocks_schursystem);
    A_schur.setZero();

    // Start of creating custom MPI datatypes for block sends
    MPI_Datatype subblockType;
    create_subblock_Type(&subblockType, rowSizePartition, blocksize, 1);

    MPI_Datatype subblockType_2;
    create_subblock_Type(&subblockType_2, rowSizePartition, blocksize, 2);

    MPI_Datatype subblock_ReceiveType;
    create_subblock_Type(&subblock_ReceiveType, blocksize, blocksize, 1);

    MPI_Datatype redschur_blockpatternType;
    if (rank == 0){
        create_ul2_redschur_blockpattern_Type(&redschur_blockpatternType, subblockType_2, blocksize, rowSizePartition, partition_blocksize);
    }
    else if (rank == partitions - 1){
        create_br2_redschur_blockpattern_Type(&redschur_blockpatternType, subblockType_2, blocksize, rowSizePartition, partition_blocksize);
    }
    else{
        create_central_redschur_blockpattern_Type(&redschur_blockpatternType, subblockType, subblockType_2,  blocksize, rowSizePartition, partition_blocksize);
    }

    // End of creating custom MPI datatypes for block sends


    // Perform the same Allgather with custom MPI Datatypes
    unsigned long comm_custom_buf_size = (blocksize * blocksize * ((partitions - 2) * 6 + 2 * 2));\
    std::complex<double>* comm_custom_buf = NULL;
    cudaMallocHost((void**)&comm_custom_buf, comm_custom_buf_size * sizeof(complex_h));
    //std::complex<double>* comm_custom_buf = new std::complex<double>[comm_custom_buf_size];
    int *receivecounts = new int[partitions];
    int *displs = new int[partitions];


    for(int i = 0; i < partitions; ++i) {
        if (i == 0){
            receivecounts[i] = 2;
            displs[i] = 0;
        }
        else if (i == partitions - 1){
            receivecounts[i] = 2;
            displs[i] = (displs[i-1] + receivecounts[i-1]);
        }
        else{
            receivecounts[i] = 6;
            displs[i] = (displs[i-1] + receivecounts[i-1]);
        }
    }

    MPI_Allgatherv(processA.data(), 1, redschur_blockpatternType, comm_custom_buf, receivecounts, displs,  subblock_ReceiveType, MPI_COMM_WORLD);
    // End of MPIALLGATHER for reduced_schur_system and inverse of said system


    // Start of Schur Inversion on GPU

    // // create and read in GPU A_schur
    // cuDoubleComplex* A_schur_gpu = NULL;

    // cudaErrchk(cudaMalloc((void**)&A_schur_gpu, (n_blocks_schursystem * blocksize) * (n_blocks_schursystem * blocksize) * sizeof(cuDoubleComplex)));
    // cudaErrchk(cudaMemset(A_schur_gpu, 0, (n_blocks_schursystem * blocksize) * (n_blocks_schursystem * blocksize) * sizeof(cuDoubleComplex)));

    // fill_reduced_schur_matrix_gpu(A_schur_gpu, comm_custom_buf, blocksize, partitions, 0);



    // Allocation of buffers for GPU inversion
    cuDoubleComplex* G_schur_gpu = NULL;
    cudaErrchk(cudaMalloc((void**)&G_schur_gpu, (n_blocks_schursystem * blocksize) * (n_blocks_schursystem * blocksize) * sizeof(cuDoubleComplex)));
    // int info_h = 0;
    // int *ipiv_d = NULL;
    // int *info_d = NULL;
    // cudaErrchk(cudaMalloc((void**)&info_d, sizeof(int)))
    // cudaErrchk(cudaMalloc((void**)&ipiv_d, n_blocks_schursystem * blocksize *sizeof(int)));


    // cuDoubleComplex *buffer = NULL;
    // int bufferSize = 0;
    // cusolverErrchk(cusolverDnZgetrf_bufferSize(cusolver_handle, n_blocks_schursystem * blocksize, n_blocks_schursystem * blocksize,
    //                                            A_schur_gpu, n_blocks_schursystem * blocksize,
	// 				       &bufferSize));
    // cudaErrchk(cudaMalloc((void**)&buffer, sizeof(cuDoubleComplex) * bufferSize));

    // // Initialize G_schur_gpu to the identity matrix
    // cuDoubleComplex* G_schur_gpu = NULL;
    // cudaErrchk(cudaMalloc((void**)&G_schur_gpu, (n_blocks_schursystem * blocksize) * (n_blocks_schursystem * blocksize) * sizeof(cuDoubleComplex)));

    // cudaErrchk(cudaMemset(G_schur_gpu, 0, (n_blocks_schursystem * blocksize) * (n_blocks_schursystem * blocksize) * sizeof(cuDoubleComplex)));

    // create_identity_GPU(reinterpret_cast<cuDoubleComplex*>(G_schur_gpu), n_blocks_schursystem * blocksize);

    // // LU factorization of A_schur_gpu
    // cudaErrchk(cudaStreamSynchronize(stream));
    // cusolverErrchk(cusolverDnZgetrf(cusolver_handle, n_blocks_schursystem * blocksize, n_blocks_schursystem * blocksize,
    // 				    A_schur_gpu, n_blocks_schursystem * blocksize, buffer, ipiv_d, info_d));
    
    // cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));
    
    // if (info_h != 0) {
    // 	std::printf("Error: LU factorization failed\n");
    // 	std::printf("info_h = %d\n", info_h);
    // }

    // cusolverErrchk(cusolverDnZgetrs(cusolver_handle, CUBLAS_OP_N, n_blocks_schursystem * blocksize, n_blocks_schursystem * blocksize,
    // 				    A_schur_gpu, n_blocks_schursystem * blocksize, ipiv_d,
    // 				    reinterpret_cast<cuDoubleComplex*>(G_schur_gpu), n_blocks_schursystem * blocksize, info_d)); 

    
    // cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));

    
    // if (info_h != 0) {
    // 	std::printf("Error: Inversion failed\n");
    // 	std::printf("info_h = %d\n", info_h);
    // } 

    // if(ipiv_d) {
	// cudaErrchk(cudaFree(ipiv_d));
    // }
    // if(info_d) {
	// cudaErrchk(cudaFree(info_d));
    // }
    // if(A_schur_gpu) {
	// cudaErrchk(cudaFree(A_schur_gpu));
    // }
    // if(buffer) {
	// cudaErrchk(cudaFree(buffer));

    //std::complex<double>* g_host_rgf_buf = new std::complex<double>[comm_custom_buf_size];
    std::complex<double>* g_host_rgf_buf = NULL;
    cudaMallocHost((void**)&g_host_rgf_buf, comm_custom_buf_size * sizeof(complex_h));
    rgf_for_subsystem(blocksize, n_blocks_schursystem*blocksize, comm_custom_buf, g_host_rgf_buf);
    
    //invert_GPU_matrix_complete(G_schur_gpu, A_schur_gpu, n_blocks_schursystem * blocksize, cusolver_handle);
    //std::complex<double>* host_testblock_complete = new std::complex<double>[blocksize*blocksize];
    //copy_rowblocks_GPU2buffer(G_schur_gpu, reinterpret_cast<cuDoubleComplex*>(host_testblock_complete), blocksize,  n_blocks_schursystem*blocksize, 1, n_blocks_schursystem-1, n_blocks_schursystem-1, 0);
    if(comm_custom_buf){
        cudaFreeHost(comm_custom_buf);
    }
    // delete[] comm_custom_buf;

    //Eigen::MatrixXcd testblock_complete = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(host_testblock_complete, blocksize, blocksize);
    //Eigen::MatrixXcd testblock_rgf = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(g_host_rgf_buf + 3*(n_blocks_schursystem-1) * blocksize * blocksize, blocksize, blocksize);
    // if(rank == 0){
    //     std::cout << testblock_complete << std::endl;
    //     std::cout << testblock_rgf << std::endl;
    // }
    //std::complex<double>* host_testblock_rgf = new std::complex<double>[blocksize*blocksize];
    // End of Schur Inversion on GPU
    fill_reduced_schur_matrix_gpu(G_schur_gpu, g_host_rgf_buf, blocksize, partitions, 0);
    //

    // Initialize G on the GPU as the Zero Matrix
    cuDoubleComplex* G_gpu = NULL;

    cudaErrchk(cudaMalloc((void**)&G_gpu, processA.rows() * processA.cols() * sizeof(cuDoubleComplex)));
    cudaErrchk(cudaMemset(G_gpu, 0, processA.rows() * processA.cols() * sizeof(cuDoubleComplex)));


    // Start of writeback of reduced inverse to full G partitions
    int stride2 = (n_blocks_schursystem * blocksize); // Stride of the reduced matrix
    if(rank == 0) {
	    int colBlock1 = (partition_blocksize - 1);
	    int rowBlock1 = (partition_blocksize - 1);
        int colBlock2 = 0;
        int rowBlock2 = 0;
        int rowBlocks = 2;
        // for(int j = 0; j < 2 * blocksize; ++j) {
        //     cudaErrchk(cudaMemcpy(G_gpu+colsSkip+rowOffset+j*l_dim, G_schur_gpu+j*(n_blocks_schursystem*blocksize), blocksize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
        // }
        copy_rowblocks_GPU2GPU(G_gpu, G_schur_gpu, blocksize, l_dim, stride2, rowBlocks, rowBlock1,  colBlock1, rowBlock2, colBlock2);
    }
    if(rank > 0 && rank < partitions - 1) {
        // Upper left double block of process-local G
        int schur_l_dim = n_blocks_schursystem * blocksize;
        int colsSkip_schur = ((rank - 1) << 1) * blocksize * schur_l_dim;
        int rowOffset_schur = (1 + ((rank - 1) << 1)) * blocksize;
        int colBlock1 = 0;
        int rowBlock1 = 0;
        int colBlock2 = 2 * (rank - 1);
        int rowBlock2 = 1 + 2 * (rank - 1);
        int rowBlocks = 2;
        // for(int j = 0; j < 2 * blocksize; ++j) {
        //     cudaErrchk(cudaMemcpy(G_gpu+j*l_dim, G_schur_gpu+colsSkip_schur+rowOffset_schur+j*schur_l_dim, blocksize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
        // }
        copy_rowblocks_GPU2GPU(G_gpu, G_schur_gpu, blocksize, l_dim, stride2, rowBlocks, rowBlock1,  colBlock1, rowBlock2, colBlock2);

        // Upper right single block of process-local G
        colsSkip_schur += (blocksize << 1) * schur_l_dim;
        int colsSkip = partition_blocksize * blocksize * l_dim;

        colBlock2 += 2;
        colBlock1 += partition_blocksize;
        rowBlocks = 1;

        // for(int j = 0; j < blocksize; ++j) {
        //     cudaErrchk(cudaMemcpy(G_gpu+colsSkip+j*l_dim, G_schur_gpu+colsSkip_schur+rowOffset_schur+j*schur_l_dim, blocksize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
        // }

        copy_rowblocks_GPU2GPU(G_gpu, G_schur_gpu, blocksize, l_dim, stride2, rowBlocks, rowBlock1,  colBlock1, rowBlock2, colBlock2);

        // Lower left single block of process-local G
        int rowOffset = (partition_blocksize - 1) * blocksize;
        colsSkip = blocksize * l_dim;
        rowOffset_schur += blocksize;
        colsSkip_schur -= blocksize * schur_l_dim;

        rowBlock1 += partition_blocksize -1;
        rowBlock2 += 1;

        colBlock1 = 1;
        colBlock2 -=1;

        // for(int j = 0; j < blocksize; ++j) {
        //     cudaErrchk(cudaMemcpy(G_gpu+colsSkip+rowOffset+j*l_dim, G_schur_gpu+colsSkip_schur+rowOffset_schur+j*schur_l_dim, blocksize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
        // }
        copy_rowblocks_GPU2GPU(G_gpu, G_schur_gpu, blocksize, l_dim, stride2, rowBlocks, rowBlock1,  colBlock1, rowBlock2, colBlock2);

        // Lower right double block of process-local G
        colsSkip = partition_blocksize * blocksize * l_dim;
        colsSkip_schur += blocksize * schur_l_dim;

        colBlock1 = partition_blocksize;
        colBlock2 += 1;
        rowBlocks = 2;
        // for(int j = 0; j < 2 * blocksize; ++j) {
        //     cudaErrchk(cudaMemcpy(G_gpu+colsSkip+rowOffset+j*l_dim, G_schur_gpu+colsSkip_schur+rowOffset_schur+j*schur_l_dim, blocksize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
        // }
        copy_rowblocks_GPU2GPU(G_gpu, G_schur_gpu, blocksize, l_dim, stride2, rowBlocks, rowBlock1,  colBlock1, rowBlock2, colBlock2);
	
    }
    if(rank == partitions - 1) {
        int schur_l_dim = n_blocks_schursystem * blocksize;
        int rowOffset_schur = (1 + ((partitions - 2) << 1)) * blocksize;
        int colsSkip_schur = (rowOffset_schur - blocksize) * schur_l_dim;

        int colBlock1 = 0;
        int rowBlock1 = 0;
        int rowBlock2 = 1 + 2*(partitions - 2);
        int colBlock2 = (rowBlock2 - 1);

        int rowBlocks = 2;
        // for(int j = 0; j < 2 * blocksize; ++j) {
        //     cudaErrchk(cudaMemcpy(G_gpu+j*l_dim, G_schur_gpu+colsSkip_schur+rowOffset_schur+j*schur_l_dim, blocksize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
        // }
        copy_rowblocks_GPU2GPU(G_gpu, G_schur_gpu, blocksize, l_dim, stride2, rowBlocks, rowBlock1,  colBlock1, rowBlock2, colBlock2);
    }
    // End of writeback of reduced inverse to full G partitions

    // Writeback G from gpu, still needed for current produce step handled by the Host
    //cudaErrchk(cudaMemcpy(G.data(), reinterpret_cast<std::complex<double>*>(G_gpu), processA.rows() * processA.cols() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));
    

    // Start of produce_schur
    //std::cout << "Process " << rank << " is producing blockrows " << start_blockrow << " to " << start_blockrow + partition_blocksize - 1 << std::endl;

    if(rank == 0) {
	    //produceSchurTopLeftCorner(processA, L, U, G, 0, partition_blocksize, blocksize);
        produceSchurTopLeftCorner_gpu(partition_blocksize, blocksize, stream, cusolver_handle, cublas_handle, A_gpu,  G_gpu, L_gpu, U_gpu, identity_d, l_dim);
        cudaErrchk(cudaMemcpy(G.data(), reinterpret_cast<std::complex<double>*>(G_gpu), processA.rows() * processA.cols() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));
    }
    if(rank > 0 && rank < partitions - 1) {
	    //produceSchurCentral_2(processA, L, U, G, partition_blocksize, blocksize);
        produceSchurcentral_gpu(partition_blocksize, blocksize, stream, cusolver_handle, cublas_handle, A_gpu,  G_gpu, L_gpu, U_gpu, identity_d, l_dim);
        cudaErrchk(cudaMemcpy(G.data(), reinterpret_cast<std::complex<double>*>(G_gpu), processA.rows() * processA.cols() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));
    }
    if(rank == partitions - 1) {
	    //produceSchurBottomRightCorner_2(processA, L, U, G, partition_blocksize, blocksize);
        produceSchurBottomRightCorner_gpu(partition_blocksize, blocksize, stream, cusolver_handle, cublas_handle, A_gpu,  G_gpu, L_gpu, U_gpu, identity_d, l_dim);
        cudaErrchk(cudaMemcpy(G.data(), reinterpret_cast<std::complex<double>*>(G_gpu), processA.rows() * processA.cols() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    // ----- End timing -----

    double elapsed_time = end_time - start_time;

    if(compare_reference == false){
        if(rank == 0) {
            std::cout << " ..took: " << elapsed_time << " s" << std::endl;
        }

        // Write the elapsed time to a file
        if(rank == 0) {
            std::ofstream time_file;

            // Format a string for the name of the file using the blocksize and the number of blocks
            std::string filename = "PSR_GPU_bs" + std::to_string(blocksize) + "_nb" + std::to_string(n_blocks) + "_world" + std::to_string(partitions) + ".txt";

            time_file.open(filename, std::ios::app);
            time_file << "Elapsed time: " << elapsed_time << " s" << std::endl;
            time_file.close();
        }
    }
    

    // End of produce_schur
    
    
    
    if(compare_reference){
        compareSINV_referenceInverse_localprodG_byblock(partitions, blocksize,
                                                         partition_blocksize, G, full_inverse, rank);
    }

    if(stream) {
        cudaErrchk(cudaStreamDestroy(stream));
    }
    if(cublas_handle) {
        cublasErrchk(cublasDestroy(cublas_handle));
    }
    if(cusolver_handle) {
        cusolverErrchk(cusolverDnDestroy(cusolver_handle));
    }
    if(A_gpu) {
        cudaErrchk(cudaFree(A_gpu));
    }
    if(L_gpu) {
        cudaErrchk(cudaFree(L_gpu));
    }
    if(U_gpu) {
        cudaErrchk(cudaFree(U_gpu));
    }
    if(G_gpu) {
        cudaErrchk(cudaFree(G_gpu));
    }
    if(identity_d) {
        cudaErrchk(cudaFree(identity_d));
    }

    MPI_Type_free(&subblockType);
    MPI_Type_free(&subblockType_2);
    MPI_Type_free(&subblock_ReceiveType);
    MPI_Type_free(&redschur_blockpatternType);

    return G;
}
