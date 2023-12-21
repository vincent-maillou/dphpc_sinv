// Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
#include "batched_lesser_greater_retarded.h"

#define cudaErrchk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      std::printf("CUDAassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#define cusolverErrchk(ans) { cusolverAssert((ans), __FILE__, __LINE__); }
inline void cusolverAssert(cusolverStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUSOLVER_STATUS_SUCCESS) 
   {
        //Did not find a counter part to cudaGetErrorString in cusolver
        std::printf("CUSOLVERassert: %s %s %d\n", cudaGetErrorString((cudaError_t)code), file, line);
        if (abort) exit(code);
   }
}


#define cublasErrchk(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUBLAS_STATUS_SUCCESS) 
   {
        //Did not find a counter part to cudaGetErrorString in cublas
        std::printf("CUBLASassert: %s %s %d\n", cudaGetErrorString((cudaError_t)code), file, line);
        if (abort) exit(code);
   }
}


void rgf_lesser_greater_batched(
    unsigned int blocksize,
    unsigned int matrix_size,
    unsigned int batch_size,
    complex_h **system_matrix_diagblk_h,
    complex_h **system_matrix_upperblk_h,
    complex_h **system_matrix_lowerblk_h,
    complex_h **self_energy_lesser_diagblk_h,
    complex_h **self_energy_lesser_upperblk_h,
    complex_h **self_energy_greater_diagblk_h,
    complex_h **self_energy_greater_upperblk_h,
    complex_h **lesser_inv_diagblk_h,
    complex_h **lesser_inv_upperblk_h,
    complex_h **greater_inv_diagblk_h,
    complex_h **greater_inv_upperblk_h,
    complex_h **retarded_inv_diagblk_h,
    complex_h **retarded_inv_upperblk_h,
    complex_h **retarded_inv_lowerblk_h)
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

    cudaEvent_t unload_lesser_greater_diag[n_blocks];
    cudaEvent_t unload_lesser_greater_upper[n_blocks];
    cudaEvent_t unload_retarded[n_blocks];
    for(unsigned int i = 0; i < n_blocks; i++){
        cudaErrchk(cudaEventCreate(&unload_lesser_greater_diag[i]))
        cudaErrchk(cudaEventCreate(&unload_lesser_greater_upper[i]))
        cudaErrchk(cudaEventCreate(&unload_retarded[i]))
        
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
    
    complex_d* system_matrix_diagblk_ptr_h[2][batch_size];
    complex_d* system_matrix_upperblk_ptr_h[2][batch_size];
    complex_d* system_matrix_lowerblk_ptr_h[2][batch_size];
    complex_d* self_energy_lesser_diagblk_ptr_h[2][batch_size];
    complex_d* self_energy_lesser_upperblk_ptr_h[2][batch_size];
    complex_d* self_energy_greater_diagblk_ptr_h[2][batch_size];
    complex_d* self_energy_greater_upperblk_ptr_h[2][batch_size];

    complex_d** system_matrix_diagblk_ptr_d[2];
    complex_d** system_matrix_upperblk_ptr_d[2];
    complex_d** system_matrix_lowerblk_ptr_d[2];
    complex_d** system_matrix_upper_lowerblk_ptr_d[2];

    complex_d** self_energy_lesser_diagblk_ptr_d[2];
    complex_d** self_energy_greater_diagblk_ptr_d[2];

    // first part is [0] and second part is [1]
    complex_d** system_matrix_diagblk_ptr_scuffed_d;
    complex_d** self_energy_lesser_diagblk_ptr_scuffed_d;


    complex_d** system_matrix_lowerblk_ptr_twice_d[2];
    complex_d** self_energy_lesser_diagblk_ptr_twice_d[2];
    complex_d** self_energy_greater_diagblk_ptr_twice_d[2];

    // first part is lesser and second part is greater
    complex_d** self_energy_lesser_greater_diagblk_ptr_d[2];
    complex_d** self_energy_lesser_greater_upperblk_ptr_d[2];

    // allocate single blocks of the matrix
    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&system_matrix_diagblk_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&system_matrix_upperblk_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&system_matrix_lowerblk_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&self_energy_lesser_diagblk_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&self_energy_lesser_upperblk_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&self_energy_greater_diagblk_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&self_energy_greater_upperblk_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));

        for(unsigned int j = 0; j < batch_size; j++){
            system_matrix_diagblk_ptr_h[i][j] = system_matrix_diagblk_d[i] + j * blocksize * blocksize;
            system_matrix_upperblk_ptr_h[i][j] = system_matrix_upperblk_d[i] + j * blocksize * blocksize;
            system_matrix_lowerblk_ptr_h[i][j] = system_matrix_lowerblk_d[i] + j * blocksize * blocksize;
            self_energy_lesser_diagblk_ptr_h[i][j] = self_energy_lesser_diagblk_d[i] + j * blocksize * blocksize;
            self_energy_lesser_upperblk_ptr_h[i][j] = self_energy_lesser_upperblk_d[i] + j * blocksize * blocksize;
            self_energy_greater_diagblk_ptr_h[i][j] = self_energy_greater_diagblk_d[i] + j * blocksize * blocksize;
            self_energy_greater_upperblk_ptr_h[i][j] = self_energy_greater_upperblk_d[i] + j * blocksize * blocksize;
        }
    }
    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&system_matrix_diagblk_ptr_d[i], batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMalloc((void**)&system_matrix_upperblk_ptr_d[i], batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMalloc((void**)&system_matrix_lowerblk_ptr_d[i], batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMalloc((void**)&system_matrix_upper_lowerblk_ptr_d[i], 2*batch_size * sizeof(complex_d*)));

        cudaErrchk(cudaMalloc((void**)&self_energy_lesser_diagblk_ptr_d[i], batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMalloc((void**)&self_energy_greater_diagblk_ptr_d[i], batch_size * sizeof(complex_d*)));

        cudaErrchk(cudaMalloc((void**)&system_matrix_lowerblk_ptr_twice_d[i], 2*batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMalloc((void**)&self_energy_lesser_diagblk_ptr_twice_d[i], 2*batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMalloc((void**)&self_energy_greater_diagblk_ptr_twice_d[i], 2*batch_size * sizeof(complex_d*)));


        cudaErrchk(cudaMalloc((void**)&self_energy_lesser_greater_diagblk_ptr_d[i], 2*batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMalloc((void**)&self_energy_lesser_greater_upperblk_ptr_d[i], 2*batch_size * sizeof(complex_d*)));

        cudaErrchk(cudaMemcpy(system_matrix_diagblk_ptr_d[i], system_matrix_diagblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(system_matrix_upperblk_ptr_d[i], system_matrix_upperblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(system_matrix_lowerblk_ptr_d[i], system_matrix_lowerblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(system_matrix_upper_lowerblk_ptr_d[i], system_matrix_upperblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(system_matrix_upper_lowerblk_ptr_d[i] + batch_size, system_matrix_lowerblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));


        cudaErrchk(cudaMemcpy(self_energy_lesser_diagblk_ptr_d[i], self_energy_lesser_diagblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(self_energy_greater_diagblk_ptr_d[i], self_energy_greater_diagblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    
        cudaErrchk(cudaMemcpy(system_matrix_lowerblk_ptr_twice_d[i], system_matrix_lowerblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(system_matrix_lowerblk_ptr_twice_d[i] + batch_size, system_matrix_lowerblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(self_energy_lesser_diagblk_ptr_twice_d[i], self_energy_lesser_diagblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(self_energy_lesser_diagblk_ptr_twice_d[i] + batch_size, self_energy_lesser_diagblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(self_energy_greater_diagblk_ptr_twice_d[i], self_energy_greater_diagblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(self_energy_greater_diagblk_ptr_twice_d[i] + batch_size, self_energy_greater_diagblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));

        cudaErrchk(cudaMemcpy(self_energy_lesser_greater_diagblk_ptr_d[i], self_energy_lesser_diagblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(self_energy_lesser_greater_diagblk_ptr_d[i] + batch_size, self_energy_greater_diagblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(self_energy_lesser_greater_upperblk_ptr_d[i], self_energy_lesser_upperblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(self_energy_lesser_greater_upperblk_ptr_d[i] + batch_size, self_energy_greater_upperblk_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    }
    cudaErrchk(cudaMalloc((void**)&system_matrix_diagblk_ptr_scuffed_d, 2*batch_size * sizeof(complex_d*)));
    cudaErrchk(cudaMemcpy(system_matrix_diagblk_ptr_scuffed_d, system_matrix_diagblk_ptr_h[0], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(system_matrix_diagblk_ptr_scuffed_d + batch_size, system_matrix_diagblk_ptr_h[1], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMalloc((void**)&self_energy_lesser_diagblk_ptr_scuffed_d, 2*batch_size * sizeof(complex_d*)));
    cudaErrchk(cudaMemcpy(self_energy_lesser_diagblk_ptr_scuffed_d, self_energy_lesser_diagblk_ptr_h[0], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(self_energy_lesser_diagblk_ptr_scuffed_d + batch_size, self_energy_lesser_diagblk_ptr_h[1], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));

    // allocate memory for the inverse
    complex_d* retarded_inv_diagblk_d = NULL;
    complex_d* retarded_inv_upperblk_d = NULL;
    complex_d* retarded_inv_lowerblk_d = NULL;
    complex_d* lesser_inv_diagblk_d = NULL;
    complex_d* lesser_inv_upperblk_d = NULL;
    complex_d* greater_inv_diagblk_d = NULL;
    complex_d* greater_inv_upperblk_d = NULL;
    complex_d* retarded_inv_diagblk_small_d[2];
    complex_d* lesser_inv_diagblk_small_d[2];
    complex_d* greater_inv_diagblk_small_d[2];

    complex_d* retarded_inv_diagblk_ptr_h[batch_size];
    complex_d* retarded_inv_upperblk_ptr_h[batch_size];
    complex_d* lesser_inv_diagblk_ptr_h[batch_size];
    complex_d* lesser_inv_upperblk_ptr_h[batch_size];
    complex_d* greater_inv_diagblk_ptr_h[batch_size];
    complex_d* greater_inv_upperblk_ptr_h[batch_size];
    complex_d* retarded_inv_diagblk_small_ptr_h[2][batch_size];
    complex_d* lesser_inv_diagblk_small_ptr_h[2][batch_size];
    complex_d* greater_inv_diagblk_small_ptr_h[2][batch_size];

    complex_d** retarded_inv_diagblk_ptr_d;
    complex_d** retarded_inv_upperblk_ptr_d;
    complex_d** retarded_inv_diagblk_ptr_twice_d;
    complex_d** retarded_inv_diagblk_small_ptr_d[2];
    complex_d** retarded_inv_diagblk_small_ptr_twice_d[2];

    complex_d** lesser_greater_inv_diagblk_small_ptr_d[2];

    complex_d** lesser_greater_inv_diagblk_ptr_d;
    complex_d** lesser_greater_inv_upperblk_ptr_d;

    cudaErrchk(cudaMalloc((void**)&retarded_inv_diagblk_d, batch_size * blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&retarded_inv_upperblk_d, batch_size * blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&retarded_inv_lowerblk_d, batch_size * blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&lesser_inv_diagblk_d, batch_size * blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&lesser_inv_upperblk_d, batch_size * blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&greater_inv_diagblk_d, batch_size * blocksize * blocksize * sizeof(complex_d)));
    cudaErrchk(cudaMalloc((void**)&greater_inv_upperblk_d, batch_size * blocksize * blocksize * sizeof(complex_d)));    
    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&retarded_inv_diagblk_small_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&lesser_inv_diagblk_small_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
        cudaErrchk(cudaMalloc((void**)&greater_inv_diagblk_small_d[i], batch_size * blocksize * blocksize * sizeof(complex_d)));
    }

    for(unsigned int j = 0; j < batch_size; j++){
        retarded_inv_diagblk_ptr_h[j] = retarded_inv_diagblk_d + j * blocksize * blocksize;
        retarded_inv_upperblk_ptr_h[j] = retarded_inv_upperblk_d + j * blocksize * blocksize;
        lesser_inv_diagblk_ptr_h[j] = lesser_inv_diagblk_d + j * blocksize * blocksize;
        lesser_inv_upperblk_ptr_h[j] = lesser_inv_upperblk_d + j * blocksize * blocksize;
        greater_inv_diagblk_ptr_h[j] = greater_inv_diagblk_d + j * blocksize * blocksize;
        greater_inv_upperblk_ptr_h[j] = greater_inv_upperblk_d + j * blocksize * blocksize;
        for(int i = 0; i < 2; i++){
            retarded_inv_diagblk_small_ptr_h[i][j] = retarded_inv_diagblk_small_d[i] + j * blocksize * blocksize;
            lesser_inv_diagblk_small_ptr_h[i][j] = lesser_inv_diagblk_small_d[i] + j * blocksize * blocksize;
            greater_inv_diagblk_small_ptr_h[i][j] = greater_inv_diagblk_small_d[i] + j * blocksize * blocksize;
        }
    }

    cudaErrchk(cudaMalloc((void**)&retarded_inv_diagblk_ptr_d, batch_size * sizeof(complex_d*)));
    cudaErrchk(cudaMalloc((void**)&retarded_inv_upperblk_ptr_d, batch_size * sizeof(complex_d*)));
    cudaErrchk(cudaMalloc((void**)&retarded_inv_diagblk_ptr_twice_d, 2*batch_size * sizeof(complex_d*)));

    cudaErrchk(cudaMalloc((void**)&lesser_greater_inv_diagblk_ptr_d, 2*batch_size * sizeof(complex_d*)));
    cudaErrchk(cudaMalloc((void**)&lesser_greater_inv_upperblk_ptr_d, 2*batch_size * sizeof(complex_d*)));


    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMalloc((void**)&retarded_inv_diagblk_small_ptr_d[i], batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMalloc((void**)&retarded_inv_diagblk_small_ptr_twice_d[i], 2*batch_size * sizeof(complex_d*)));
        cudaErrchk(cudaMalloc((void**)&lesser_greater_inv_diagblk_small_ptr_d[i], 2*batch_size * sizeof(complex_d*)));
    }

    cudaErrchk(cudaMemcpy(retarded_inv_diagblk_ptr_d, retarded_inv_diagblk_ptr_h, batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(retarded_inv_upperblk_ptr_d, retarded_inv_upperblk_ptr_h, batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));

    // copy the pointers twice for stacking lesser and greater
    cudaErrchk(cudaMemcpy(retarded_inv_diagblk_ptr_twice_d, retarded_inv_diagblk_ptr_h, batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(retarded_inv_diagblk_ptr_twice_d + batch_size, retarded_inv_diagblk_ptr_h, batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));

    for(int i = 0; i < 2; i++){
        cudaErrchk(cudaMemcpy(retarded_inv_diagblk_small_ptr_d[i], retarded_inv_diagblk_small_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(retarded_inv_diagblk_small_ptr_twice_d[i], retarded_inv_diagblk_small_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(retarded_inv_diagblk_small_ptr_twice_d[i] + batch_size, retarded_inv_diagblk_small_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(lesser_greater_inv_diagblk_small_ptr_d[i], lesser_inv_diagblk_small_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
        cudaErrchk(cudaMemcpy(lesser_greater_inv_diagblk_small_ptr_d[i] + batch_size, greater_inv_diagblk_small_ptr_h[i], batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    }

    cudaErrchk(cudaMemcpy(lesser_greater_inv_diagblk_ptr_d, lesser_inv_diagblk_ptr_h, batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(lesser_greater_inv_diagblk_ptr_d + batch_size, greater_inv_diagblk_ptr_h, batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(lesser_greater_inv_upperblk_ptr_d, lesser_inv_upperblk_ptr_h, batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(lesser_greater_inv_upperblk_ptr_d + batch_size, greater_inv_upperblk_ptr_h, batch_size * sizeof(complex_d*), cudaMemcpyHostToDevice));

    //memory for pivoting
    int *ipiv_d = NULL;
    int *info_d = NULL;
    cudaErrchk(cudaMalloc((void**)&info_d, batch_size * sizeof(int)))
    cudaErrchk(cudaMalloc((void**)&ipiv_d, batch_size * blocksize*sizeof(int)));


    
    // ----- END OF INIT SECTION -----

    cudaErrchk(cudaMemcpyAsync(system_matrix_diagblk_d[stream_compute], reinterpret_cast<const complex_d*>(system_matrix_diagblk_h[0]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_compute]));

    cudaErrchk(cudaMemcpyAsync(self_energy_lesser_diagblk_d[stream_compute], reinterpret_cast<const complex_d*>(self_energy_lesser_diagblk_h[0]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_compute]));

    cudaErrchk(cudaMemcpyAsync(self_energy_greater_diagblk_d[stream_compute], reinterpret_cast<const complex_d*>(self_energy_greater_diagblk_h[0]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_compute]));


    cublasErrchk(cublasZgetrfBatched(
            cublas_handle[stream_compute],
            blocksize,
            system_matrix_diagblk_ptr_d[stream_compute], blocksize, ipiv_d,
            info_d, batch_size));
    // inversion
    cublasErrchk(cublasZgetriBatched(
                                cublas_handle[stream_compute],
                                blocksize,
                                system_matrix_diagblk_ptr_d[stream_compute],
                                blocksize,
                                ipiv_d,
                                retarded_inv_diagblk_ptr_d,
                                blocksize,
                                info_d,
                                batch_size));

    // record finishing the inverse of the first block
    cudaErrchk(cudaEventRecord(schur_inverted[0], stream[stream_compute]));



    // use self_energy_lesser_upperblk_d[stream_compute]
    // as temporary buffer
    // g_lesser_greater[0:blocksize, 0:blocksize] =
    //     g_retarded[0:blocksize, 0:blocksize] @ 
    //     Sigma_lesser_greater[0:blocksize, 0:blocksize] @ 
    //     g_retarded[0:blocksize, 0:blocksize].conj().T

    alpha = make_cuDoubleComplex(1.0, 0.0);
    beta = make_cuDoubleComplex(0.0, 0.0);

    cublasErrchk(cublasZgemmBatched(
        cublas_handle[stream_compute],
        CUBLAS_OP_N, CUBLAS_OP_C,
        blocksize, blocksize, blocksize,
        &alpha,
        self_energy_lesser_greater_diagblk_ptr_d[stream_compute], blocksize,
        retarded_inv_diagblk_ptr_twice_d, blocksize,
        &beta,
        self_energy_lesser_greater_upperblk_ptr_d[stream_compute], blocksize, 2*batch_size));
    cublasErrchk(cublasZgemmBatched(
        cublas_handle[stream_compute],
        CUBLAS_OP_N, CUBLAS_OP_N,
        blocksize, blocksize, blocksize,
        &alpha,
        retarded_inv_diagblk_ptr_twice_d, blocksize,
        self_energy_lesser_greater_upperblk_ptr_d[stream_compute], blocksize,
        &beta,
        lesser_greater_inv_diagblk_ptr_d, blocksize, 2*batch_size));
    
    cudaErrchk(cudaEventRecord(lesser_greater_calculated[0], stream[stream_compute]));

    //wait for the inverse of the first block
    cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[0]));
    // 0. Inverse of the first block
    cudaErrchk(cudaMemcpyAsync(retarded_inv_diagblk_h[0], retarded_inv_diagblk_d,
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
    cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], lesser_greater_calculated[0]));
    cudaErrchk(cudaMemcpyAsync(lesser_inv_diagblk_h[0], lesser_inv_diagblk_d,
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
    cudaErrchk(cudaMemcpyAsync(greater_inv_diagblk_h[0], greater_inv_diagblk_d,
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));

    // unloading finished
    cudaErrchk(cudaEventRecord(unload_lesser_greater_diag[0], stream[stream_memunload]));


    // first memcpy happens before loop
    cudaErrchk(cudaMemcpyAsync(system_matrix_diagblk_d[stream_memload],
                reinterpret_cast<const complex_d*>(system_matrix_diagblk_h[1]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(system_matrix_upperblk_d[stream_memload],
                reinterpret_cast<const complex_d*>(system_matrix_upperblk_h[0]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(system_matrix_lowerblk_d[stream_memload],
                reinterpret_cast<const complex_d*>(system_matrix_lowerblk_h[0]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(self_energy_lesser_diagblk_d[stream_memload],
                reinterpret_cast<const complex_d*>(self_energy_lesser_diagblk_h[1]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(self_energy_lesser_upperblk_d[stream_memload],
                reinterpret_cast<const complex_d*>(self_energy_lesser_upperblk_h[0]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(self_energy_greater_diagblk_d[stream_memload],
                reinterpret_cast<const complex_d*>(self_energy_greater_diagblk_h[1]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
    cudaErrchk(cudaMemcpyAsync(self_energy_greater_upperblk_d[stream_memload],
                reinterpret_cast<const complex_d*>(self_energy_greater_upperblk_h[0]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));



    // // 1. Forward substitution (performed left to right)
    for (unsigned int i = 1; i < n_blocks; ++i) {


        int stream_memload = (i+1) % 2;
        int stream_compute = i % 2;
        int stream_memunload = 2;


        if(i < n_blocks-1){
            // load the blocks for the next iteration
            cudaErrchk(cudaMemcpyAsync(system_matrix_diagblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(system_matrix_diagblk_h[i+1]),
                        batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(system_matrix_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(system_matrix_upperblk_h[i]),
                        batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(system_matrix_lowerblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(system_matrix_lowerblk_h [i]),
                        batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(self_energy_lesser_diagblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(self_energy_lesser_diagblk_h[i+1]),
                        batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(self_energy_lesser_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(self_energy_lesser_upperblk_h[i]),
                        batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(self_energy_greater_diagblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(self_energy_greater_diagblk_h[i+1]),
                        batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(self_energy_greater_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(self_energy_greater_upperblk_h[i]),
                        batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
        }




        //wait for the schur inverse from the previous iteration
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], schur_inverted[i-1]));
        // without this the solution is not correct
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], lesser_greater_calculated[i-1]));

        // MatMul tmp = eig_lowerblk[i-1] * eig_inv_diagblk[i-1]
        // use the retarded_inv_diagblk_d from last iteration
        // use retarded_inv_diagblk_small_d[stream_compute] as tmp
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            system_matrix_lowerblk_ptr_d[stream_compute], blocksize,
            retarded_inv_diagblk_ptr_d, blocksize,
            &beta,
            retarded_inv_diagblk_small_ptr_d[stream_compute], blocksize, batch_size));
        
        //MatMul schur complement = eig_diagblk[i] - tmp * eig_upperblk[i-1]
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);

        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            retarded_inv_diagblk_small_ptr_d[stream_compute], blocksize,
            system_matrix_upperblk_ptr_d[stream_compute], blocksize,
            &beta,
            system_matrix_diagblk_ptr_d[stream_compute], blocksize, batch_size));

        // first temporary products for lesser and greater which use g_retarded[i_minus_one_, i_minus_one_]
        //calculate lesser and greater inverse
        //System_matrix[i_, i_minus_one_] @ g_lesser_greater[i_minus_one_, i_minus_one_]
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);

        // use  lesser_inv_upperblk_d as temporary buffer and greater_inv_upperblk_d
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            system_matrix_lowerblk_ptr_twice_d[stream_compute], blocksize,
            lesser_greater_inv_diagblk_ptr_d, blocksize,
            &beta,
            lesser_greater_inv_upperblk_ptr_d, blocksize, 2*batch_size));

        
        // System_matrix[i_, i_minus_one_] @
        //                 g_lesser_greater[i_minus_one_, i_minus_one_]
        //             + Sigma_lesser_greater[i_minus_one_, i_].conj().T @
        //                 g_retarded[i_minus_one_, i_minus_one_].conj().T

        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_C, CUBLAS_OP_C,
            blocksize, blocksize, blocksize,
            &alpha,
            self_energy_lesser_greater_upperblk_ptr_d[stream_compute], blocksize,
            retarded_inv_diagblk_ptr_twice_d, blocksize,
            &beta,
            lesser_greater_inv_upperblk_ptr_d, blocksize, 2*batch_size));


        // wait to not overwrite block to unload_lesser_greater_diag
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], unload_lesser_greater_diag[i-1]));


        // inverse schur complement
        cublasErrchk(cublasZgetrfBatched(
                cublas_handle[stream_compute],
                blocksize,
                system_matrix_diagblk_ptr_d[stream_compute], blocksize, ipiv_d,
                info_d, batch_size));


        // inversion
        cublasErrchk(cublasZgetriBatched(
                                    cublas_handle[stream_compute],
                                    blocksize,
                                    system_matrix_diagblk_ptr_d[stream_compute],
                                    blocksize,
                                    ipiv_d,
                                    retarded_inv_diagblk_ptr_d,
                                    blocksize,
                                    info_d,
                                    batch_size));

        // record finishing of computation in step i
        cudaErrchk(cudaEventRecord(schur_inverted[i], stream[stream_compute]));

        //calculate lesser and greater inverse
        // Sigma_lesser_greater[i_, i_]
        // - tmp @
        //     Sigma_lesser_greater[i_minus_one_, i_] 
        // tmp = retarded_inv_diagblk_small_d[stream_compute]

        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            retarded_inv_diagblk_small_ptr_twice_d[stream_compute], blocksize,
            self_energy_lesser_greater_upperblk_ptr_d[stream_compute], blocksize,
            &beta,
            self_energy_lesser_greater_diagblk_ptr_d[stream_compute], blocksize, 2*batch_size));


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
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize, blocksize,
            &alpha,
            lesser_greater_inv_upperblk_ptr_d, blocksize,
            system_matrix_lowerblk_ptr_twice_d[stream_compute], blocksize,
            &beta,
            self_energy_lesser_greater_diagblk_ptr_d[stream_compute], blocksize, 2*batch_size));



        // g_lesser_greater[i_, i_] = (
        //     g_retarded[i_, i_]
        //     @ (
        //  self_energy_lesser_diagblk_d[stream_compute]/self_energy_greater_diagblk_d
        //     )
        //     @ g_retarded[i_, i_].conj().T
        // use self_energy_lesser_upperblk_d[stream_compute] as temporary buffer
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize, blocksize,
            &alpha,
            self_energy_lesser_greater_diagblk_ptr_d[stream_compute], blocksize,
            retarded_inv_diagblk_ptr_twice_d, blocksize,
            &beta,
            self_energy_lesser_greater_upperblk_ptr_d[stream_compute], blocksize, 2*batch_size));

        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            retarded_inv_diagblk_ptr_twice_d, blocksize,
            self_energy_lesser_greater_upperblk_ptr_d[stream_compute], blocksize,
            &beta,
            lesser_greater_inv_diagblk_ptr_d, blocksize, 2*batch_size));


        cudaErrchk(cudaEventRecord(lesser_greater_calculated[i], stream[stream_compute]));

        // wait to unload_lesser_greater_diag for the finish of computations
        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[i]));
        // lesser inv upperblk  for the small g
        // last small g is not needed
        cudaErrchk(cudaMemcpyAsync(retarded_inv_diagblk_h[i],
                    retarded_inv_diagblk_d,
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));            

        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], lesser_greater_calculated[i]));
        cudaErrchk(cudaMemcpyAsync(lesser_inv_diagblk_h[i],
                    lesser_inv_diagblk_d,
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        cudaErrchk(cudaMemcpyAsync(greater_inv_diagblk_h[i],
                    greater_inv_diagblk_d,
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        // unloading finished
        cudaErrchk(cudaEventRecord(unload_lesser_greater_diag[i], stream[stream_memunload]));



    }
    int stream_memload_before = (n_blocks) % 2;
    int stream_compute_before = (n_blocks-1) % 2;



    cudaErrchk(cudaMemcpyAsync(system_matrix_upperblk_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(system_matrix_upperblk_h[n_blocks-2]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    cudaErrchk(cudaMemcpyAsync(system_matrix_lowerblk_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(system_matrix_lowerblk_h[n_blocks-2]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    cudaErrchk(cudaMemcpyAsync(self_energy_lesser_upperblk_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(self_energy_lesser_upperblk_h[n_blocks-2]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    cudaErrchk(cudaMemcpyAsync(self_energy_greater_upperblk_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(self_energy_greater_upperblk_h[n_blocks-2]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));


    cudaErrchk(cudaStreamWaitEvent(stream[stream_memload_before], unload_lesser_greater_diag[n_blocks-2]));
    cudaErrchk(cudaMemcpyAsync(retarded_inv_diagblk_small_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(retarded_inv_diagblk_h[n_blocks-2]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    cudaErrchk(cudaMemcpyAsync(lesser_inv_diagblk_small_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(lesser_inv_diagblk_h[n_blocks-2]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));
    cudaErrchk(cudaMemcpyAsync(greater_inv_diagblk_small_d[stream_memload_before],
                reinterpret_cast<const complex_d*>(greater_inv_diagblk_h[n_blocks-2]),
                batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload_before]));    



    // 2. Backward substitution (performed right to left)
    for(int i = n_blocks-2; i >= 0; --i){

        // fix stream compute to be the stream which loaded
        // blocks before the loop
        stream_memload = (stream_compute_before + (i - n_blocks + 2) ) % 2;
        stream_compute = (stream_memload_before + (i - n_blocks + 2) ) % 2;
        stream_memunload = 2;


        if(i > 0){
            cudaErrchk(cudaMemcpyAsync(system_matrix_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(system_matrix_upperblk_h[i-1]),
                        batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(system_matrix_lowerblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(system_matrix_lowerblk_h[i-1]),
                        batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));


            cudaErrchk(cudaMemcpyAsync(retarded_inv_diagblk_small_d[stream_memload],
                        reinterpret_cast<const complex_d*>(retarded_inv_diagblk_h[i-1]),
                        batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));

            cudaErrchk(cudaMemcpyAsync(lesser_inv_diagblk_small_d[stream_memload],
                        reinterpret_cast<const complex_d*>(lesser_inv_diagblk_h[i-1]),
                        batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(greater_inv_diagblk_small_d[stream_memload],
                        reinterpret_cast<const complex_d*>(greater_inv_diagblk_h[i-1]),
                        batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
                    
            cudaErrchk(cudaMemcpyAsync(self_energy_lesser_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(self_energy_lesser_upperblk_h[i-1]),
                        batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
            cudaErrchk(cudaMemcpyAsync(self_energy_greater_upperblk_d[stream_memload],
                        reinterpret_cast<const complex_d*>(self_energy_greater_upperblk_h[i-1]),
                        batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyHostToDevice, stream[stream_memload]));
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
        complex_d *buf4_lesser_d = self_energy_lesser_upperblk_d[stream_compute];
        complex_d *buf4_greater_d = self_energy_greater_upperblk_d[stream_compute];

        complex_d **buf4_lesser_greater_ptr_d = self_energy_lesser_greater_upperblk_ptr_d[stream_compute];

        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize, blocksize,
            &alpha,
            retarded_inv_diagblk_ptr_twice_d, blocksize,
            self_energy_lesser_greater_upperblk_ptr_d[stream_compute], blocksize,
            &beta,
            self_energy_lesser_greater_diagblk_ptr_d[stream_compute], blocksize, 2*batch_size));


        // self_energy_lesser_greater_upperblk_d[stream_compute] will be overwritten
        // okay since it not used anymore
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize, blocksize,
            &alpha,
            self_energy_lesser_greater_diagblk_ptr_d[stream_compute], blocksize,
            retarded_inv_diagblk_small_ptr_twice_d[stream_compute], blocksize,
            &beta,
            buf4_lesser_greater_ptr_d, blocksize, 2*batch_size));


        // buf2 = (G_tmp @    
        //     System_matrix[i_plus_one_, i_]) 
        // buf2 = self_energy_lesser_diagblk_d[stream_compute]
        alpha = make_cuDoubleComplex(1.0, 0.0);
        complex_d **buf2_ptr_d = self_energy_lesser_diagblk_ptr_d[stream_compute];
        complex_d **buf2_ptr_twice_d = self_energy_lesser_diagblk_ptr_twice_d[stream_compute];
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            retarded_inv_diagblk_ptr_d, blocksize,
            system_matrix_lowerblk_ptr_d[stream_compute], blocksize,
            &beta,
            buf2_ptr_d, blocksize, batch_size));

        // buf1 = (g_retarded[i_, i_] @
        //     System_matrix[i_, i_plus_one_])
        // self_energy_greater_diagblk_d[stream_compute] is not overwritten by memcpy
        // buf1 = self_energy_greater_diagblk_d[stream_compute]
        // since it is only used in the forward pass
        complex_d **buf1_ptr_d = self_energy_greater_diagblk_ptr_d[stream_compute];
        complex_d **buf1_ptr_twice_d = self_energy_greater_diagblk_ptr_twice_d[stream_compute];
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            retarded_inv_diagblk_small_ptr_d[stream_compute], blocksize,
            system_matrix_upperblk_ptr_d[stream_compute], blocksize,
            &beta,
            buf1_ptr_d, blocksize, batch_size));

        // buf7 = - buf2 @
        //     g_retarded[i_, i_]      
        // self_energy_greater_diagblk_d[stream_memload] only used in the forward pass
        complex_d* buf7_d = self_energy_greater_diagblk_d[stream_memload];
        complex_d **buf7_ptr_d = self_energy_greater_diagblk_ptr_d[stream_memload];
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            buf2_ptr_d, blocksize,
            retarded_inv_diagblk_small_ptr_d[stream_compute], blocksize,
            &beta,
            buf7_ptr_d, blocksize, batch_size));

        // G_tmp   =  g_retarded[i_, i_] - (buf1 @
        //                                     buf7)
        
        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            buf1_ptr_d, blocksize,
            buf7_ptr_d, blocksize,
            &beta,
            retarded_inv_diagblk_small_ptr_d[stream_compute], blocksize, batch_size));
        

        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], unload_retarded[i+1]));

        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            buf1_ptr_d, blocksize,
            retarded_inv_diagblk_ptr_d, blocksize,
            &beta,
            retarded_inv_upperblk_ptr_d, blocksize, batch_size));


        // retarded_inv_diagblk_small_d[stream_compute] saves now G_tmp
        cudaErrchk(cudaMemcpyAsync(retarded_inv_diagblk_d,
                    retarded_inv_diagblk_small_d[stream_compute],
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));
        cudaErrchk(cudaMemcpyAsync(retarded_inv_lowerblk_d,
                    buf7_d,
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));

        cudaErrchk(cudaEventRecord(schur_inverted[i], stream[stream_compute]));

        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], schur_inverted[i]));
    
        cudaErrchk(cudaMemcpyAsync(retarded_inv_diagblk_h[i],
                    retarded_inv_diagblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        cudaErrchk(cudaMemcpyAsync(retarded_inv_upperblk_h[i],
                    retarded_inv_upperblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        cudaErrchk(cudaMemcpyAsync(retarded_inv_lowerblk_h[i],
                    retarded_inv_lowerblk_d,
                    blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));

        cudaErrchk(cudaEventRecord(unload_retarded[i], stream[stream_memunload]));

        // buf5 = (buf2 @ g_lesser_greater[i_, i_])
        // buf5 = system_matrix_diagblk_d
        // buf5_lesser = system_matrix_diagblk_d[stream_compute]
        // buf5_greater = system_matrix_diagblk_d[stream_memload]
        // important to fix streams such that pointers do not swap in the combined call
        complex_d *buf5_lesser_d = system_matrix_diagblk_d[0];
        complex_d *buf5_greater_d = system_matrix_diagblk_d[1];
        // scuffed because first half points on stream_compute and second half on stream_memload
        complex_d **buf5_lesser_greater_ptr_d = system_matrix_diagblk_ptr_scuffed_d;



        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            buf2_ptr_twice_d, blocksize,
            lesser_greater_inv_diagblk_small_ptr_d[stream_compute], blocksize,
            &beta,
            buf5_lesser_greater_ptr_d, blocksize, 2*batch_size));


        // buf3 is  for both lesser and greater and now memory for both seperately is needed
        // buf3_lesser = system_matrix_upperblk_d[stream_compute],
        // buf3_greater = system_matrix_lowerblk_d[stream_compute],
        // buf3 = (
        //     G_lesser_greater[i_plus_one_, i_plus_one_] @
        //     buf1.conj().T
        // )
        complex_d **buf3_lesser_ptr_d = system_matrix_upperblk_ptr_d[stream_compute];
        complex_d **buf3_greater_ptr_d = system_matrix_lowerblk_ptr_d[stream_compute];
        complex_d **buf3_lesser_greater_ptr_d = system_matrix_upper_lowerblk_ptr_d[stream_compute];

        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize, blocksize,
            &alpha,
            lesser_greater_inv_diagblk_ptr_d, blocksize,
            buf1_ptr_twice_d, blocksize,
            &beta,
            buf3_lesser_greater_ptr_d, blocksize, 2*batch_size));


        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], unload_lesser_greater_upper[i+1]));
        // G_lesser_greater[i_plus_one_, i_] =(
        //     buf4
        //     - buf5
        //     - buf3
        // )
        // no batched geam version


        // use self_energy_greater_diagblk_d[stream_memload] as temporary buffer
        complex_d *tmp_lesser_d = self_energy_lesser_diagblk_d[stream_compute];
        complex_d *tmp_greater_d = self_energy_lesser_diagblk_d[stream_memload];
        complex_d **tmp_lesser_ptr_d = self_energy_lesser_diagblk_ptr_d[stream_compute];
        complex_d **tmp_greater_ptr_d = self_energy_lesser_diagblk_ptr_d[stream_memload];

        //playing tricks with geam
        // since the memory is contiguous
        // the batch can be interpreted as a matrix
        alpha = make_cuDoubleComplex(-1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgeam(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, batch_size * blocksize,
            &alpha,
            buf4_lesser_d, blocksize,
            &beta,
            buf5_lesser_d, blocksize,
            tmp_lesser_d, blocksize
        ));
        cublasErrchk(cublasZgeam(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, batch_size * blocksize,
            &alpha,
            buf4_greater_d, blocksize,
            &beta,
            buf5_greater_d, blocksize,
            tmp_greater_d, blocksize
        ));

        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(quatrexblasZgeamBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_C, CUBLAS_OP_C,
            blocksize, blocksize,
            &alpha,
            tmp_lesser_ptr_d, blocksize,
            &beta,
            buf3_lesser_ptr_d, blocksize,
            lesser_greater_inv_upperblk_ptr_d, blocksize,
            batch_size
        ));
        cublasErrchk(quatrexblasZgeamBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_C, CUBLAS_OP_C,
            blocksize, blocksize,
            &alpha,
            tmp_greater_ptr_d, blocksize,
            &beta,
            buf3_greater_ptr_d, blocksize,
            lesser_greater_inv_upperblk_ptr_d + batch_size, blocksize,
            batch_size
        ));


        cudaErrchk(cudaEventRecord(lesser_greater_calculated_upper[i], stream[stream_compute]));

        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], lesser_greater_calculated_upper[i]));

        cudaErrchk(cudaMemcpyAsync(lesser_inv_upperblk_h[i],
                    lesser_inv_upperblk_d,
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        cudaErrchk(cudaMemcpyAsync(greater_inv_upperblk_h[i],
                    greater_inv_upperblk_d,
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));

        cudaErrchk(cudaEventRecord(unload_lesser_greater_upper[i], stream[stream_memunload]));

        // buf6 = (buf1 @ buf4)
        // buf6_lesser = self_energy_lesser_diagblk_d[stream_compute]
        // buf6_greater = self_energy_lesser_diagblk_d[stream_memload]
        complex_d *buf6_lesser_d = self_energy_lesser_diagblk_d[0];
        complex_d *buf6_greater_d = self_energy_lesser_diagblk_d[1];
        complex_d **buf6_lesser_greater_ptr_d = self_energy_lesser_diagblk_ptr_scuffed_d;

        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            buf1_ptr_twice_d, blocksize,
            buf4_lesser_greater_ptr_d, blocksize,
            &beta,
            buf6_lesser_greater_ptr_d, blocksize, 2*batch_size));

        // buf8 = (buf1 @ buf5)
        // buf8_lesser = self_energy_lesser_upperblk_d[stream_compute]
        // buf8_greater = self_energy_greater_upperblk_d[stream_compute]
        complex_d *buf8_lesser_d = self_energy_lesser_upperblk_d[stream_compute];
        complex_d *buf8_greater_d = self_energy_greater_upperblk_d[stream_compute];
        complex_d **buf8_lesser_greater_ptr_d = self_energy_lesser_greater_upperblk_ptr_d[stream_compute];

        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            buf1_ptr_twice_d, blocksize,
            buf5_lesser_greater_ptr_d, blocksize,
            &beta,
            buf8_lesser_greater_ptr_d, blocksize, 2*batch_size));


        // g_lesser_greater[i_, i_]
        // + buf1 @ buf3
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgemmBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, blocksize, blocksize,
            &alpha,
            buf1_ptr_twice_d, blocksize,
            buf3_lesser_greater_ptr_d, blocksize,
            &beta,
            lesser_greater_inv_diagblk_small_ptr_d[stream_compute], blocksize, 2*batch_size));


        // playing tricks with geam
        // since the memory is contiguous
        // the batch can be interpreted as a matrix
        // - buf6
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(-1.0, 0.0);
        cublasErrchk(cublasZgeam(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, batch_size*blocksize,
            &alpha,
            lesser_inv_diagblk_small_d[stream_compute], blocksize,
            &beta,
            buf6_lesser_d, blocksize,
            lesser_inv_diagblk_small_d[stream_compute], blocksize
        ));
        cublasErrchk(cublasZgeam(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, batch_size*blocksize,
            &alpha,
            greater_inv_diagblk_small_d[stream_compute], blocksize,
            &beta,
            buf6_greater_d, blocksize,
            greater_inv_diagblk_small_d[stream_compute], blocksize
        ));

        // + buf8
        //complex_d *tmp2 = self_energy_greater_diagblk_ptr_d[stream_compute];;
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(cublasZgeam(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, batch_size*blocksize,
            &alpha,
            lesser_inv_diagblk_small_d[stream_compute], blocksize,
            &beta,
            buf8_lesser_d, blocksize,
            lesser_inv_diagblk_small_d[stream_compute], blocksize
        ));        
        cublasErrchk(cublasZgeam(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_N,
            blocksize, batch_size*blocksize,
            &alpha,
            greater_inv_diagblk_small_d[stream_compute], blocksize,
            &beta,
            buf8_greater_d, blocksize,
            greater_inv_diagblk_small_d[stream_compute], blocksize
        ));
        
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(1.0, 0.0);
        cublasErrchk(quatrexblasZgeamBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize,
            &alpha,
            lesser_greater_inv_diagblk_small_ptr_d[stream_compute], blocksize,
            &beta,
            buf6_lesser_greater_ptr_d, blocksize,
            lesser_greater_inv_diagblk_small_ptr_d[stream_compute], blocksize,
            2*batch_size
        ));
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(-1.0, 0.0);
        cublasErrchk(quatrexblasZgeamBatched(
            cublas_handle[stream_compute],
            CUBLAS_OP_N, CUBLAS_OP_C,
            blocksize, blocksize,
            &alpha,
            lesser_greater_inv_diagblk_small_ptr_d[stream_compute], blocksize,
            &beta,
            buf8_lesser_greater_ptr_d, blocksize,
            lesser_greater_inv_diagblk_small_ptr_d[stream_compute], blocksize,
            2*batch_size
        ));
    

        // wait to not overwrite blocks to unload_lesser_greater_diag
        cudaErrchk(cudaStreamWaitEvent(stream[stream_compute], unload_lesser_greater_diag[i+1]));        

        cudaErrchk(cudaMemcpyAsync(lesser_inv_diagblk_d,
                    lesser_inv_diagblk_small_d[stream_compute],
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));

        cudaErrchk(cudaMemcpyAsync(greater_inv_diagblk_d,
                    greater_inv_diagblk_small_d[stream_compute],
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToDevice, stream[stream_compute]));

        cudaErrchk(cudaEventRecord(lesser_greater_calculated[i], stream[stream_compute]));

        // wait to unload_lesser_greater_diag for the finish of computations
        cudaErrchk(cudaStreamWaitEvent(stream[stream_memunload], lesser_greater_calculated[i]));

        cudaErrchk(cudaMemcpyAsync(lesser_inv_diagblk_h[i],
                    lesser_inv_diagblk_d,
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        cudaErrchk(cudaMemcpyAsync(greater_inv_diagblk_h[i],
                    greater_inv_diagblk_d,
                    batch_size * blocksize * blocksize * sizeof(complex_d), cudaMemcpyDeviceToHost, stream[stream_memunload]));
        // unloading finished
        cudaErrchk(cudaEventRecord(unload_lesser_greater_diag[i], stream[stream_memunload]));
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
        if(retarded_inv_diagblk_small_d[i]) {
            cudaErrchk(cudaFree(retarded_inv_diagblk_small_d[i]));
        }
    }
    if(retarded_inv_diagblk_d) {
        cudaErrchk(cudaFree(retarded_inv_diagblk_d));
    }
    if(retarded_inv_upperblk_d) {
        cudaErrchk(cudaFree(retarded_inv_upperblk_d));
    }
    if(retarded_inv_lowerblk_d) {
        cudaErrchk(cudaFree(retarded_inv_lowerblk_d));
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



    if(ipiv_d){
        cudaErrchk(cudaFree(ipiv_d));
    }
    if(info_d){
        cudaErrchk(cudaFree(info_d));
    }

    // deallocate pointer arrays
    for(int i = 0; i < 2; i++){
        if(system_matrix_diagblk_ptr_d[i]) {
            cudaErrchk(cudaFree(system_matrix_diagblk_ptr_d[i]));
        }
        if(system_matrix_upperblk_ptr_d[i]) {
            cudaErrchk(cudaFree(system_matrix_upperblk_ptr_d[i]));
        }
        if(system_matrix_lowerblk_ptr_d[i]) {
            cudaErrchk(cudaFree(system_matrix_lowerblk_ptr_d[i]));
        }
        if(system_matrix_upper_lowerblk_ptr_d[i]) {
            cudaErrchk(cudaFree(system_matrix_upper_lowerblk_ptr_d[i]));
        }

        if(self_energy_lesser_diagblk_ptr_d[i]) {
            cudaErrchk(cudaFree(self_energy_lesser_diagblk_ptr_d[i]));
        }

        if(self_energy_greater_diagblk_ptr_d[i]) {
            cudaErrchk(cudaFree(self_energy_greater_diagblk_ptr_d[i]));
        }


        if(self_energy_lesser_greater_diagblk_ptr_d[i]) {
            cudaErrchk(cudaFree(self_energy_lesser_greater_diagblk_ptr_d[i]));
        }
        if(self_energy_lesser_diagblk_ptr_twice_d[i]){
            cudaErrchk(cudaFree(self_energy_lesser_diagblk_ptr_twice_d[i]));
        }
        if(self_energy_greater_diagblk_ptr_twice_d[i]){
            cudaErrchk(cudaFree(self_energy_greater_diagblk_ptr_twice_d[i]));
        }


        if(self_energy_lesser_greater_upperblk_ptr_d[i]) {
            cudaErrchk(cudaFree(self_energy_lesser_greater_upperblk_ptr_d[i]));
        }

        if(system_matrix_lowerblk_ptr_twice_d[i]) {
            cudaErrchk(cudaFree(system_matrix_lowerblk_ptr_twice_d[i]));
        }

        if(retarded_inv_diagblk_small_ptr_d[i]) {
            cudaErrchk(cudaFree(retarded_inv_diagblk_small_ptr_d[i]));
        }
        if(retarded_inv_diagblk_small_ptr_twice_d){
            cudaErrchk(cudaFree(retarded_inv_diagblk_small_ptr_twice_d[i]));
        }


        if(lesser_greater_inv_diagblk_small_ptr_d[i]){
            cudaErrchk(cudaFree(lesser_greater_inv_diagblk_small_ptr_d[i]));
        }

        if(lesser_inv_diagblk_small_d[i]){
            cudaErrchk(cudaFree(lesser_inv_diagblk_small_d[i]));
        }
        if(greater_inv_diagblk_small_d[i]){
            cudaErrchk(cudaFree(greater_inv_diagblk_small_d[i]));
        }

    }
    if(system_matrix_diagblk_ptr_scuffed_d){
        cudaErrchk(cudaFree(system_matrix_diagblk_ptr_scuffed_d));
    }
    if(self_energy_lesser_diagblk_ptr_scuffed_d){
        cudaErrchk(cudaFree(self_energy_lesser_diagblk_ptr_scuffed_d));
    }


    if(retarded_inv_diagblk_ptr_d){
        cudaErrchk(cudaFree(retarded_inv_diagblk_ptr_d));
    }
    if(retarded_inv_upperblk_ptr_d){
        cudaErrchk(cudaFree(retarded_inv_upperblk_ptr_d));
    }
    if(retarded_inv_diagblk_ptr_twice_d){
        cudaErrchk(cudaFree(retarded_inv_diagblk_ptr_twice_d));
    }

    if(lesser_greater_inv_diagblk_ptr_d){
        cudaErrchk(cudaFree(lesser_greater_inv_diagblk_ptr_d));
    }
    if(lesser_greater_inv_upperblk_ptr_d){
        cudaErrchk(cudaFree(lesser_greater_inv_upperblk_ptr_d));
    }


    // dealocate event
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
        if(unload_lesser_greater_diag[i]){
            cudaErrchk(cudaEventDestroy(unload_lesser_greater_diag[i]));
        }
        if(unload_lesser_greater_upper[i]){
            cudaErrchk(cudaEventDestroy(unload_lesser_greater_upper[i]));
        }
        if(unload_retarded[i]){
            cudaErrchk(cudaEventDestroy(unload_retarded[i]));
        }
    } 
}
