// Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
#include "utils.h"

bool load_binary_matrix(
    const char *filename, 
    std::complex<double> *matrix, 
    int rows, 
    int cols)
{
    std::FILE *fp;

    fp = std::fopen(filename, "rb");
    if (fp == nullptr) {
        std::printf("Error opening file\n");
        std::printf("Filename: %s\n", filename);
        return false;
    }

    std::fread(matrix, sizeof(std::complex<double>), rows * cols, fp);

    std::fclose(fp);
    return true;
}

template<typename T>
bool load_binary_array(
    std::string filename, 
    T *array,
    int size)
{
    std::FILE *fp;

    fp = std::fopen(filename.c_str(), "rb");
    if (fp == nullptr) {
        std::printf("Error opening file\n");
        return false;
    }

    std::fread(array, sizeof(T), size, fp);

    std::fclose(fp);
    return true;
}
template bool load_binary_array<double>(std::string filename, double* array, int size);
template bool load_binary_array<int>(std::string filename, int* array, int size);
template bool load_binary_array<std::complex<double>>(std::string filename, std::complex<double>* array, int size);


void free_matrix(
    std::complex<double> *matrix)
{
    free(matrix);
}


void print_matrix(
    std::complex<double> *matrix, 
    int rows, 
    int cols)
{
    // Access matrix elements
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            std::printf("%f + %fi ", std::real(matrix[i * cols + j]), std::imag(matrix[i * cols + j]));
        std::printf("\n");
    }
}

bool load_matrix_parameters(
    const char *filename, 
    unsigned int *matrice_size, 
    unsigned int *blocksize)
{
    FILE *fp;

    fp = std::fopen(filename, "r");
    if (fp == NULL) {
        std::printf("Error opening file\n");
        return true;
    }

    std::fscanf(fp, "%u %u", matrice_size, blocksize);

    std::fclose(fp);

    return false;
}

bool load_matrix_parameters_batched(
    const char *filename, 
    unsigned int *matrice_size, 
    unsigned int *blocksize,
    unsigned int *batchsize)
{
    FILE *fp;

    fp = std::fopen(filename, "r");
    if (fp == NULL) {
        std::printf("Error opening file\n");
        return true;
    }

    std::fscanf(fp, "%u %u %u", matrice_size, blocksize, batchsize);

    std::fclose(fp);

    return false;
}


template<typename T>
bool load_text_array(
    const char *filename, 
    T *array,
    int size)
{

    std::ifstream ifile(filename, std::ios::in);
    if (!ifile.is_open()) {
        std::printf("Error opening file\n");
        return false;
    }

    double num = T(0.0);
    //keep storing values from the text file so long as data exists:
    for (int i = 0; i < size; i++) {
        ifile >> num;
        array[i] = (T)num;
    }

    ifile.close();

    return true;
}
// Explicit instantiation of the template
// else not found in compilation
// other option would be to put the implementation in the header file
template bool load_text_array<double>(const char* filename, double* array, int size);
template bool load_text_array<int>(const char* filename, int* array, int size);

template <typename T>
bool save_text_array(
    char *filename,
    const T* array,
    int size)
{

    std::ofstream file(filename);
    if (file.is_open()) {
        for(int i = 0; i < size; i++){
            file << array[i] << " "; 
        }
        file.close();
        std::printf("Array data written to file.\n");
        return true;
    } else {
        std::printf("Unable to open the file for writing.\n");
        return false;
    }
}
template bool save_text_array<double>(char* filename, const double* array, int size);

template<typename T>
void sparse_to_dense(
    T *dense_matrix,
    T *data,
    int *indices,
    int *indptr,
    int matrice_size)
{

    for (int i = 0; i < matrice_size; i++) {
        for (int j = 0; j < matrice_size; j++) {
            // could not work for complex data type
            dense_matrix[i*matrice_size + j] = T(0);
        }
    }

    for(int i = 0; i < matrice_size; i++){
        for(int j = indptr[i]; j < indptr[i+1]; j++){
            dense_matrix[i*matrice_size + indices[j]] = data[j];
        }
    }
}

template void sparse_to_dense<double>(double *dense_matrix,
    double *data,
    int *indices,
    int *indptr,
    int matrice_size);


template<typename T>
bool assert_array_magnitude(
    T *array_test,
    T *array_ref,
    double abstol,
    double reltol,
    int size)
{
    double sum_difference = 0.0;
    double sum_ref = 0.0;
    for (int i = 0; i < size; i++) {
        sum_difference += std::abs(array_test[i] - array_ref[i])*std::abs(array_test[i] - array_ref[i]);
        sum_ref += std::abs(array_ref[i])*std::abs(array_ref[i]);

    }
    sum_difference = std::sqrt(sum_difference);
    sum_ref = std::sqrt(sum_ref);
    if (sum_difference > reltol * sum_ref + abstol) {
        return false;
    }
    return true;
}
template bool assert_array_magnitude<std::complex<double>>(
    std::complex<double> *array_test,
    std::complex<double> *array_ref,
    double abstol,
    double reltol,
    int size);

template<typename T>
void transform_diagblk(
    T *matrix_diagblk,
    T *matrix_diagblk_h,
    unsigned int blocksize,
    unsigned int matrix_size)
{
    #pragma omp parallel for
    for(unsigned int i = 0; i < blocksize * matrix_size; i++){
        // block index
        int k = i / (blocksize * blocksize);
        // index inside block
        int h = i % (blocksize * blocksize);
        // row inside block
        int m = h % blocksize;
        // col inside block
        int n = h / blocksize;
        matrix_diagblk_h[i] = matrix_diagblk[m*matrix_size + k*blocksize + n];
    }
}
template void transform_diagblk<std::complex<double>>(
    std::complex<double> *matrix_diagblk,
    std::complex<double> *matrix_diagblk_h,
    unsigned int blocksize,
    unsigned int matrix_size);

template<typename T>
void transform_offblk(
    T *matrix_offblk,
    T *matrix_offblk_h,
    unsigned int blocksize,
    unsigned int off_diag_size)
{
    #pragma omp parallel for
    for(unsigned int i = 0; i < blocksize * off_diag_size; i++){
        // block index
        int k = i / (blocksize * blocksize);
        // index inside block
        int h = i % (blocksize * blocksize);
        // row inside block
        int m = h % blocksize;
        // col inside block
        int n = h / blocksize;
        matrix_offblk_h[i] = matrix_offblk[m*off_diag_size + k*blocksize + n];

    }
}
template void transform_offblk<std::complex<double>>(
    std::complex<double> *matrix_offblk,
    std::complex<double> *matrix_offblk_h,
    unsigned int blocksize,
    unsigned int off_diag_size);

