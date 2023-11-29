#include "PSR.h"

void createblockMatrixType(MPI_Datatype* blockMatrixType, int stride, int blocksize) {
    MPI_Datatype blockType;
    MPI_Type_vector(blocksize, blocksize, stride, MPI_DOUBLE_COMPLEX, &blockType);
    MPI_Type_commit(&blockType);

    MPI_Aint extent;
    MPI_Type_extent(blockType, &extent);

    MPI_Type_create_resized(blockType, 0, extent, blockMatrixType);
    MPI_Type_commit(blockMatrixType);
}

void create_subblock_Type(MPI_Datatype* subblockType, int stride, int blocksize, int rowBlocks) {
    
    MPI_Type_vector(rowBlocks * blocksize, blocksize, stride, MPI_DOUBLE_COMPLEX, subblockType);

    MPI_Type_commit(subblockType);

}

void create_resized_subblock_Type(MPI_Datatype* subblockType_resized, MPI_Datatype subblockType, int stride, int blocksize, int rowBlocks) {
    
    MPI_Type_create_resized(subblockType, 0, blocksize * blocksize  * sizeof(std::complex<double>), subblockType_resized);

    MPI_Type_commit(subblockType_resized);

}

void create_ul2_redschur_blockpattern_Type(MPI_Datatype* blockPatternType, MPI_Datatype subblockType_2, int blocksize, int stride, int partition_blocksize) {
    //Creating an MPI datatype for the upper left double block sent after distributed Schur reduction

    int *blockcounts =  new int[1];
    blockcounts[0] = 1;

    long int *displacements = new long int[1];
    displacements[0] = blocksize * sizeof(std::complex<double>) * (partition_blocksize -1) + stride * blocksize * sizeof(std::complex<double>) * (partition_blocksize -1);

    MPI_Datatype *blocktypes = new MPI_Datatype[1];
    blocktypes[0] = subblockType_2;

    MPI_Type_create_struct(1, blockcounts, displacements, blocktypes, blockPatternType);
    MPI_Type_commit(blockPatternType);


    delete[] blockcounts;
    delete[] displacements;
    delete[] blocktypes;
}


void create_br2_redschur_blockpattern_Type(MPI_Datatype* blockPatternType, MPI_Datatype subblockType_2, int blocksize, int stride, int partition_blocksize) {
    //Creating an MPI datatype for the bottom right double block sent after distributed Schur reduction

    int *blockcounts =  new int[1];
    blockcounts[0] = 1;

    long int *displacements = new long int[1];
    displacements[0] = 0;

    MPI_Datatype *blocktypes = new MPI_Datatype[1];
    blocktypes[0] = subblockType_2;

    MPI_Type_create_struct(1, blockcounts, displacements, blocktypes, blockPatternType);
    MPI_Type_commit(blockPatternType);


    delete[] blockcounts;
    delete[] displacements;
    delete[] blocktypes;
}

void create_central_redschur_blockpattern_Type(MPI_Datatype* blockPatternType, MPI_Datatype subblockType, MPI_Datatype subblockType_2, int blocksize, int stride, int partition_blocksize) {
    //Creating an MPI datatype for the central block sent after distributed Schur reduction
    int * blockcounts = new int[4];
    blockcounts[0] = 1;
    blockcounts[1] = 1;
    blockcounts[2] = 1;
    blockcounts[3] = 1;

    long int *displacements = new long int[4];
    displacements[0] = 0;
    displacements[1] = stride * blocksize * sizeof(std::complex<double>) * (partition_blocksize);
    displacements[2] = blocksize * sizeof(std::complex<double>) * (partition_blocksize - 1) + stride * blocksize * sizeof(std::complex<double>);
    displacements[3] = blocksize * sizeof(std::complex<double>) * (partition_blocksize - 1) + stride * blocksize * sizeof(std::complex<double>) * (partition_blocksize);

    MPI_Datatype *blocktypes = new MPI_Datatype[4];
    blocktypes[0] = subblockType_2;
    blocktypes[1] = subblockType;
    blocktypes[2] = subblockType;
    blocktypes[3] = subblockType_2;

    MPI_Type_create_struct(4, blockcounts, displacements, blocktypes, blockPatternType);
    MPI_Type_commit(blockPatternType);

    delete[] blockcounts;  
    delete[] displacements;
    delete[] blocktypes;
}



void create_receive_example_block_pattern_Type(MPI_Datatype* ReceiveExample_blockPatternType, MPI_Datatype ur2_blockPatternType, MPI_Datatype subblockType_3, int blocksize, int stride) {
    //Creating an MPI datatype for the lower left double block
    // MPI_Datatype subblock_type;
    // create_subblock_Type(&subblock_type, stride, blocksize);
    MPI_Datatype receive_temp;
    int *blockcounts =  new int[2];
    blockcounts[0] = 1;
    blockcounts[1] = 1;

    long int *displacements = new long int[2];
    displacements[0] = 0;
    displacements[1] = blocksize * sizeof(std::complex<double>) * stride + 2 * blocksize * sizeof(std::complex<double>);
    MPI_Datatype *blocktypes = new MPI_Datatype[2];
    blocktypes[0] = ur2_blockPatternType;
    blocktypes[1] = subblockType_3; 

    MPI_Type_create_struct(2, blockcounts, displacements, blocktypes, &receive_temp);
    MPI_Type_commit(&receive_temp);

    MPI_Aint extent;
    MPI_Type_extent(receive_temp, &extent);

    MPI_Type_create_resized(receive_temp, 0, extent, ReceiveExample_blockPatternType);
    MPI_Type_commit(ReceiveExample_blockPatternType);
    MPI_Type_free(&receive_temp);

    delete[] blockcounts;
    delete[] displacements;
    delete[] blocktypes;
}