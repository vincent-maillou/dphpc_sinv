# NVCC = nvcc
# CXX = /usr/local/mpich-4.0.2/gcc11/bin/mpic++
# #CXX = g++
# MPICH_DIR = /usr/local/mpich-4.0.2/gcc11
# CUDA_HOME = /usr/local/cuda-11.8
# MKLROOT = /usr/pack/intel_compiler-2020-af/x64/compilers_and_libraries_2019.0.117/linux/mkl
# EIGENROOT = /usr/scratch/mont-fort17/almaeder/libraries/eigen/
# #src = /home/dleonard/Documents/dphpc_sinv/src/dphpc_sinv/PSR/
# src = /home/vmaillou/Documents/dphpc_sinv/src/dphpc_sinv/PSR

# CXXFLAGS = -Wall -Wextra -fopenmp -std=c++14 -I${EIGENROOT} -I${CUDA_HOME}/include -I${MKLROOT}/include -I${MPICH_DIR}/include -I${src}
# NVCCFLAGS = -arch=sm_60 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -std=c++14 -Xcompiler -fopenmp -I${CUDA_HOME}/include
# LDMKLFLAGS = -L${MKLROOT}/lib/intel64 -Wl,--start-group -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -Wl,--end-group -L${MPICH_DIR}/lib -lmpi
# LDFLAGS = -L${CUDA_HOME}/lib64 -lcudart -lcublas -lcusolver -lcuda -lpthread -lm -ldl -std=c++14 -lgomp
# SOURCES= PSR.cpp, PSR_testutils.cpp, PSR_MPI_utils.cpp, PSR_CUDA_utils.cpp, PSR_main.cpp

NVCC = nvcc
#CXX = /usr/local/mpich-4.0.2/gcc11/bin/mpic++
#CXX = g++
MKLROOT = /opt/intel/oneapi/mkl/2021.2.0
EIGENROOT = /users/amaeder/Documents/eigen
#src = /home/dleonard/Documents/dphpc_sinv/src/dphpc_sinv/PSR/
src = /home/vmaillou/Documents/dphpc_sinv/src/dphpc_sinv/PSR

CXXFLAGS = -Wall -Wextra -fopenmp -std=c++14 -I"${MPICH_DIR}/include" -I${EIGENROOT} -I${CUDA_HOME}/include -I${MKLROOT}/include -I${MPICH_DIR}/include -I${src}
NVCCFLAGS = -arch=sm_60 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -std=c++14 -Xcompiler -fopenmp -I${CUDA_HOME}/include
LDMKLFLAGS = -L${MKLROOT}/lib/intel64 -Wl,--start-group -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -Wl,--end-group -L${MPICH_DIR}/lib -lmpich
LDFLAGS = -L"${MPICH_DIR}/lib" -L${CUDA_HOME}/lib64 -lcudart -lcublas -lcusolver -lcuda -lpthread -lm -ldl -std=c++14 -lgomp 
SOURCES= PSR.cpp, PSR_testutils.cpp, PSR_MPI_utils.cpp, PSR_CUDA_utils.cpp, PSR_main.cpp



CPP_SOURCES = $(SOURCES)
CU_SOURCES = $(filter %.cu, $(SOURCES))

#CPP_OBJ_FILES=$(CPP_SOURCES:.cpp=.o)
CPP_OBJ_FILES = PSR.o PSR_testutils.o PSR_MPI_utils.o PSR_CUDA_utils.o PSR_main.o
CU_OBJ_FILES =$(CU_SOURCES:.cu=.o)
BINARY = main.o


.PHONY: all
all: $(BINARY)

# Rule for compiling C++ source files
%.o: %.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@

# Rule for compiling CUDA source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# todo make a nice makefile
# create a nice makefile which compiles all the files to a binary
# and cleans up the mess afterwards
# with build folder and stuff
$(BINARY): $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	$(CXX) $(CXXFLAGS) $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $@ $(LDMKLFLAGS) $(LDFLAGS)




.PHONY: clean
clean:
	rm -f main *.o
