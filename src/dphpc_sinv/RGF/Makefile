NVCC = nvcc
CXX = /usr/bin/g++
CUDA_ROOT = /opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2
MKLROOT = /usr/pack/intel_compiler-2020-af/x64/compilers_and_libraries_2019.0.117/linux/mkl
EIGENROOT = /home/vincent-maillou/Documents/Libraries/eigen/

CXXFLAGS = -Wall -Wextra -fopenmp -std=c++14 -I${EIGENROOT}
NVCCFLAGS = -arch=sm_60 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -std=c++14 -Xcompiler -fopenmp -I${CUDA_ROOT}/include
LDFLAGS =
SOURCES = dense_rgf.cpp utils.cpp

CPP_SOURCES = $(filter %.cpp, $(SOURCES))
CU_SOURCES = $(filter %.cu, $(SOURCES))

CPP_OBJ_FILES=$(CPP_SOURCES:.cpp=.o)
CU_OBJ_FILES =$(CU_SOURCES:.cu=.o)

OUTPUT = dense_rgf sparse_rgf


.PHONY: all
all: $(OUTPUT)

dense_rgf: dense_rgf.o utils.o
	$(CXX) $(CXXFLAGS) dense_rgf.o utils.o -o $@

sparse_rgf: sparse_rgf.o utils.o
	$(CXX) $(CXXFLAGS) sparse_rgf.o utils.o -o $@

# Rule for compiling C++ source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for compiling CUDA source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@


.PHONY: clean
clean:
	rm -f $(OUTPUT) *.o

	