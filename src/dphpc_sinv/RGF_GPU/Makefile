NVCC = nvcc
CXX = g++
CUDA_ROOT = /usr/local/cuda
MKLROOT = /usr/pack/intel_compiler-2020-af/x64/compilers_and_libraries_2019.0.117/linux/mkl
EIGENROOT = /usr/scratch/mont-fort17/almaeder/libraries/eigen/

CXXFLAGS = -O3 -Wall -Wextra -fopenmp -std=c++14 -I${EIGENROOT} -I${CUDA_ROOT}/include
NVCCFLAGS = -arch=sm_60 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -std=c++14 -Xcompiler -fopenmp -I${CUDA_ROOT}/include
LDFLAGS = -L"${CUDA_ROOT}/lib64" -lcuda -lcudart -lcusolver -lcublas -lcusparse 
SOURCES = benchmark_retarded_main.cpp batched_retarded.cpp single_retarded.cpp utils.cpp
SOURCES_TEST = test_main.cpp single_lesser_greater_retarded.cpp single_lesser_greater.cpp single_retarded.cpp batched_lesser_greater_retarded.cpp batched_lesser_greater.cpp batched_retarded.cpp utils.cpp batched_geam.cu

CPP_SOURCES = $(filter %.cpp, $(SOURCES))
CU_SOURCES = $(filter %.cu, $(SOURCES))
CPP_SOURCES_TEST = $(filter %.cpp, $(SOURCES_TEST))
CU_SOURCES_TEST = $(filter %.cu, $(SOURCES_TEST))

CPP_OBJ_FILES=$(CPP_SOURCES:.cpp=.o)
CU_OBJ_FILES =$(CU_SOURCES:.cu=.o)
CPP_OBJ_FILES_TEST=$(CPP_SOURCES_TEST:.cpp=.o)
CU_OBJ_FILES_TEST =$(CU_SOURCES_TEST:.cu=.o)

BINARY = benchmark_retarded_main

# TOODO build folder and source folder...

.PHONY: all
all: $(BINARY)

$(BINARY): $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	$(CXX) $(CXXFLAGS) $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $@ $(LDFLAGS)

.PHONY: test
test: test_main
	python test_matrix_generation/generate_batched_dense.py -ms 6 -bls 2 -bas 3
	./test_main

test_main: $(CPP_OBJ_FILES_TEST) $(CU_OBJ_FILES_TEST)
	$(CXX) $(CXXFLAGS) $(CPP_OBJ_FILES_TEST) $(CU_OBJ_FILES_TEST) -o $@ $(LDFLAGS)


# Rule for compiling C++ source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for compiling CUDA source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@



.PHONY: clean
clean:
	rm -f  *.o test $(BINARY)

	