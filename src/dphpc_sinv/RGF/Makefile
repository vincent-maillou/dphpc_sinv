NVCC = nvcc
CXX = /usr/bin/g++
CUDA_ROOT = /opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2
MKLROOT = /usr/pack/intel_compiler-2020-af/x64/compilers_and_libraries_2019.0.117/linux/mkl
EIGENROOT = /home/vincent-maillou/Documents/Libraries/eigen/

CXXFLAGS = -Wall -Wextra -fopenmp -std=c++14 -I${EIGENROOT}
NVCCFLAGS = -arch=sm_60 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -std=c++14 -Xcompiler -fopenmp -I${CUDA_ROOT}/include
LDFLAGS =
SOURCES= main.cpp utils.cpp

CPP_SOURCES = $(filter %.cpp, $(SOURCES))
CU_SOURCES = $(filter %.cu, $(SOURCES))

CPP_OBJ_FILES=$(CPP_SOURCES:.cpp=.o)
CU_OBJ_FILES =$(CU_SOURCES:.cu=.o)
BINARY = main


.PHONY: all
all: $(BINARY)

# todo make a nice makefile
# create a nice makefile which compiles all the files to a binary
# and cleans up the mess afterwards
# with build folder and stuff
$(BINARY): $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	$(CXX) $(CXXFLAGS) $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $@ $(LDFLAGS)


# Rule for compiling C++ source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for compiling CUDA source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@



.PHONY: clean
clean:
	rm -f main *.o

	