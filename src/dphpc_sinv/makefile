
CC = nvcc
CUDA_ROOT = /usr/local/cuda
MKLROOT = /usr/pack/intel_compiler-2020-af/x64/compilers_and_libraries_2019.0.117/linux/mkl
CFLAGS = -Xcompiler -Wall -Xcompiler -Wextra -std=c++14 -I${CUDA_ROOT}/include -I"${MKLROOT}/include"
LDFLAGS = -L"${CUDA_ROOT}/lib64" -lcuda -lcudart -lcublas -lcusolver -lcusparse
SOURCES= main.cpp system_solve_benchmark.cu utils.cpp
OBJECTS=$(SOURCES:.cpp=.o)
BINARY = main

.PHONY: all
all: $(SOURCES) $(BINARY)

# todo make a nice makefile
# create a nice makefile which compiles all the files to a binary
# and cleans up the mess afterwards
# with build folder and stuff
$(BINARY): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) -c $< -o $@

.cu.o:
	$(CC) $(CFLAGS) -c $< -o $@


.PHONY: clean
clean:
	rm -f main *.o

	