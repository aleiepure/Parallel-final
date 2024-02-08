# Intro to Parallel Computing, Final Project
# A.Y. 2023/2024
# Authors: Alessandro Iepure, 228023
#		   Lorenzo Fasol, 227561
#          Riccardo Minella, 227326
#
# Makefile

# Compilers
ifeq ($(shell uname), Darwin)
CC = gcc-13
MPICC = mpicc
else
CC = gcc
MPICC = mpicc
NVCC = nvcc
endif

# Compilers flags
CFLAGS = -Wall -std=c99
NVCCFLAGS = -Wall -std=c99

# Source files
SEQ_SRCS = src/sequential.c
OMP_SRCS = src/omp.c
MPI_SRCS = src/mpi.c
CUDA_SRCS = src/cuda.cu

# Output directory
OUT_DIR = out

# Default target
all: seq omp mpi cuda

# Compile sequential version
.phony: seq
seq:
	$(CC) $(CFLAGS) $(SEQ_SRCS) -o $(OUT_DIR)/$@ -lm

# Compile OpenMP version
.phony: omp
omp:
	$(CC) $(CFLAGS) -fopenmp $(OMP_SRCS) -o $(OUT_DIR)/$@ -lm

# Compile MPI version
.phony: mpi
mpi:
ifdef MPICC
	$(MPICC) $(CFLAGS) $(MPI_SRCS) -o $(OUT_DIR)/$@ -lm
else
	@echo "MPICC not found. Skipping MPI compilation."
endif

# Compile CUDA version
.phony: cuda
cuda:
ifdef NVCC
	$(NVCC) $(NVCCFLAGS) $(CUDA_SRCS) -o $(OUT_DIR)/$@
else
	@echo "NVCC not found. Skipping CUDA compilation."
endif

# Clean up object files and executable
.phony: clean
clean:
	rm -rf $(OUT_DIR)/*
