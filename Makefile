# Intro to Parallel Computing, Final Project
# A.Y. 2023/2024
# Authors: Alessandro Iepure, 228023
#		   Lorenzo Fasol, 227561
#          Riccardo Minella, 227326
#
# Makefile

# Compilers
CC = gcc-13
MPICC = mpicc
NVCC = nvcc

# Compilers flags
CFLAGS = -Wall -Wextra -std=c99
NVCCFLAGS = -Wall,-Wextra
MPICCFLAGS = -Wall -Wextra

# Source files
SEQ_SRCS = src/sequential.c
OMP_SRCS = src/main.c
MPI_SRCS = src/main.c
CUDA_SRCS = src/main.c

# Output directory
OUT_DIR = out

# Default target
all: seq omp mpi #cuda

# Compile sequential version
.phony: seq
seq:
	$(CC) $(CFLAGS) $(SEQ_SRCS) -o $(OUT_DIR)/$@

# Compile OpenMP version
.phony: omp
omp:
	$(CC) $(CFLAGS) -fopenmp $(OMP_SRCS) -o $(OUT_DIR)/$@

# Compile MPI version
.phony: mpi
mpi:
	$(MPICC) $(MPICCFLAGS) $(MPI_SRCS) -o $(OUT_DIR)/$@

# Compile CUDA version
.phony: cuda
cuda:
	$(NVCC) $(NVCCFLAGS) $(CUDA_SRCS) -o $(OUT_DIR)/$@

# Clean up object files and executable
.phony: clean
clean:
	rm -rf $(OUT_DIR)/*
