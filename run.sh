#! /bin/bash

# Abort on error
set -e

# Load required modules
module load gcc91
module load mpich-3.2.1--gcc-9.1.0
module load cuda-11.3

# Workaround cluster issues with compiler versions
gcc() {
    gcc-9.1.0 "$@"
}
gcc --version
mpicc --version
nvcc --version

# Change to the working directory
cd /home/$USER/Parallel-final

# Create output directories
mkdir -p out
mkdir -p results

# Compile the code
make all

# Submit the jobs
qsub jobs/seq.pbs

qsub jobs/omp.pbs

qsub jobs/mpi/mpi_np2.pbs
qsub jobs/mpi/mpi_np4.pbs
qsub jobs/mpi/mpi_np8.pbs
qsub jobs/mpi/mpi_np16.pbs
qsub jobs/mpi/mpi_np32.pbs
qsub jobs/mpi/mpi_np64.pbs

qsub jobs/cuda.pbs

watch qstat -u "$USER"
