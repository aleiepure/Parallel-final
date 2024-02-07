#! /bin/bash
set -e

module load gcc91
module load mpich-3.2.1--gcc-9.1.0

gcc() {
    gcc-9.1.0 "$@"
}
gcc --version
mpicc --version

# To check the architecture
lscpu

# Select the working directory
cd /home/$USER/final

mkdir -p out

make seq
make omp
make mpi

qsub src/pbs/seq.pbs
qsub src/pbs/omp.pbs
qsub src/pbs/mpi.pbs

watch qstat -u $USER