# Introduction to Parallel Computing (A.Y. 2023-2024) - Final Project

This repo contains the solution of the [final project](./IntroParcoFinalProjects.pdf) of the course "Introduction to Parallel Computing - prof. Vella" from the University of Trento.

## Build and run
### Automatic (HPC Cluster)
To run the code, clone the repo and follow these commands:
```shell
git clone https://github.com/aleiepure/Parallel-final
cd Parallel-final
./run.sh
```
The executed script takes care of compilation, folder creation and submission to the cluster's scheduler.

### Manual
To run outside the cluster, you can compile the code via the provided Makefile.

> [!IMPORTANT]
> Make sure to create both `out/` and `results/` folders before compiling and executing respectively.

> [!IMPORTANT]
> To compile the OpenMP version, make sure to have OpenMP installed.

> [!IMPORTANT]
> To compile the MPI version, make sure to have an MPI distribution installed such 
> as OpenMPI, MPICH or similar.

> [!IMPORTANT]
> To compile the CUDA version, make sure to have an NVIDIA GPU and the CUDA Toolkit 
> installed.

The possible make targets are:
| Target | Description                                                  | 
|--------|--------------------------------------------------------------|
| seq    | Builds the sequential algorithm only as `out/seq`            |
| omp    | Builds the OpenMP algorithm only as `out/omp`                |
| mpi    | Builds the MPI algorithm only as `out/mpi`                   |
| cuda   | Builds the CUDA algorithm only as `out/cuda`                 |
| all    | Builds all the algorithms. See above for the binaries names. |
| clean  | Deletes all previous compiled binaries from `out/`           |

To execute run 
|   Version  | Command |
|------------|-------------------------------------------------------------------|
| Sequential | `./out/seq <path to an image>`                                    |
|   OpenMP   | `OMP_NUM_THREADS=<number of threads> ./out/omp <path to an image>`|
|    MPI     | `mpiexec -np <number of cores> ./out/mpi <path to an image>`      |
|   CUDA     | `./out/cuda <path to an image>`                                   |

## Results
Resulting images will be available under `results/` after execution. A report analyzing the results is available [here](report/report.pdf).
