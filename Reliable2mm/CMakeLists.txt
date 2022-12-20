cmake_minimum_required(VERSION 3.22)
project(Distributed_systems C)

set(CMAKE_C_STANDARD 17)

find_package(MPI REQUIRED)

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)
set(MPI_GUESS_LIBRARY_NAME MPICH2)

add_executable(Distributed_systems 2mm_mpi.c)
add_executable(test main2.c)