mpicc main.c -o main -lm
mpiexec -n $1 ./main 16