#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <sys/types.h>

int main(int argc, char *argv[])
{
    printf("Hello\n");
    int *a;
    MPI_Init(&argc, &argv);
    int process_id;
    // Получаем номер конкретного процесса на котором запущена программа
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    a = (int*) malloc(4*sizeof(int));
    if (process_id == 0){
        MPI_Send(a, 4, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(a, 4, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    free(a);
    printf("Here\n");
    MPI_Finalize();
    printf("Here\n");
    free(a);
    return 0;
}
