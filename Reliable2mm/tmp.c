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
    MPI_Init(&argc, &argv);
    int process_id;
    // Получаем номер конкретного процесса на котором запущена программа
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    int **a;
    a = (int **)malloc(4 * sizeof(int *));
    for (int i = 0; i < 4; i++)
        a[i] = (int *)malloc(4 * sizeof(int));
    if (process_id == 0)
    {
        for (int i = 0; i < 4; i++)
        {
            a[i][0] = 1;
            a[i][1] = 2;
            a[i][2] = 3;
            a[i][3] = 4;
        }
    }
    //} else {
    //    MPI_Recv(a, 4, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //}
    if (process_id == 2)
        printf("Here\n");
    for (int i = 0; i < 4; i++)
        MPI_Bcast(a[i], 4, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Barrier(MPI_COMM_WORLD);
    if (process_id == 2)
        for (int i = 0; i < 4; i++)
        {
            if (process_id == 2)
                printf("Here\n");
            for (int j = 0; j < 4; j++)
            {
                if (process_id == 2)
                    printf("Here %d\n", a[i][j]);
                printf("%d: a[%d] = %d\n", process_id, i, a[i][j]);
            }
        }
    for (int i = 0; i < 4; i++)
        free(a[i]);
    free(a);
    if (process_id == 2)
        printf("Here\n");
    MPI_Finalize();
    // ßprintf("Here\n");
    // free(a);
    return 0;
}
