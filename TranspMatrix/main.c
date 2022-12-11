#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mpi.h>



int main(int argc, char **argv) {
    
    int size = 4;
    
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    size = (int) sqrt((double) size);
    int mat_size[2] = {size, size};
    
    int my_rank, n_proc;

    int ndims = 2;

    int periods[2] = {1, 1};

    MPI_Comm cart_comm;

    MPI_Cart_create(MPI_COMM_WORLD, ndims, mat_size, periods, 1, &cart_comm);

    MPI_Comm_rank(cart_comm, &my_rank);
    MPI_Comm_size(cart_comm, &n_proc);
    //printf("My id = %2d, n_proc = %2d\n", my_rank, n_proc);
    
    srand(my_rank + size);
    int send_buf[size * size], recv_buf[size * size];
    for (int i = 0; i < size * size; i++) {
        send_buf[i] = rand();
        recv_buf[i] = send_buf[i];
    }
    
    
    MPI_Request send_r[size * size];
    
    if (my_rank == 3){
        printf("%d:\n", my_rank);
        for (int i = 0; i < size*size; i++){
            printf("%5d: %5d\n", i, send_buf[i]);
        }
        printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    //printf("My id = %2d, getting my coords\n", my_rank);

    int my_coords[2];
    MPI_Cart_coords(cart_comm, my_rank, ndims, my_coords);
    //printf("My id = %2d, my coords is <%2d, %2d>\n", my_rank, my_coords[0], my_coords[1]);
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            if ((i != 0) || (j != 0)){
                int target_rank, target_coords[2];
                target_coords[0] = (my_coords[0] + i) % size;
                target_coords[1] = (my_coords[1] + j) % size;
                MPI_Cart_rank(cart_comm, target_coords, &target_rank);

                MPI_Isend(&send_buf[target_rank], 1, MPI_INT, target_rank, 0, MPI_COMM_WORLD, &send_r[target_rank]);
            }
        }
    }


    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            if ((i != 0) || (j != 0)){
                int recv_rank, recv_coords[2];
                recv_coords[0] = (my_coords[0] + i) % size;
                recv_coords[1] = (my_coords[1] + j) % size;
                MPI_Cart_rank(cart_comm, recv_coords, &recv_rank);

                MPI_Recv(&recv_buf[recv_rank], 1, MPI_INT, recv_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }   
    }
    for (int i = 0; i < size*size; i++){
        if (i != my_rank)
            MPI_Wait(&send_r[i], MPI_STATUS_IGNORE);
    }
    
    printf("%5d: %5d\n", my_rank, recv_buf[3]);

    MPI_Finalize();
    return 0;
}