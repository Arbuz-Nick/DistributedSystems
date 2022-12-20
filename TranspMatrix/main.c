#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mpi.h>

int main(int argc, char **argv)
{

    int size = 4;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    size = (int)sqrt((double)size);
    int mat_size[2] = {size, size};

    int my_rank, n_proc;

    int ndims = 2;

    int periods[2] = {1, 1};

    MPI_Comm cart_comm;

    MPI_Cart_create(MPI_COMM_WORLD, ndims, mat_size, periods, 1, &cart_comm);

    MPI_Comm_rank(cart_comm, &my_rank);
    MPI_Comm_size(cart_comm, &n_proc);

    srand(my_rank + size);
    int send_buf[size * size][size * size], recv_buf[size * size][size * size], result[size * size];
    for (int i = 0; i < size * size; i++)
    {
        for (int j = 0; j < size * size; j++)
            if (i == my_rank)
            {
                send_buf[i][j] = rand() % 10;
                recv_buf[i][j] = send_buf[i][j];
            }
            else
            {
                send_buf[i][j] = 0;
                recv_buf[i][j] = send_buf[i][j];
            }
    }
    MPI_Request send_r[size * size];

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < size * size; i++)
    {
        result[i] = send_buf[my_rank][i];
    }
    if (my_rank == 9)
    {
        printf("Proc %2d\n", my_rank);
        for (int i = 0; i < size * size; i++)
        {
            printf("\t%7d: %2d\n", i, result[i]);
        }
    }

    int source, dest;
    MPI_Status status;
    int my_coords[2];
    MPI_Cart_coords(cart_comm, my_rank, ndims, my_coords);
    int counter = 1;

    // Step 1
    if (my_coords[0] == 0 || my_coords[0] == 3)
    {
        if (my_coords[0] == 0)
            MPI_Cart_shift(cart_comm, 0, 1, &source, &dest);
        else
            MPI_Cart_shift(cart_comm, 0, -1, &source, &dest);
        MPI_Send(send_buf, size * size * size * size, MPI_INT, dest, 0, MPI_COMM_WORLD);
    }
    else
    {
        if (my_coords[0] == 1)
            MPI_Cart_shift(cart_comm, 0, 1, &source, &dest);
        else
            MPI_Cart_shift(cart_comm, 0, -1, &source, &dest);
        MPI_Recv(recv_buf, size * size * size * size, MPI_INT, source, 0, MPI_COMM_WORLD, &status);
        for (int i = 0; i < size * size; i++)
        {
            for (int j = 0; j < size * size; j++)
            {
                if (recv_buf[i][j] != 0 && send_buf[i][j] == 0)
                {
                    send_buf[i][j] = recv_buf[i][j];
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Step 2

    if (my_coords[0] == 1 || my_coords[0] == 2)
    {
        if (my_coords[1] == 0 || my_coords[1] == 3)
        {
            if (my_coords[1] == 0)
                MPI_Cart_shift(cart_comm, 1, 1, &source, &dest);
            else
                MPI_Cart_shift(cart_comm, 1, -1, &source, &dest);

            MPI_Send(send_buf, size * size * size * size, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }
        else
        {
            if (my_coords[1] == 1)
                MPI_Cart_shift(cart_comm, 1, 1, &source, &dest);
            else
                MPI_Cart_shift(cart_comm, 1, -1, &source, &dest);

            MPI_Recv(recv_buf, size * size * size * size, MPI_INT, source, 0, MPI_COMM_WORLD, &status);

            for (int i = 0; i < size * size; i++)
            {
                for (int j = 0; j < size * size; j++)
                {
                    if (recv_buf[i][j] != 0 && send_buf[i][j] == 0)
                    {
                        send_buf[i][j] = recv_buf[i][j];
                    }
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Step 3-4
    for (int i = 0; i < 2; i++)
    {
        if ((my_coords[0] == 1 || my_coords[0] == 2) && (my_coords[1] == 1 || my_coords[1] == 2))
        {
            if (my_coords[0] == 1)
                MPI_Cart_shift(cart_comm, 0, 1, &source, &dest);
            else
                MPI_Cart_shift(cart_comm, 0, -1, &source, &dest);
            MPI_Sendrecv(send_buf, size * size * size * size, MPI_INT, dest, 0,
                         recv_buf, size * size * size * size, MPI_INT, dest, 0,
                         MPI_COMM_WORLD, &status);
            for (int i = 0; i < size * size; i++)
            {
                for (int j = 0; j < size * size; j++)
                {
                    if (recv_buf[i][j] != 0 && send_buf[i][j] == 0)
                    {
                        send_buf[i][j] = recv_buf[i][j];
                    }
                }
            }
            if (my_coords[1] == 1)
                MPI_Cart_shift(cart_comm, 1, 1, &source, &dest);
            else
                MPI_Cart_shift(cart_comm, 1, -1, &source, &dest);
            MPI_Sendrecv(send_buf, size * size * size * size, MPI_INT, dest, 0,
                         recv_buf, size * size * size * size, MPI_INT, dest, 0,
                         MPI_COMM_WORLD, &status);
            for (int i = 0; i < size * size; i++)
            {
                for (int j = 0; j < size * size; j++)
                {
                    if (recv_buf[i][j] != 0 && send_buf[i][j] == 0)
                    {
                        send_buf[i][j] = recv_buf[i][j];
                    }
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // Step 5

    if ((my_coords[0] != 0 && my_coords[0] != 3) || (my_coords[1] != 0 && my_coords[1] != 3))
    {

        if (my_coords[0] == 0 || my_coords[0] == 3 || my_coords[1] == 0 || my_coords[1] == 3)
        {

            if (my_coords[0] == 0)
                MPI_Cart_shift(cart_comm, 0, -1, &source, &dest);
            else if (my_coords[0] == 3)
                MPI_Cart_shift(cart_comm, 0, 1, &source, &dest);
            else if (my_coords[1] == 0)
                MPI_Cart_shift(cart_comm, 1, -1, &source, &dest);
            else
                MPI_Cart_shift(cart_comm, 1, 1, &source, &dest);
            MPI_Recv(recv_buf, size * size * size * size, MPI_INT, source, 0, MPI_COMM_WORLD, &status);
            for (int i = 0; i < size * size; i++)
            {
                for (int j = 0; j < size * size; j++)
                {
                    if (recv_buf[i][j] != 0 && send_buf[i][j] == 0)
                    {
                        send_buf[i][j] = recv_buf[i][j];
                    }
                }
            }
        }
        else if (my_coords[0] == 1)
            if (my_coords[1] == 1)
            {
                MPI_Cart_shift(cart_comm, 0, -1, &source, &dest);
                MPI_Send(send_buf, size * size * size * size, MPI_INT, dest, 0, MPI_COMM_WORLD);
                MPI_Cart_shift(cart_comm, 1, -1, &source, &dest);
                MPI_Send(send_buf, size * size * size * size, MPI_INT, dest, 0, MPI_COMM_WORLD);
            }
            else
            {
                MPI_Cart_shift(cart_comm, 0, -1, &source, &dest);
                MPI_Send(send_buf, size * size * size * size, MPI_INT, dest, 0, MPI_COMM_WORLD);
                MPI_Cart_shift(cart_comm, 1, 1, &source, &dest);
                MPI_Send(send_buf, size * size * size * size, MPI_INT, dest, 0, MPI_COMM_WORLD);
            }
        else if (my_coords[1] == 1)
        {
            MPI_Cart_shift(cart_comm, 0, 1, &source, &dest);
            MPI_Send(send_buf, size * size * size * size, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Cart_shift(cart_comm, 1, -1, &source, &dest);
            MPI_Send(send_buf, size * size * size * size, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Cart_shift(cart_comm, 0, 1, &source, &dest);
            MPI_Send(send_buf, size * size * size * size, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Cart_shift(cart_comm, 1, 1, &source, &dest);
            MPI_Send(send_buf, size * size * size * size, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Step 6
    if (my_coords[0] == 0 || my_coords[0] == 3)
    {
        if (my_coords[1] == 0 || my_coords[1] == 3)
        {
            if (my_coords[1] == 0)
                MPI_Cart_shift(cart_comm, 1, -1, &source, &dest);
            else
                MPI_Cart_shift(cart_comm, 1, 1, &source, &dest);
            MPI_Recv(recv_buf, size * size * size * size, MPI_INT, source, 0, MPI_COMM_WORLD, &status);
            for (int i = 0; i < size * size; i++)
            {
                for (int j = 0; j < size * size; j++)
                {
                    if (recv_buf[i][j] != 0 && send_buf[i][j] == 0)
                    {
                        send_buf[i][j] = recv_buf[i][j];
                    }
                }
            }
        }
        else
        {
            if (my_coords[1] == 1)
                MPI_Cart_shift(cart_comm, 1, -1, &source, &dest);
            else
                MPI_Cart_shift(cart_comm, 1, 1, &source, &dest);
            MPI_Send(send_buf, size * size * size * size, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }
    }

    for (int i = 0; i < size * size; i++)
    {
        result[i] = send_buf[i][my_rank];
    }
    if (my_rank == 0)
        printf("\n");
    MPI_Barrier(MPI_COMM_WORLD);
    printf("\tRank %2d: %2d\n", my_rank, result[9]);
    MPI_Finalize();
    return 0;
}