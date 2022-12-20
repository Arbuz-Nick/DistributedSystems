/* Include benchmark-specific header. */
#ifndef _2MM_H
#define _2MM_H
// #define MINI_DATASET 1
#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#define MINI_DATASET
#endif
#if !defined(NI) && !defined(NJ) && !defined(NK) && !defined(NL)
#ifdef MINI_DATASET
#define NI 16
#define NJ 18
#define NK 22
#define NL 24
#endif
#ifdef SMALL_DATASET
#define NI 40
#define NJ 50
#define NK 70
#define NL 80
#endif
#ifdef MEDIUM_DATASET
#define NI 180
#define NJ 190
#define NK 210
#define NL 220
#endif
#ifdef LARGE_DATASET
#define NI 800
#define NJ 900
#define NK 1100
#define NL 1200
#endif
#ifdef EXTRALARGE_DATASET
#define NI 1600
#define NJ 1800
#define NK 2200
#define NL 2400
#endif
#endif
#endif

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#include <mpi-ext.h>
#include <sys/types.h>
#include <signal.h>

#define FIRST_THREAD 0
int process_num;
double bench_t_start, bench_t_end;
char *file_path;
int a = 0;
int n = 0;

int rank = MPI_PROC_NULL, verbose = 1; /* makes this global (for printfs) */
MPI_Comm world, reserve;
char **gargv;

int MPIX_Comm_replace(MPI_Comm comm, MPI_Comm *newcomm) {
    MPI_Comm icomm, /* the intercomm between the spawnees and the old (shrinked) world */
    scomm,      /* the local comm for each sides of icomm */
    mcomm;      /* the intracomm, merged from icomm */
    MPI_Group cgrp, sgrp, dgrp;
    int rc, flag, rflag, i, nc, ns, nd, crank, srank, drank;

    redo:
    if (comm == MPI_COMM_NULL) { /* am I a new process? */
        /* I am a new spawnee, waiting for my new rank assignment
         * it will be sent by rank 0 in the old world */
        MPI_Comm_get_parent(&icomm);
        scomm = MPI_COMM_WORLD;
        MPI_Recv(&crank, 1, MPI_INT, 0, 1, icomm, MPI_STATUS_IGNORE);
        if (verbose) {
            MPI_Comm_rank(scomm, &srank);
            printf("Spawnee %d: crank=%d\n", srank, crank);
        }
    } else {
        /* I am a survivor: Spawn the appropriate number
         * of replacement processes (we check that this operation worked
         * before we process further) */
        /* First: remove dead processes */
        MPIX_Comm_shrink(comm, &scomm);
        MPI_Comm_size(scomm, &ns);
        MPI_Comm_size(comm, &nc);
        nd = nc - ns; /* number of deads */
        if (0 == nd) {
            /* Nobody was dead to start with. We are done here */
            MPI_Comm_free(&scomm);
            *newcomm = comm;
            return MPI_SUCCESS;
        }
        /* We handle failures during this function ourselves... */
        MPI_Comm_set_errhandler(scomm, MPI_ERRORS_RETURN);

        rc = MPI_Comm_spawn(gargv[0], &gargv[1], nd, MPI_INFO_NULL,
                            0, scomm, &icomm, MPI_ERRCODES_IGNORE);
        flag = (MPI_SUCCESS == rc);
        MPIX_Comm_agree(scomm, &flag);
        if (!flag) {
            if (MPI_SUCCESS == rc) {
                MPIX_Comm_revoke(icomm);
                MPI_Comm_free(&icomm);
            }
            MPI_Comm_free(&scomm);
            if (verbose)
                fprintf(stderr, "%04d: comm_spawn failed, redo\n", rank);
            goto redo;
        }

        /* remembering the former rank: we will reassign the same
         * ranks in the new world. */
        MPI_Comm_rank(comm, &crank);
        MPI_Comm_rank(scomm, &srank);
        /* the rank 0 in the scomm comm is going to determine the
         * ranks at which the spares need to be inserted. */
        if (0 == srank) {
            /* getting the group of dead processes:
             *   those in comm, but not in scomm are the deads */
            MPI_Comm_group(comm, &cgrp);
            MPI_Comm_group(scomm, &sgrp);
            MPI_Group_difference(cgrp, sgrp, &dgrp);
            /* Computing the rank assignment for the newly inserted spares */
            for (i = 0; i < nd; i++) {
                MPI_Group_translate_ranks(dgrp, 1, &i, cgrp, &drank);
                /* sending their new assignment to all new procs */
                MPI_Send(&drank, 1, MPI_INT, i, 1, icomm);
            }
            MPI_Group_free(&cgrp);
            MPI_Group_free(&sgrp);
            MPI_Group_free(&dgrp);
        }
    }

    /* Merge the intercomm, to reconstruct an intracomm (we check
     * that this operation worked before we proceed further) */
    rc = MPI_Intercomm_merge(icomm, 1, &mcomm);
    rflag = flag = (MPI_SUCCESS == rc);
    MPIX_Comm_agree(scomm, &flag);
    if (MPI_COMM_WORLD != scomm)
        MPI_Comm_free(&scomm);
    MPIX_Comm_agree(icomm, &rflag);
    MPI_Comm_free(&icomm);
    if (!(flag && rflag)) {
        if (MPI_SUCCESS == rc) {
            MPI_Comm_free(&mcomm);
        }
        if (verbose)
            fprintf(stderr, "%d: Intercomm_merge failed, redo\n", rank);
        goto redo;
    }

    /* Now, reorder mcomm according to original rank ordering in comm
     * Split does the magic: removing spare processes and reordering ranks
     * so that all surviving processes remain at their former place */
    rc = MPI_Comm_split(mcomm, 1, crank, newcomm);

    /* Split or some of the communications above may have failed if
     * new failures have disrupted the process: we need to
     * make sure we succeeded at all ranks, or retry until it works. */
    flag = (MPI_SUCCESS == rc);
    MPIX_Comm_agree(mcomm, &flag);
    MPI_Comm_free(&mcomm);
    if (!flag) {
        if (MPI_SUCCESS == rc) {
            MPI_Comm_free(newcomm);
        }
        if (verbose)
            fprintf(stderr, "%d: comm_split failed, redo\n", rank);
        goto redo;
    }

    /* restore the error handler */
    if (MPI_COMM_NULL != comm) {
        MPI_Errhandler errh;
        MPI_Comm_get_errhandler(comm, &errh);
        MPI_Comm_set_errhandler(*newcomm, errh);
    }

    return MPI_SUCCESS;
}

static double rtclock() {
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, NULL);
    if (stat != 0)
        printf("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start() {
    bench_t_start = rtclock();
}

void bench_timer_stop() {
    bench_t_end = rtclock();
}

void bench_timer_print() {
    FILE *fout;
    fout = fopen(file_path, "a+");
    fprintf(fout, "%0.6lf;%d;%d*%d*%d*%d\n", bench_t_end - bench_t_start, process_num, NI, NJ, NK, NL);

    printf("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}

static void init_array(int ni,
                       int nj,
                       int nk,
                       int nl,
                       float *alpha,
                       float *beta,
                       float **A,
                       float **B,
                       float **C,
                       float **D) {
    int i, j;
    *alpha = 1.2;
    *beta = 1.4;
    for (i = 0; i < ni; i++)
        for (j = 0; j < nk; j++) {
            A[i][j] = (float) ((i * j + 1) % ni) / ni;
        }
    for (i = 0; i < nk; i++)
        for (j = 0; j < nj; j++) {
            B[i][j] = (float) (i * (j + 1) % nj) / nj;
        }
    for (i = 0; i < nj; i++)
        for (j = 0; j < nl; j++)
            C[i][j] = (float) ((i * (j + 3) + 1) % nl) / nl;

    for (i = 0; i < ni; i++)
        for (j = 0; j < nl; j++)
            D[i][j] = (float) (i * (j + 2) % nk) / nk;
}

static void print_array(int ni, int nl, float D[ni][nl]) {
    int i, j;
    fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
    fprintf(stderr, "begin dump: %s", "D");
    for (i = 0; i < ni; i++) {
        fprintf(stderr, "\n");
        for (j = 0; j < nl; j++) {
            fprintf(stderr, "%0.2f ", D[i][j]);
        }
    }
    fprintf(stderr, "\nend   dump: %s\n", "D");
    fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void mm(int rank, int n_proc,
               int ni, int nk, int nj,
               float **A,
               float **B,
               float **tmp,
               float alpha) {
    int i, j, k;
    for (i = 0; i < ni; i++)
        for (j = 0; j < nj; j++)
            tmp[i][j] = 0.0f;
    for (i = 0; i < ni; i++)
        for (k = 0; k < nk; ++k)
            for (j = 0; j < nj; j++) {
                tmp[i][j] += alpha * A[i][k] * B[k][j];
                // printf("%d: %2d, %2d, %2d; %.1f\n", rank, i, j, k, alpha);
            }
}

static void kernel_2mm(int ni,
                       int nj,
                       int nk,
                       int nl,
                       int ni_d,
                       float alpha,
                       float beta,
                       float **tmp_AB,
                       float **tmp_ABC,
                       float **A,
                       float **B,
                       float **C,
                       float **D) {
    if (rank == FIRST_THREAD) {
        printf("Nthread = %d\n", process_num);
    }
    // 0------------------------0//
    MPI_Barrier(world);

    mm(rank, process_num, ni_d, nk, nj, A, B, tmp_AB, alpha);
    mm(rank, process_num, ni_d, nj, nl, tmp_AB, C, tmp_ABC, 1);

    for (int i = 0; i < ni_d; i++) {
        for (int j = 0; j < nl; j++) {
            D[i][j] = D[i][j] * beta + tmp_ABC[i][j];
        }
    }
}

int main(int argc, char **argv) {
    setbuf(stdout, 0);
    if (argc == 1) {
        file_path = "./result_mpi_polus.csv";
    } else {
        file_path = argv[1];
    }
    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int nl = NL;
    int ret_code;

    MPI_Init(&argc, &argv);
    MPI_Comm_get_parent(&world);

    if (MPI_COMM_NULL == world) {
        /* First run: Let's create an initial world,
         * a copy of MPI_COMM_WORLD */
        MPI_Comm_dup(MPI_COMM_WORLD, &world);
        MPI_Comm_size(world, &process_num);
        MPI_Comm_rank(world, &rank);
        /* We set an errhandler on world, so that a failure is not fatal anymore. */
        MPI_Comm_set_errhandler(world, MPI_ERRORS_RETURN);
    } else {
        /* I am a spare, lets get the repaired world */
        MPIX_Comm_replace(MPI_COMM_NULL, &world);
        MPI_Comm_size(world, &process_num);
        MPI_Comm_rank(world, &rank);
        /* We set an errhandler on world, so that a failure is not fatal anymore. */
        MPI_Comm_set_errhandler(world, MPI_ERRORS_RETURN);
        // without goto
    }

    // Получаем номер конкретного процесса на котором запущена программа
    MPI_Comm_rank(world, &rank);
    // Получаем количество запущенных процессов
    MPI_Comm_size(world, &process_num);
    float alpha;
    float beta;

    int ni_d = ni / process_num;
    int ni_r = ni % process_num;
    for (int i = 0; i < ni_r; i++) {
        if (i == rank) {
            ni_d++;
        }
    }

    float **tmp_AB;
    tmp_AB = (float **) malloc((ni_d) * sizeof(float *));
    for (int i = 0; i < ni_d; i++) {
        tmp_AB[i] = (float *) malloc(nj * sizeof(float));
    }

    float **tmp_ABC;
    tmp_ABC = (float **) malloc((ni_d) * sizeof(float *));
    for (int i = 0; i < ni_d; i++) {
        tmp_ABC[i] = (float *) malloc(nl * sizeof(float));
    }
    // float A[ni][nk];
    float **A;

    float **B;
    B = (float **) malloc(nk * sizeof(float *));
    for (int i = 0; i < nk; i++) {
        B[i] = (float *) malloc(nj * sizeof(float));
    }

    float **C;
    C = (float **) malloc(nj * sizeof(float *));
    for (int i = 0; i < nj; i++) {
        C[i] = (float *) malloc(nl * sizeof(float));
    }

    float **D;

    float **cols_A;
    cols_A = (float **) malloc((ni_d) * sizeof(float *));
    for (int i = 0; i < ni_d; i++) {
        cols_A[i] = (float *) malloc(nk * sizeof(float));
    }

    float **cols_D;
    cols_D = (float **) malloc((ni_d) * sizeof(float *));
    for (int i = 0; i < ni_d; i++) {
        cols_D[i] = (float *) malloc(nl * sizeof(float));
    }

    A = (float **) malloc((ni) * sizeof(float *));
    for (int i = 0; i < ni; i++) {
        A[i] = (float *) malloc(nk * sizeof(float));
    }

    D = (float **) malloc((ni) * sizeof(float *));
    for (int i = 0; i < ni; i++) {
        D[i] = (float *) malloc(nl * sizeof(float));
    }

    if (rank == 0) {

        init_array(ni, nj, nk, nl, &alpha, &beta, A, B, C, D);

        for (int i = 0; i < ni_d; i++) {
            for (int j = 0; j < nk; j++)
                cols_A[i][j] = A[i][j];
            for (int j = 0; j < nl; j++)
                cols_A[i][j] = D[i][j];
        }
        printf("All data is on main\n");
        fflush(stdout);
    }
    // Send A
    int flag = 1;
    while (flag) {

        for (int i = 0; i < ni / process_num; i++) {
            MPI_Scatter(A[i * process_num], nk, MPI_FLOAT, cols_A[i], nk,
                        MPI_FLOAT, 0, world);
        }
        for (int i = 1; i < ni_r; i++) {
            if (rank == 0)
                MPI_Send(A[process_num * (ni / process_num) + i], nk, MPI_FLOAT, i,
                         0, world);
            else if (rank == i)
                MPI_Recv(cols_A[ni_d - 1], nk, MPI_FLOAT, 0, 0, world,
                         MPI_STATUS_IGNORE);
        }

        // Send D
        for (int i = 0; i < ni / process_num; i++)
            MPI_Scatter(D[i * process_num], nl, MPI_FLOAT, cols_D[i], nl,
                        MPI_FLOAT, 0, world);
        for (int i = 1; i < ni_r; i++) {
            if (rank == 0)
                MPI_Send(D[process_num * (ni / process_num) + i], nl, MPI_FLOAT, i,
                         0, world);
            else if (rank == i)
                MPI_Recv(cols_D[ni_d - 1], nl, MPI_FLOAT, 0, 0, world,
                         MPI_STATUS_IGNORE);
        }

        for (int i = 0; i < nk; i++)
            MPI_Bcast(B[i], nj, MPI_FLOAT, 0, world);
        for (int i = 0; i < nj; i++)
            MPI_Bcast(C[i], nl, MPI_FLOAT, 0, world);
        srand(time(NULL));

        if ((rand() % 5 == 1) && rank != 0) {
            printf("%d: Terminated\n", rank);
            raise(SIGKILL);
        }

        ret_code = MPI_Bcast(&alpha, 1, MPI_FLOAT, 0, world);

        int was_error = ret_code != MPI_SUCCESS;

        MPI_Allreduce(&was_error, &was_error, 1, MPI_INT, MPI_MAX, world);

        if (was_error) {
            printf("Bcast failed, restarting...\n");
            MPIX_Comm_replace(world, &reserve);
            MPI_Comm_free(&world);
            world = reserve;
            continue;
        }

        MPI_Bcast(&beta, 1, MPI_FLOAT, 0, world);

        if (rank == 0)
            printf("%d: Barrier%d\n", rank, a++); // 0
        MPI_Barrier(world);

        if (rank == FIRST_THREAD) {
            bench_timer_start();
        }

        kernel_2mm(ni, nj, nk, nl, ni_d,
                   alpha, beta,
                   tmp_AB, tmp_ABC,
                   cols_A, B, C,
                   cols_D);

        if (rank == 0)
            printf("%d: Barrier%d\n", rank, a++); // 1
        MPI_Barrier(world);

        for (int i = 0; i < ni / process_num; i++) {
            MPI_Gather(cols_D[i], nl, MPI_FLOAT, D[i * process_num], nl, MPI_FLOAT, 0, world);
        }

        for (int i = 1; i < ni_r; i++) {
            if (rank == 0)
                MPI_Recv(D[process_num * (ni / process_num) + i], nl, MPI_FLOAT, i,
                         0, world, MPI_STATUS_IGNORE);
            else if (rank == i)
                MPI_Send(cols_D[ni_d - 1], nl, MPI_FLOAT, 0, 0, world);
        }
        break;
    }
    if (rank == FIRST_THREAD) {
        bench_timer_stop();
        printf("Timer stoped\n");
        bench_timer_print();
    }

//
//    for (int i = 0; i < ni; i++) {
//        free(A[i]);
//        free(D[i]);
//    }
//    free(A);
//    free(D);
//    for (int i = 0; i < nk; i++)
//        free(B[i]);
//
//    free(B);
//    for (int i = 0; i < nj; i++)
//        free(C[i]);
//    free(C);
//    for (int i = 0; i < ni_d; i++) {
//        free(tmp_AB[i]);
//        free(tmp_ABC[i]);
//    }
//    free(tmp_AB);
//    free(tmp_ABC);

    if (rank == 0)
        printf("%d: Barrier%d\n", rank, a++); // 2
    MPI_Barrier(world);
    MPI_Finalize();

    // if (argc > 42 && !strcmp(argv[0], ""))
    //   print_array(ni, nl, *D);

    return 0;
}
