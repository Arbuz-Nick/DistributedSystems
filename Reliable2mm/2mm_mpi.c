/* Include benchmark-specific header. */
#include "2mm_mpi.h"
#define FIRST_THREAD 0
int process_num;
int process_id;
double bench_t_start, bench_t_end;
char *file_path;
int a = 0;
int n = 0;
static double rtclock()
{
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, NULL);
  if (stat != 0)
    printf("Error return from gettimeofday: %d", stat);
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start()
{
  bench_t_start = rtclock();
}

void bench_timer_stop()
{
  bench_t_end = rtclock();
}

void bench_timer_print()
{
  FILE *fout;
  fout = fopen(file_path, "a+");
  printf("opened\n");
  fprintf(fout, "%0.6lf;%d;%d*%d*%d*%d\n", bench_t_end - bench_t_start, process_num, NI, NJ, NK, NL);
  printf("file printed\n");

  printf("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}

static void init_array(int ni,
                       int nj,
                       int nk,
                       int nl,
                       float *alpha,
                       float *beta,
                       float A[ni][nk],
                       float B[nk][nj],
                       float C[nj][nl],
                       float D[ni][nl])
{
  int i, j;

  *alpha = 1.2;
  *beta = 1.4;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (float)((i * j + 1) % ni) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (float)(i * (j + 1) % nj) / nj;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      C[i][j] = (float)((i * (j + 3) + 1) % nl) / nl;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (float)(i * (j + 2) % nk) / nk;
}

static void print_array(int ni, int nl, float D[ni][nl])
{
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "D");
  for (i = 0; i < ni; i++)
  {
    fprintf(stderr, "\n");
    for (j = 0; j < nl; j++)
    {
      fprintf(stderr, "%0.2f ", D[i][j]);
    }
  }
  fprintf(stderr, "\nend   dump: %s\n", "D");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void mm(int rank, int n_proc,
               int ni, int nj, int nk,
               float A[ni][nj],
               float B[nj][nk],
               float tmp[ni][nk],
               float alpha)
{
  printf("\t %d: Mul AB\n", process_id);
  int i, j, k;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      tmp[i][j] = 0.0f;
  for (i = 0; i < ni; i++)
    for (k = 0; k < nk; ++k)
      for (j = 0; j < nj; j++)
      {
        // printf("\t %2d, %2d, %2d\n", i, j, k);
        tmp[i][j] += alpha * A[i][k] * B[k][j];
      }
}

static void kernel_2mm(int ni,
                       int nj,
                       int nk,
                       int nl,
                       int ni_d,
                       float alpha,
                       float beta,
                       float tmp_AB[ni_d][nj],
                       float tmp_ABC[ni_d][nl],
                       float A[ni_d][nk],
                       float B[nk][nj],
                       float C[nj][nl],
                       float D[ni_d][nl])
{
  if (process_id == FIRST_THREAD)
  {
    printf("Nthread = %d\n", process_num);
  }
  // 0------------------------0//
  MPI_Barrier(MPI_COMM_WORLD);

  mm(process_id, process_num, ni_d, nk, nj, A, B, tmp_AB, alpha);

  MPI_Barrier(MPI_COMM_WORLD);

  mm(process_id, process_num, ni_d, nj, nl, tmp_AB, C, tmp_ABC, 1);

  MPI_Barrier(MPI_COMM_WORLD);

  for (int i = 0; i < ni_d; i++)
  {
    for (int j = 0; j < nl; j++)
    {
      D[i][j] = D[i][j] * beta + tmp_ABC[i][j];
    }
  }
}

int main(int argc, char **argv)
{
  setbuf(stdout, 0);
  if (argc == 1)
  {
    file_path = "./result_mpi_polus.csv";
  }
  else
  {
    file_path = argv[1];
  }

  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;
  MPI_Init(&argc, &argv);
  // Получаем номер конкретного процесса на котором запущена программа
  MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
  // Получаем количество запущенных процессов
  MPI_Comm_size(MPI_COMM_WORLD, &process_num);
  float alpha;
  float beta;
  printf("PID = %5d ID = %5d\n", getpid(), process_id);
  int ni_d = ni / process_num;
  int ni_r = ni % process_num;
  for (int i = 0; i < ni_r; i++)
  {
    if (i == process_id)
    {
      ni_d++;
    }
  }

  // float(*tmp)[ni][nj];
  // tmp = (float(*)[ni][nj])malloc((ni) * (nj) * sizeof(float));
  float(*tmp_AB)[ni_d][nj];
  tmp_AB = (float(*)[ni_d][nj])malloc((ni_d) * (nj) * sizeof(float));
  float(*tmp_ABC)[ni_d][nl];
  tmp_ABC = (float(*)[ni_d][nl])malloc((ni_d) * (nl) * sizeof(float));

  float(*A)[ni][nk];
  //--------------//
  float(*B)[nk][nj];
  B = (float(*)[nk][nj])malloc((nk) * (nj) * sizeof(float));
  float(*C)[nj][nl];
  C = (float(*)[nj][nl])malloc((nj) * (nl) * sizeof(float));
  float(*D)[ni][nl];
  //--------------//
  float(*cols_A)[ni_d][nk];
  cols_A = (float(*)[ni_d][nk])malloc((ni_d) * (nk) * sizeof(float));
  float(*cols_D)[ni_d][nl];
  cols_D = (float(*)[ni_d][nl])malloc((ni_d) * (nl) * sizeof(float));

  if (process_id == 0)
  {
    A = (float(*)[ni][nk])malloc((ni) * (nk) * sizeof(float));
    D = (float(*)[ni][nl])malloc((ni) * (nl) * sizeof(float));
    init_array(ni, nj, nk, nl, &alpha, &beta, *A, *B, *C, *D);
    for (int i = 0; i < ni_d; i++)
    {
      for (int j = 0; j < nk; j++)
        (*cols_A)[i][j] = (*A)[i][j];
      for (int j = 0; j < nl; j++)
        (*cols_A)[i][j] = (*D)[i][j];
    }
    printf("All data is on main\n");
  }
  printf("%d: Barrier%d\n", process_id, a++); // 0
  if (process_id == FIRST_THREAD)
    scanf("%d", &n);
  MPI_Barrier(MPI_COMM_WORLD);

  // Send A
  for (int i = 0; i < ni / process_num; i++)
    MPI_Scatter(*A + i * process_num * nk, nk, MPI_FLOAT, (*cols_A)[i], nk,
                MPI_FLOAT, 0, MPI_COMM_WORLD);

  printf("%d: Barrier%d\n", process_id, a++); // 1
  if (process_id == FIRST_THREAD)
    scanf("%d", &n);
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 1; i < ni_r; i++)
  {
    if (process_id == 0)
      MPI_Send(*A + nk * process_num * (ni / process_num) + i * nk, nk, MPI_FLOAT, i,
               0, MPI_COMM_WORLD);
    else if (process_id == i)
      MPI_Recv((*cols_A)[ni_d - 1], nk, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  }

  printf("%d: Barrier%d\n", process_id, a++); // 2
  if (process_id == FIRST_THREAD)
    scanf("%d", &n);
  MPI_Barrier(MPI_COMM_WORLD);

  // Send D
  for (int i = 0; i < ni / process_num; i++)
    MPI_Scatter(*D + i * process_num * nl, nl, MPI_FLOAT, *(cols_D)[i], nl,
                MPI_FLOAT, 0, MPI_COMM_WORLD);
  printf("%d: Barrier%d\n", process_id, a++); // 3
  if (process_id == FIRST_THREAD)
    scanf("%d", &n);
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 1; i < ni_r; i++)
  {
    if (process_id == 0)
      MPI_Send(*D + nl * process_num * (ni / process_num) + i * nl, nl, MPI_FLOAT, i,
               0, MPI_COMM_WORLD);
    else if (process_id == i)
      MPI_Recv((*cols_D)[ni_d - 1], nl, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  }
  printf("%d: Barrier%d\n", process_id, a++); // 4
  if (process_id == FIRST_THREAD)
    scanf("%d", &n);
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Bcast(*B, nk * nj, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(*C, nj * nl, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&alpha, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&beta, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  printf("%d: Barrier%d\n", process_id, a++); // 5
  if (process_id == FIRST_THREAD)
    scanf("%d", &n);
  MPI_Barrier(MPI_COMM_WORLD);

  if (process_id == FIRST_THREAD)
  {
    printf("Ready to multiply\n");
    printf("ni: %d; nk: %d; nj: %d; nl: %d; ni_d: %d\n", ni, nk, nj, nl, ni_d);
    bench_timer_start();
  }
  printf("%d: Barrier%d\n", process_id, a++); // 6
  if (process_id == FIRST_THREAD)
    scanf("%d", &n);
  MPI_Barrier(MPI_COMM_WORLD);
  kernel_2mm(ni, nj, nk, nl, ni_d,
             alpha, beta,
             *tmp_AB, *tmp_ABC,
             *cols_A, *B, *C,
             *cols_D);
  printf("%d: Barrier%d\n", process_id, a++); // 7
  if (process_id == FIRST_THREAD)
    scanf("%d", &n);
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < ni / process_num; i++)
    MPI_Gather(*(cols_D)[i], nl, MPI_FLOAT, *D + i * process_num * nl, nl, MPI_FLOAT, 0, MPI_COMM_WORLD);
  printf("%d: Barrier%d\n", process_id, a++); // 8
  if (process_id == FIRST_THREAD)
    scanf("%d", &n);
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 1; i < ni_r; i++)
  {
    if (process_id == 0)
      MPI_Recv(*D + nl * process_num * (ni / process_num) + i * nl, nl, MPI_FLOAT, i,
               0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    else if (process_id == i)
      MPI_Send((*cols_D)[ni_d - 1], nl, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  }
  printf("%d: Barrier%d\n", process_id, a++); // 9
  if (process_id == FIRST_THREAD)
    scanf("%d", &n);
  MPI_Barrier(MPI_COMM_WORLD);

  if (process_id == FIRST_THREAD)
  {
    printf("Done\n");
    bench_timer_stop();
    printf("Timer stoped\n");
    bench_timer_print();
    printf("Free A\n");
    free(A);
    printf("Free d\n");
    free(D);
    // free(cols_A);
    free(B);
    printf("Free C\n");
    free(C);
    printf("Free C\n");
    // free(cols_D);
    printf("Free C\n");
    free(tmp_AB);
    printf("Free ABC\n");
    free(tmp_ABC);
  }
  printf("%d: Barrier%d\n", process_id, a++); // 10
  if (process_id == FIRST_THREAD)
    scanf("%d", &n);
  MPI_Barrier(MPI_COMM_WORLD);
  printf("%d: Ready to finalize\n", process_id); // 9
  MPI_Finalize();

  // if (argc > 42 && !strcmp(argv[0], ""))
  //   print_array(ni, nl, *D);
  printf("The end\n");
  
  return 0;
}
