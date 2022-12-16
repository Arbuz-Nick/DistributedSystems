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
                       float **D)
{
  int i, j;
  *alpha = 1.2;
  *beta = 1.4;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
    {
      A[i][j] = (float)((i * j + 1) % ni) / ni;
    }
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
    {
      B[i][j] = (float)(i * (j + 1) % nj) / nj;
    }
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
               int ni, int nk, int nj,
               float **A,
               float **B,
               float **tmp,
               float alpha)
{
  int i, j, k;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      tmp[i][j] = 0.0f;
  for (i = 0; i < ni; i++)
    for (k = 0; k < nk; ++k)
      for (j = 0; j < nj; j++)
      {
        tmp[i][j] += alpha * A[i][k] * B[k][j];
        // printf("%d: %2d, %2d, %2d; %.1f\n", process_id, i, j, k, alpha);
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
                       float **D)
{
  if (process_id == FIRST_THREAD)
  {
    printf("Nthread = %d\n", process_num);
  }
  // 0------------------------0//
  MPI_Barrier(MPI_COMM_WORLD);

  mm(process_id, process_num, ni_d, nk, nj, A, B, tmp_AB, alpha);
  mm(process_id, process_num, ni_d, nj, nl, tmp_AB, C, tmp_ABC, 1);


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

  int ni_d = ni / process_num;
  int ni_r = ni % process_num;
  for (int i = 0; i < ni_r; i++)
  {
    if (i == process_id)
    {
      ni_d++;
    }
  }

  float **tmp_AB;
  tmp_AB = (float **)malloc((ni_d) * sizeof(float *));
  for (int i = 0; i < ni_d; i++)
  {
    tmp_AB[i] = (float *)malloc(nj * sizeof(float));
  }

  float **tmp_ABC;
  tmp_ABC = (float **)malloc((ni_d) * sizeof(float *));
  for (int i = 0; i < ni_d; i++)
  {
    tmp_ABC[i] = (float *)malloc(nl * sizeof(float));
  }
  // float A[ni][nk];
  float **A;

  float **B;
  B = (float **)malloc(nk * sizeof(float *));
  for (int i = 0; i < nk; i++)
  {
    B[i] = (float *)malloc(nj * sizeof(float));
  }

  float **C;
  C = (float **)malloc(nj * sizeof(float *));
  for (int i = 0; i < nj; i++)
  {
    C[i] = (float *)malloc(nl * sizeof(float));
  }

  float **D;

  float **cols_A;
  cols_A = (float **)malloc((ni_d) * sizeof(float *));
  for (int i = 0; i < ni_d; i++)
  {
    cols_A[i] = (float *)malloc(nk * sizeof(float));
  }

  float **cols_D;
  cols_D = (float **)malloc((ni_d) * sizeof(float *));
  for (int i = 0; i < ni_d; i++)
  {
    cols_D[i] = (float *)malloc(nl * sizeof(float));
  }

  A = (float **)malloc((ni) * sizeof(float *));
  for (int i = 0; i < ni; i++)
  {
    A[i] = (float *)malloc(nk * sizeof(float));
  }

  D = (float **)malloc((ni) * sizeof(float *));
  for (int i = 0; i < ni; i++)
  {
    D[i] = (float *)malloc(nl * sizeof(float));
  }

  if (process_id == 0)
  {

    init_array(ni, nj, nk, nl, &alpha, &beta, A, B, C, D);

    for (int i = 0; i < ni_d; i++)
    {
      for (int j = 0; j < nk; j++)
        cols_A[i][j] = A[i][j];
      for (int j = 0; j < nl; j++)
        cols_A[i][j] = D[i][j];
    }
    printf("All data is on main\n");
  }
  // Send A

  for (int i = 0; i < ni / process_num; i++)
  {
    MPI_Scatter(A[i * process_num], nk, MPI_FLOAT, cols_A[i], nk,
                MPI_FLOAT, 0, MPI_COMM_WORLD);
  }
  for (int i = 1; i < ni_r; i++)
  {
    if (process_id == 0)
      MPI_Send(A[process_num * (ni / process_num) + i], nk, MPI_FLOAT, i,
               0, MPI_COMM_WORLD);
    else if (process_id == i)
      MPI_Recv(cols_A[ni_d - 1], nk, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  }

  // Send D
  for (int i = 0; i < ni / process_num; i++)
    MPI_Scatter(D[i * process_num], nl, MPI_FLOAT, cols_D[i], nl,
                MPI_FLOAT, 0, MPI_COMM_WORLD);
  for (int i = 1; i < ni_r; i++)
  {
    if (process_id == 0)
      MPI_Send(D[process_num * (ni / process_num) + i], nl, MPI_FLOAT, i,
               0, MPI_COMM_WORLD);
    else if (process_id == i)
      MPI_Recv(cols_D[ni_d - 1], nl, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  }

  for (int i = 0; i < nk; i++)
    MPI_Bcast(B[i], nj, MPI_FLOAT, 0, MPI_COMM_WORLD);
  for (int i = 0; i < nj; i++)
    MPI_Bcast(C[i], nl, MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Bcast(&alpha, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&beta, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  if (process_id == 0) printf("%d: Barrier%d\n", process_id, a++); // 0
  MPI_Barrier(MPI_COMM_WORLD);

  if (process_id == FIRST_THREAD)
  {
    bench_timer_start();
  }

  kernel_2mm(ni, nj, nk, nl, ni_d,
             alpha, beta,
             tmp_AB, tmp_ABC,
             cols_A, B, C,
             cols_D);

  if (process_id == 0) printf("%d: Barrier%d\n", process_id, a++); // 1
  MPI_Barrier(MPI_COMM_WORLD);

  for (int i = 0; i < ni / process_num; i++)
  {
    MPI_Gather(cols_D[i], nl, MPI_FLOAT, D[i * process_num], nl, MPI_FLOAT, 0, MPI_COMM_WORLD);
  }

  for (int i = 1; i < ni_r; i++)
  {
    if (process_id == 0)
      MPI_Recv(D[process_num * (ni / process_num) + i], nl, MPI_FLOAT, i,
               0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    else if (process_id == i)
      MPI_Send(cols_D[ni_d - 1], nl, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  }

  if (process_id == FIRST_THREAD)
  {
    bench_timer_stop();
    printf("Timer stoped\n");
    bench_timer_print();
  }
  for (int i = 0; i < ni; i++)
  {
    free(A[i]);
    free(D[i]);
  }
  free(A);
  free(D);
  for (int i = 0; i < nk; i++)
    free(B[i]);

  free(B);
  for (int i = 0; i < nj; i++)
    free(C[i]);
  free(C);
  for (int i = 0; i < ni_d; i++)
  {
    free(tmp_AB[i]);
    free(tmp_ABC[i]);
  }
  free(tmp_AB);
  free(tmp_ABC);

  if (process_id == 0) printf("%d: Barrier%d\n", process_id, a++); // 2
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  // if (argc > 42 && !strcmp(argv[0], ""))
  //   print_array(ni, nl, *D);

  return 0;
}
