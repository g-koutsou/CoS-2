#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

/*
 * Block length in M direction
 */
#ifndef BM
#define BM 16
#endif

/*
 * Block length in N direction
 */
#ifndef BN
#define BN 16
#endif


/*
 * Returns seconds elapsed since t0
 */
double
stop_watch(double t0)
{
  struct timeval tp;
  gettimeofday(&tp, NULL);
  double t1 = tp.tv_sec + tp.tv_usec*1e-6;  
  return t1-t0;
}

/*
 * Allocate memory with minimal error detection
 */
void *
alloc(size_t size)
{
  void *ptr = malloc(size);
  if(ptr == NULL) {
    fprintf(stderr, "malloc() returned NULL, quitting\n");
    exit(3);
  }
  return ptr;
}

/*
 * Print usage (to stderr)
 */
void
usage(char *argv[])
{
  fprintf(stderr,
	  " Usage: %s <M> <N>\n",
	  argv[0]);
  return;
}

/*
 * Initialize a matrix of size MxN with random numbers
 */
void
rand_mat(double *A, int M, int N)
{
  for(int i=0; i<M; i++)
    for(int j=0; j<N; j++)
      A[j + i*N] = drand48();
  
  return;
}

/*
 * Initialize a matrix of size MxN with zeros
 */
void
zero_mat(double *A, int M, int N)
{
  for(int i=0; i<M; i++)
    for(int j=0; j<N; j++)
      A[j + i*N] = 0.0;
  
  return;
}

/*
 * Matrix multiplication of C = A*B, with A: MxN and B: NxM
 */
void
mat_mul(double *C, int M, int N, double *A, double *B)
{
  for(int i=0; i<M; i++) {
    for(int j=0; j<M; j++)
      C[i*M + j] = 0;

    /*
      _TODO_A_
      
      Implement the matrix-matrix multiplication naively
     */
  }
  return;
}

#ifdef BLCK
/*
 * Matrix multiplication of C = A*B, with A: MxN and B: NxM and
 * blocking of block length BM in M direction and BN in N direction
 */
void
mat_mul_blocked(double *C, int M, int N, double *A, double *B)
{
  double Ab[BM*BN];
  double Bb[BM*BN];
  double Cb[BM*BM];
  for(int i=0; i<M; i+=BM)
    for(int j=0; j<M; j+=BM) {

      /* Set Cb to zero */
      for(int ib=0; ib<BM; ib++)	
	for(int jb=0; jb<BM; jb++) {
	  Cb[ib*BM + jb] = 0;
	}
      
      /*
       * _TODO_B_
       *
       * Implement the blocked matrix-matrix multiplication
       * Steps:
       * 
       * 0. Loop over k-blocks
       * 1. Fill Ab with the block that begins from A[i:][k:]
       * 2. Fill Bb with the block that begins from B[k:][j:]
       * 3. Do the little matrix matrix multiplication Cb = Ab*Bb
       *
       */
      
      /*
       * Copy Cb to C[i][j];
       */

      for(int ib=0; ib<BM; ib++)
	for(int jb=0; jb<BM; jb++) {
	  C[(i+ib)*M + (j+jb)] = Cb[ib*BM + jb];
	}      
    }
  return;
}
#endif

int
main(int argc, char *argv[])
{
  int nargs = 3;  
  if(argc != nargs) {
    usage(argv);
    return 1;
  } 
  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  
  double *A = alloc(sizeof(double)*M*N);
  double *B = alloc(sizeof(double)*M*N);

  rand_mat(A, M, N);
  rand_mat(B, N, M);

  double *C = alloc(sizeof(double)*M*M);
  zero_mat(C, M, M);
  
  mat_mul(C, M, N, A, B);
  {
    double t0 = stop_watch(0);
    mat_mul(C, M, N, A, B);
    t0 = stop_watch(t0);
    double beta_fp = 0 /* _TODO_A_ calculate beta_fp from timing t0 */;
    printf(" ORIG: M = %d, N = %d,", M, N);
    printf(" took: %4.2e sec,", t0);
    printf(" P = %4.2e Mflop/s\n", beta_fp);
  }

#ifdef BLCK
  double *Cb = alloc(sizeof(double)*M*M);
  zero_mat(Cb, M, M);

  mat_mul_blocked(Cb, M, N, A, B);
  {
    double t0 = stop_watch(0);
    mat_mul_blocked(Cb, M, N, A, B);
    t0 = stop_watch(t0);
    double beta_fp = 0 /* _TODO_A_ calculate beta_fp from timing t0 */;    
    printf(" BLCK: M = %d, N = %d,", M, N);
    printf(" took: %4.2e sec,", t0);
    printf(" P = %4.2e Mflop/s, BM = %d, BN = %d\n", beta_fp, BM, BN);
  }
#endif
  
#ifdef BLCK
  double eps = 1e-12;
  double diff = 0;
  for(int i=0; i<M*M; i++) {
    diff += fabs((C[i] - Cb[i])/C[i]);
  }
  /*
   * If the difference between the flat and blocked result is larger
   * than eps, complain to stdout and write the two matrices to file
   * "diffs.out".
   */
  diff /= (double)M*M;
  if(diff > eps) {
    printf(" Non zero diff: %e\n", diff);
    FILE *fp = fopen("diffs.out", "w");
    for(int i=0; i<M*M; i++)
      fprintf(fp, "%e\n", fabs((C[i]-Cb[i])/C[i]));
    fclose(fp);
  }
#endif
  
  free(A);
  free(B);
  free(C);

#ifdef BLCK
  free(Cb);
#endif 
  return 0;
}
