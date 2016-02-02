#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <immintrin.h>

/* Allows modifying at compile time, e.g. -DNREP=20 */
#ifndef NREP
#define NREP 10
#endif

/* Number of floats that fit in a 256-bit vector */
#define VL 8

/* Speed of light */
#define C (3e8)

/* 
   Structure which holds space-time coords 
*/
typedef struct {
  float __attribute__((aligned(32))) x[VL];
  float __attribute__((aligned(32))) y[VL];
  float __attribute__((aligned(32))) z[VL];
  float __attribute__((aligned(32))) t[VL];
  float __attribute__((aligned(32))) s[VL];
} st_coords;

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
  void *ptr;
  posix_memalign(&ptr, 64, size);
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
	  " Usage: %s <L>\n", argv[0]);
  return;
}

/*
 * Compute: s = sqrt( t**2 - x**2 - y**2 - z**2 ), with s, t, x, y, z
 * member variables of the st_coords structure arr.
 *
 * Traverse elements randomly
 */
void
comp_s(st_coords *arr, int L)
{
  for(int j=0; j<(L/VL); j++) {
    int i = rand() % (L/VL);
    __m256 x = _mm256_load_ps(&arr[i].x[0]);
    __m256 y = _mm256_load_ps(&arr[i].y[0]);
    __m256 z = _mm256_load_ps(&arr[i].z[0]);
    __m256 t = _mm256_load_ps(&arr[i].t[0]);
#ifdef FMA
    register __m256 s0;
    s0 = _mm256_mul_ps(x, x);
    s0 = _mm256_fmadd_ps(y, y, s0);
    s0 = _mm256_fmadd_ps(z, z, s0);
    s0 = _mm256_fmsub_ps(t, t, s0);
    s0 = _mm256_sqrt_ps(s0);
#else
    register __m256 s0, s1;
    s1 = _mm256_mul_ps(x, x);
    s0 = _mm256_mul_ps(y, y);
    s1 = _mm256_add_ps(s0, s1);
    s0 = _mm256_mul_ps(z, z);
    s1 = _mm256_add_ps(s0, s1);
    s0 = _mm256_mul_ps(t, t);
    s1 = _mm256_sub_ps(s0, s1);
    s0 = _mm256_sqrt_ps(s1);
#endif
    
    _mm256_store_ps(&arr[i].s[0], s0);
  }
  return;
}

int
main(int argc, char *argv[])
{
  if(argc != 2) {
    usage(argv);
    return 1;
  }  
  int L = atoi(argv[1]);
  st_coords *arr = alloc(sizeof(st_coords)*(L/VL));
  for(int i=0; i<L/VL; i++)
    for(int j=0; j<VL; j++) {
      arr[i].x[j] = drand48();
      arr[i].y[j] = drand48();
      arr[i].z[j] = drand48();
      arr[i].t[j] = drand48()*C;
    }

  {
    /* Warm-up */
    comp_s(arr, L);
    double t0acc = 0;
    double t1acc = 0;
    int n = 1;
    /* 
       Loop accumulating run-time. Stop when the average time has less
       than a 10% error
    */       
    while(1) {
      double t0 = stop_watch(0);
      for(int i=0; i<NREP; i++)
	comp_s(arr, L);
      t0 = stop_watch(t0)/(double)NREP;
      t0acc += t0;
      t1acc += t0*t0;
      if(n > 2) {
	double ave = t0acc/n;
	double err = sqrt(t1acc/n - ave*ave)/sqrt(n);
	if(err/ave < 0.1) {
	  t0acc = ave;
	  t1acc = err;
	  break;
	}
      }
      n++;
    }
    printf(" Done L = %d, in %3.1e +/- %3.1e secs, %g Mflop/s\n",
	   L, t0acc, t1acc, (double)9*L/1e6/t0acc);
  }
  
  free(arr);
  return 0;
}