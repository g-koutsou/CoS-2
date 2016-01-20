#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <complex.h>

#define ND 3
#define ALIGNMENT 32
#define VEC_LEN 8    /* number of floats that fit into a 256-bit vector reg. */

/*
 * Trivial structure for scalar field
 */
typedef struct {
  float phi[2][VEC_LEN];
} field;

/*
 * Structure for gauge links
 */
typedef struct {
  float u[ND][2][VEC_LEN];
} link;

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
 * malloc with minimal error detection
 */
void *
alloc(size_t size)
{
  void *ptr;
  posix_memalign(&ptr, ALIGNMENT, size);
  if(ptr == NULL) {
    fprintf(stderr, " malloc() returned NULL. Out of memory?\n");
    exit(-1);
  }
  return ptr;
}

/*
 * allocates a new field and returns its starting address
 */
field *
new_field(size_t L)
{
  field *ptr = alloc(sizeof(field)*L*L*(L/VEC_LEN));
  return ptr;
}

/*
 * allocates a new U(1) gauge field and returns its starting address
 */
link *
new_links(size_t L)
{
  link *ptr = alloc(sizeof(link)*L*L*(L/VEC_LEN));
  return ptr;
}

/*
 * Fills u with random entries on the unit circle, gaussianly
 * distributed around 1 + 0*i, using Box-Mueller.
 *
 * The sites are traversed in the same order as the non-vectorized
 * version, so that an identical random gauge field is produced
 * between the codes
 */
void
rand_links(size_t L, link *g)
{
  int lz = L/VEC_LEN;
  for(int z=0; z<L; z++) 
    for(int y=0; y<L; y++) 
      for(int x=0; x<L; x++) {
	int z0 = z % lz;
	int z1 = z / lz;
	int v = x + y*L + z0*L*L;
	for(int d=0; d<ND; d++) {
	  double u0 = drand48();
	  double u1 = drand48();
	  double phi = sqrt(-2*log(u0))*sin(2*M_PI*u1)*M_PI;
	  /* Real part */
	  g[v].u[d][0][z1] = cos(phi);
	  
	  /* Imaginary part */
	  g[v].u[d][1][z1] = sin(phi);
	}
      }
  return;
}


/*
 * Fills x with random entries on the unit circle, gaussianly
 * distributed around 1 + 0*i, using Box-Mueller.
 *
 * The sites are traversed in the same order as the non-vector
 * version, so that an identical random scalar field is produced
 * between the codes
 */
void
rand_field(size_t L, field *p)
{
  int lz = L/VEC_LEN;
  for(int z=0; z<L; z++) 
    for(int y=0; y<L; y++) 
      for(int x=0; x<L; x++) {
	int z0 = z % lz;
	int z1 = z / lz;
	int v = x + y*L + z0*L*L;
	double u0 = drand48();
	double u1 = drand48();
	double phi = sqrt(-2*log(u0))*sin(2*M_PI*u1)*M_PI;
	p[v].phi[0][z1] = cos(phi);
	p[v].phi[1][z1] = sin(phi);
      }
  return;
}

/*
 * Fills x with zeros
 */
void
zero_field(size_t L, field *x)
{
  for(int i=0; i<L*L*(L/VEC_LEN); i++)
    for(int j=0; j<VEC_LEN; j++) {
      x[i].phi[0][j] = 0.0;
      x[i].phi[1][j] = 0.0;
    }
  return;
}

/*
 * Shift the array of length VEC_LEN by one element up
 */
static void
shift_up(float x[VEC_LEN])
{
  float swap = x[0];
  for(int j=0; j<VEC_LEN-1; j++)
    x[j] = x[j+1];
  x[VEC_LEN-1] = swap;
  return;
}

/*
 * Shift the array of length VEC_LEN by one element down
 */
static void
shift_dn(float x[VEC_LEN])
{
  float swap = x[VEC_LEN-1];
  for(int j=0; j<VEC_LEN-1; j++)
    x[VEC_LEN-j-1] = x[VEC_LEN-j-2];
  x[0] = swap;
  return;
}

/*
 * Applies U(1) gauge laplacian to phi_in, with background field u,
 * and returns in phi_out
 */
void
lapl(size_t L, field *out, field *in, link *g)
{
#pragma omp parallel
  {
  float NDx2 = 2*ND;
  int lz = L/VEC_LEN;
#pragma omp for
  for(int z=0; z<lz; z++) {
    int z0 = z*L*L;
    int zp = ((z + lz + 1)%lz)*L*L;
    int zm = ((z + lz - 1)%lz)*L*L;
    for(int y=0; y<L; y++) {
      int y0 = y*L;
      int yp = ((y + 1)%L)*L;
      int ym = ((y + L - 1)%L)*L;
      for(int x=0; x<L; x++) {
	int v000 = x + y0 + z0;
	int vp00 = x + y0 + zp;
	int vm00 = x + y0 + zm;
	int v0p0 = x + yp + z0;
	int v0m0 = x + ym + z0;
	int v00p = (x+1)%L + y0 + z0;
	int v00m = (L+x-1)%L + y0 + z0;

	float *p_v000_re = &(in[v000].phi[0][0]);
	float *p_v000_im = &(in[v000].phi[1][0]);	

	float *q_v000_re = &(out[v000].phi[0][0]);
	float *q_v000_im = &(out[v000].phi[1][0]);	
	
	float *p_v00p_re = &(in[v00p].phi[0][0]);
	float *p_v00p_im = &(in[v00p].phi[1][0]);
	
	float *p_v00m_re = &(in[v00m].phi[0][0]);
	float *p_v00m_im = &(in[v00m].phi[1][0]);

	float *p_v0p0_re = &(in[v0p0].phi[0][0]);
	float *p_v0p0_im = &(in[v0p0].phi[1][0]);
	
	float *p_v0m0_re = &(in[v0m0].phi[0][0]);
	float *p_v0m0_im = &(in[v0m0].phi[1][0]);

	float p_vp00_re[VEC_LEN];
	float p_vp00_im[VEC_LEN];
	
	float p_vm00_re[VEC_LEN];
	float p_vm00_im[VEC_LEN];

	for(int j=0; j<VEC_LEN; j++) {
	  p_vp00_re[j] = in[vp00].phi[0][j];
	  p_vp00_im[j] = in[vp00].phi[1][j];
                      
	  p_vm00_re[j] = in[vm00].phi[0][j];
	  p_vm00_im[j] = in[vm00].phi[1][j];
	}

	float *u_v000_re[ND] = {&(g[v000].u[0][0][0]),
				&(g[v000].u[1][0][0]),
				&(g[v000].u[2][0][0])};
	float *u_v000_im[ND] = {&(g[v000].u[0][1][0]),
				&(g[v000].u[1][1][0]),
				&(g[v000].u[2][1][0])};

	float *u_v00m_re = &(g[v00m].u[2][0][0]);
	float *u_v00m_im = &(g[v00m].u[2][1][0]);
	
	float *u_v0m0_re = &(g[v0m0].u[1][0][0]);
	float *u_v0m0_im = &(g[v0m0].u[1][1][0]);
	
	float u_vm00_re[VEC_LEN];
	float u_vm00_im[VEC_LEN];

	for(int j=0; j<VEC_LEN; j++) {
	  u_vm00_re[j] = g[vm00].u[0][0][j];
	  u_vm00_im[j] = g[vm00].u[0][1][j];
	}

	if(z == 0) {
	  /* _TODO_C_
	   * 
	   * Use shift_dn() or shift_up() to appropriately shift the
	   * necessary structures, when on th boundary
	   */
	}
	
	if(z == lz-1) {
	  /* _TODO_C_
	   * 
	   * Use shift_dn() or shift_up() to appropriately shift the
	   * necessary structures, when on the boundary
	   */
	}
	
	_Complex float pr, pi;
	for(int j=0; j<VEC_LEN; j++) {
	  pr  =
	    p_v00p_re[j]*u_v000_re[2][j] -
	    p_v00p_im[j]*u_v000_im[2][j];

	  pr +=
	    p_v00m_re[j]*u_v00m_re[j] +
	    p_v00m_im[j]*u_v00m_im[j];	    

	  pr +=
	    p_v0p0_re[j]*u_v000_re[1][j] -
	    p_v0p0_im[j]*u_v000_im[1][j];

	  pr +=
	    p_v0m0_re[j]*u_v0m0_re[j] +
	    p_v0m0_im[j]*u_v0m0_im[j];	    

	  pr +=
	    p_vp00_re[j]*u_v000_re[0][j] -
	    p_vp00_im[j]*u_v000_im[0][j];

	  pr +=
	    p_vm00_re[j]*u_vm00_re[j] +
	    p_vm00_im[j]*u_vm00_im[j];	    
	  
	  pi  =
	    p_v00p_re[j]*u_v000_im[2][j] +
	    p_v00p_im[j]*u_v000_re[2][j];

	  pi +=
	    p_v00m_im[j]*u_v00m_re[j] -
	    p_v00m_re[j]*u_v00m_im[j];
	  
	  pi +=
	    p_v0p0_re[j]*u_v000_im[1][j] +
	    p_v0p0_im[j]*u_v000_re[1][j];

	  pi +=
	    p_v0m0_im[j]*u_v0m0_re[j] -
	    p_v0m0_re[j]*u_v0m0_im[j];
	  
	  pi +=
	    p_vp00_re[j]*u_v000_im[0][j] +
	    p_vp00_im[j]*u_v000_re[0][j];

	  pi +=
	    p_vm00_im[j]*u_vm00_re[j] -
	    p_vm00_re[j]*u_vm00_im[j];
	  
	  q_v000_re[j] = NDx2*p_v000_re[j] - pr;
	  q_v000_im[j] = NDx2*p_v000_im[j] - pi;
	}
      }
    }
  }
  }
  return;
}

/*
 * returns y = x^H x for vector x of length L*L*L
 */
double
xdotx(size_t L, field *x)
{
  double y = 0; 
  for(int i=0; i<L*L*(L/VEC_LEN); i++)
    for(int j=0; j<VEC_LEN; j++) {
      float phi_re = x[i].phi[0][j];
      float phi_im = x[i].phi[1][j];      
      y += phi_re*phi_re + phi_im*phi_im;
    }
  return y;
}

/*
 * returns z = x^H y for vectors x, y of length L*L*L
 */
_Complex double
xdoty(size_t L, field *x, field *y)
{
  _Complex double z = 0; 
  for(int i=0; i<L*L*(L/VEC_LEN); i++)
    for(int j=0; j<VEC_LEN; j++) {
      float x_re = x[i].phi[0][j];
      float x_im = x[i].phi[1][j];      
      float y_re = y[i].phi[0][j];
      float y_im = y[i].phi[1][j];      
      z += y_re*x_re + y_im*x_im;
      z += _Complex_I * (x_re*y_im - x_im*y_re);
    }
  return z;
}

/*
 * returns y = x - y for vectors y, x, of length L*L*L
 */
void
xmy(size_t L, field *x, field *y)
{
  for(int i=0; i<L*L*(L/VEC_LEN); i++)
    for(int j=0; j<VEC_LEN; j++) {
      y[i].phi[0][j] = x[i].phi[0][j] - y[i].phi[0][j];
      y[i].phi[1][j] = x[i].phi[1][j] - y[i].phi[1][j];
    }
  return;
}

/*
 * returns y = x for vectors y, x, of length L*L*L
 */
void
xeqy(size_t L, field *x, field *y)
{
  for(int i=0; i<L*L*(L/VEC_LEN); i++)
    for(int j=0; j<VEC_LEN; j++) {
      x[i].phi[0][j] = y[i].phi[0][j];
      x[i].phi[1][j] = y[i].phi[1][j];
    }
  
  return;
}

/*
 * returns y = a*x+y for vectors y, x, of length L*L*L and scalar a
 */
void
axpy(size_t L, _Complex float a, field *x, field *y)
{
  float ar = creal(a);
  float ai = cimag(a);
  for(int i=0; i<L*L*(L/VEC_LEN); i++)
    for(int j=0; j<VEC_LEN; j++) {
      y[i].phi[0][j] = ar*x[i].phi[0][j] - ai*x[i].phi[1][j] + y[i].phi[0][j];
      y[i].phi[1][j] = ar*x[i].phi[1][j] + ai*x[i].phi[0][j] + y[i].phi[1][j];
    }

  return;
}

/*
 * returns y = x+a*y for vectors y, x, of length L*L*L and scalar a
 */
void
xpay(size_t L, field *x, _Complex float a, field *y)
{
  float ar = creal(a);
  float ai = cimag(a);
  for(int i=0; i<L*L*(L/VEC_LEN); i++)
    for(int j=0; j<VEC_LEN; j++) {
      y[i].phi[0][j] = x[i].phi[0][j] + ar*y[i].phi[0][j] - ai*y[i].phi[1][j];
      y[i].phi[1][j] = x[i].phi[1][j] + ar*y[i].phi[1][j] + ai*y[i].phi[0][j];
    }
  
  return;
}

/*
 * Solves lapl(u) x = b, for x, given b, using Conjugate Gradient
 */
void
cg(size_t L, field *x, field *b, link *g)
{
  int max_iter = 100;
  float tol = 1e-9;

  /* Temporary fields needed for CG */
  field *r = new_field(L);
  field *p = new_field(L);
  field *Ap = new_field(L);

  /* Initial residual and p-vector */
  lapl(L, r, x, g);
  xmy(L, b, r);
  xeqy(L, p, r);

  /* Initial r-norm and b-norm */
  float rr = xdotx(L, r);  
  float bb = xdotx(L, b);
  double t_lapl = 0;
  int iter = 0;
  for(iter=0; iter<max_iter; iter++) {
    printf(" %6d, res = %+e\n", iter, rr/bb);
    if(sqrt(rr/bb) < tol)
      break;
    double t = stop_watch(0);
    lapl(L, Ap, p, g);
    t_lapl += stop_watch(t);
    float pAp = xdoty(L, p, Ap);
    float alpha = rr/pAp;
    axpy(L, alpha, p, x);
    axpy(L, -alpha, Ap, r);
    float r1r1 = xdotx(L, r);
    float beta = r1r1/rr;
    xpay(L, r, beta, p);
    rr = r1r1;
  }

  /* Recompute residual after convergence */
  lapl(L, r, x, g);
  xmy(L, b, r);
  rr = xdotx(L, r);

  double beta_fp = 50*((double)L*L*L)/(t_lapl/(double)iter)*1e-9;
  double beta_io = 40*((double)L*L*L)/(t_lapl/(double)iter)*1e-9;
  printf(" Converged after %6d iterations, res = %+e\n", iter, rr/bb);  
  printf(" Time in lapl(): %+6.3e sec/call, %4.2e Gflop/s, %4.2e GB/s\n",
	 t_lapl/(double)iter, beta_fp, beta_io);  

  free(r);
  free(p);
  free(Ap);
  return;
}

/*
 * Usage info
 */
void
usage(char *argv[])
{
  fprintf(stderr, " Usage: %s L\n", argv[0]);
  return;
}

int
main(int argc, char *argv[])
{
  if(argc != 2) {
    usage(argv);
    exit(1);
  }

  char *e;
  size_t L = (int)strtoul(argv[1], &e, 10);
  if(*e != '\0') {
    usage(argv);
    exit(2);
  }

  field *b = new_field(L);
  field *x = new_field(L);
  link *g = new_links(L);

  rand_links(L, g);
  rand_field(L, b);
  zero_field(L, x);

  cg(L, x, b, g);
  
  free(b);
  free(x);
  free(g);
  return 0;
}
