#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <complex.h>

#define ND 3
#define ALIGNMENT 32

#define LEXIC_T(z, y, x) ((((x)+Sx)%Sx) + (((y)+ Sy)%Sy)*Sx + (((z)+L)%L)*Sx*Sy)

/*
 * Trivial structure for scalar field
 */
typedef struct {
  _Complex float phi;
} field;

/*
 * Structure for gauge links
 */
typedef struct {
  _Complex float u[ND];
} link;

/*
 * Structure to hold lattice size information
 */
typedef struct {
  size_t L;
  size_t lx;
  size_t ly;
  size_t Sx;
  size_t Sy;
} latparams;

/*
 * Initialises latparams structure, given L, Sx and Sy
 * Checks whether L is divisable by Sx and Sy
 */
latparams
init_latparams(size_t L, size_t Sy, size_t Sx)
{
  latparams lp;
  if(L % Sx != 0) {
    fprintf(stderr, " Sx must devide L\n");
    exit(3);
  }
  if(L % Sy != 0) {
    fprintf(stderr, " Sy must devide L\n");
    exit(3);
  }
  lp.L = L;     /* The length of the 3D box */
  lp.Sy = Sy;   /* The block length along y */
  lp.Sx = Sx;   /* The block length along x */
  lp.lx = L/Sx; /* The number of blocks along x */
  lp.ly = L/Sy; /* The number of blocks along y */
  return lp;
}

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
field **
new_field(latparams s)
{
  size_t L = s.L;
  size_t Sx = s.Sx;
  size_t Sy = s.Sy;
  size_t lx = s.lx;
  size_t ly = s.ly;  
  field *ptr = alloc(sizeof(field)*L*L*L);
  field **ret = alloc(sizeof(field *)*lx*ly);
  for(int i=0; i<ly*lx; i++)
    ret[i] = &(ptr[i*Sx*Sy*L]);
  
  return ret;
}

/*
 * allocates a new U(1) gauge field and returns its starting address
 */
link **
new_links(latparams s)
{
  size_t L = s.L;
  size_t Sx = s.Sx;
  size_t Sy = s.Sy;
  size_t lx = s.lx;
  size_t ly = s.ly;
  link *ptr = alloc(sizeof(link)*L*L*L);
  link **ret = alloc(sizeof(link *)*lx*ly);
  for(int i=0; i<ly*lx; i++)
    ret[i] = &(ptr[i*Sx*Sy*L]);

  return ret;
}

/*
 * Frees memory allocated with new_field() 
 */
void
del_field(field **x)
{
  free(x[0]);  
  free(x);
  return;
}

/*
 * Frees memory allocated with new_links() 
 */
void
del_links(link **x)
{
  free(x[0]);  
  free(x);
  return;
}

/*
 * Fills u with random entries on the unit circle, gaussianly
 * distributed around 1 + 0*i, using Box-Mueller.
 *
 * The sites are traversed in the same order as the non-blockd version,
 * so that an identical random gauge field is produced between the codes
 */
void
rand_links(latparams s, link **g)
{
  size_t L = s.L;
  size_t Sx = s.Sx;
  size_t Sy = s.Sy;
  size_t lx = s.lx;
  size_t ly = s.ly;
  for(int z=0; z<L; z++)
    for(int y=0; y<L; y++)
      for(int x=0; x<L; x++) {
	int xt = x % Sx; /* Coordinates (x, y) within the block */
	int yt = y % Sy;  
	int xs = x / Sx; /* Coordinates (x, y) of the block */
	int ys = y / Sy;
	int xyz = xt + Sx * (yt + Sy * z);  /* Lexic. index within the block */
	int tc = xs + ys*lx;                /* Lexic. index of the block */
	for(int d=0; d<ND; d++) {
	  double u0 = drand48();
	  double u1 = drand48();
	  double phi = sqrt(-2*log(u0))*sin(2*M_PI*u1)*M_PI;
	  g[tc][xyz].u[d] = cos(phi) + _Complex_I*sin(phi);
	}
      }
  return;
}

/*
 * Fills x with random entries on the unit circle, gaussianly
 * distributed around 1 + 0*i, using Box-Mueller
 *
 * The sites are traversed in the same order as the non-blockd version,
 * so that an identical random scalar field is produced between the codes
 */
void
rand_field(latparams s, field **p)
{
  size_t L = s.L;
  size_t Sx = s.Sx;
  size_t Sy = s.Sy;
  size_t lx = s.lx;
  size_t ly = s.ly;
  for(int z=0; z<L; z++)
    for(int y=0; y<L; y++)
      for(int x=0; x<L; x++) {
	int xt = x % Sx; /* Coordinates (x, y) within the block */
	int yt = y % Sy;  
	int xs = x / Sx; /* Coordinates (x, y) of the block */
	int ys = y / Sy;
	int xyz = xt + Sx * (yt + Sy * z);   /* Lexic. index within the block */
	int tc = xs + ys*lx;                 /* Lexic. index of the block */
	double u0 = drand48();
	double u1 = drand48();
	double phi = sqrt(-2*log(u0))*sin(2*M_PI*u1)*M_PI;
	p[tc][xyz].phi = cos(phi) + _Complex_I*sin(phi);
      }
  return;
}

/*
 * Fills x with zeros
 */
void
zero_field(latparams s, field **x)
{
  size_t L = s.L;
  size_t Sx = s.Sx;
  size_t Sy = s.Sy;
  size_t lx = s.lx;
  size_t ly = s.ly;
  for(int i=0; i<lx*ly; i++)
    for(int j=0; j<L*Sx*Sy; j++) {
      x[i][j].phi = 0.0;
    }
  return;
}

/*
 * Applies U(1) gauge laplacian to phi_in, with background field u,
 * and returns in phi_out
 */
void
lapl(latparams s, field **out, field **in, link **g)
{
  size_t L = s.L;
  size_t Sx = s.Sx;
  size_t Sy = s.Sy;
  size_t lx = s.lx;
  size_t ly = s.ly;    
  float NDx2 = 2*ND;
  /*
   * _TODO_B_
   *
   * Use OpenMP to distribute blocks over threads
   */

  /* Loop over blocks
   */
  for(int y=0; y<ly; y++)
    for(int x=0; x<lx; x++) {
      /* Loop over z within tower of blocks, and then over x,y inside
	 block
       */
      int tc = x + lx*y;              /* Lexic. index of block */
      int tc0p = (x+1)%lx + lx*y;     /* Lexic. index of block in +x direction */
      int tc0m = (lx+x-1)%lx + lx*y;  /* Lexic. index of block in -x direction */
      int tcp0 = x + lx*((y+1)%ly);   /* Lexic. index of block in +y direction */
      int tcm0 = x + lx*((y-1+ly)%ly);/* Lexic. index of block in -y direction */
      field *qtc = in[tc];            /* Input field at current block */
      link *gtc = g[tc];              /* Gauge field at current block */
      field *ptc = out[tc];           /* Output field at current block */
      for(int z=0; z<L; z++) {
	int z0 = z*Sx*Sy;
	int zp = ((z + L + 1)%L)*Sx*Sy;
	int zm = ((z + L - 1)%L)*Sx*Sy;
	for(int ty=1; ty<Sy-1; ty++) {
	  int y0 = ty*Sx;
	  int yp = (ty + 1)*Sx;
	  int ym = (ty - 1)*Sx;
	  for(int tx=1; tx<Sx-1; tx++) {
	    int v000 = tx + y0 + z0;
	    int vp00 = tx + y0 + zp;
	    int vm00 = tx + y0 + zm;
	    int v0p0 = tx + yp + z0;
	    int v0m0 = tx + ym + z0;
	    int v00p = tx+1 + y0 + z0;
	    int v00m = tx-1 + y0 + z0;

	    _Complex float p;
	    /*
	     * _TODO_B_
	     *
	     * Complete the laplacian here. Note that in x and y we
	     * loop from 1 to S-2, so the neighbors don't need to be
	     * taken care of here.
	     *
	     * Hint -- you'll need to use:
	     *
	     * qtc[v00p].phi
	     * gtc[v000].u[2]
	     * qtc[v00m].phi
	     * conj(gtc[v00m].u[2])
	     *
	     * qtc[v0p0].phi
	     * gtc[v000].u[1] 
	     * qtc[v0m0].phi
	     * conj(gtc[v0m0].u[1])
	     *
	     * qtc[vp00].phi
	     * gtc[v000].u[0] 
	     * qtc[vm00].phi
	     * conj(gtc[vm00].u[0])
	     *
	     */ 
	    p  = 0; /* Direction x */
	    p += 0; /* Direction y */
	    p += 0; /* Direction z */
	    ptc[v000].phi = NDx2*qtc[v000].phi - p;
	  }
	}
	
	/*
	  tx = 0 and tx = Sx-1 edges
	*/
	for(int ty=1; ty<Sy-1; ty++) {
	  /*
	    tx = 0;
	  */
	  {
	    int y0 = ty*Sx;
	    int yp = (ty + 1)*Sx;
	    int ym = (ty - 1)*Sx;
	    int v000 = y0 + z0;
	    int vp00 = y0 + zp;
	    int vm00 = y0 + zm;
	    int v0p0 = yp + z0;
	    int v0m0 = ym + z0;
	    int v00p = 1 + y0 + z0;
	    /* v00m is in another block, with xt = Sx - 1 in that block */
	    int v00m = Sx - 1 + y0 + z0;
	    _Complex float p;
	    p  = qtc[v00p].phi*gtc[v000].u[2];
	    p += in[tc0m][v00m].phi*conj(g[tc0m][v00m].u[2]);
	    p += qtc[v0p0].phi*gtc[v000].u[1] + qtc[v0m0].phi*conj(gtc[v0m0].u[1]);
	    p += qtc[vp00].phi*gtc[v000].u[0] + qtc[vm00].phi*conj(gtc[vm00].u[0]);
	    ptc[v000].phi = NDx2*qtc[v000].phi - p;
	  }
	  
	  /*
	    tx = Sx-1;
	  */
	  {
	    int y0 = ty*Sx;
	    int yp = (ty + 1)*Sx;
	    int ym = (ty - 1)*Sx;
	    int v000 = Sx-1 + y0 + z0;
	    int vp00 = Sx-1 + y0 + zp;
	    int vm00 = Sx-1 + y0 + zm;
	    int v0p0 = Sx-1 + yp + z0;
	    int v0m0 = Sx-1 + ym + z0;
	    int v00m = Sx-2 + y0 + z0;
	    /* v00p is in another block, with xt = 0 in that block */
	    int v00p = 0 + y0 + z0;
	    _Complex float p;
	    p  = in[tc0p][v00p].phi*gtc[v000].u[2];
	    p += qtc[v00m].phi*conj(gtc[v00m].u[2]);
	    p += qtc[v0p0].phi*gtc[v000].u[1] + qtc[v0m0].phi*conj(gtc[v0m0].u[1]);
	    p += qtc[vp00].phi*gtc[v000].u[0] + qtc[vm00].phi*conj(gtc[vm00].u[0]);
	    ptc[v000].phi = NDx2*qtc[v000].phi - p;
	  }	  	  

	}
	/*
	  ty = 0 and ty = Sy-1 edges
	 */
	for(int tx=1; tx<Sx-1; tx++) {
	  /*
	    ty = 0;
	  */
	  {
	    int v000 = tx + z0;
	    int vp00 = tx + zp;
	    int vm00 = tx + zm;
	    int v00p = tx+1 + z0;
	    int v00m = tx-1 + z0;
	    int v0p0 = tx + Sx + z0;
	    /* v0m0 is in another block, with ty = Sy - 1 in that block */
	    int v0m0 = tx + (Sy-1)*Sx + z0;
	    _Complex float p;
	    p  = qtc[v00p].phi*gtc[v000].u[2] + qtc[v00m].phi*conj(gtc[v00m].u[2]);
	    p += qtc[v0p0].phi*gtc[v000].u[1];
	    p += in[tcm0][v0m0].phi*conj(g[tcm0][v0m0].u[1]);
	    p += qtc[vp00].phi*gtc[v000].u[0] + qtc[vm00].phi*conj(gtc[vm00].u[0]);
	    ptc[v000].phi = NDx2*qtc[v000].phi - p;
	  }
	  /*
	    ty = Sy-1;
	  */
	  {
	    int y0 = (Sy-1)*Sx;
	    int v000 = tx + y0 + z0;
	    int vp00 = tx + y0 + zp;
	    int vm00 = tx + y0 + zm;
	    int v00p = tx+1 + y0 + z0;
	    int v00m = tx-1 + y0 + z0;
	    int v0m0 = tx + (Sy-2)*Sx + z0;
	    /* v0p0 is in another block, with ty = 0 in that block */
	    int v0p0 = tx + z0;
	    _Complex float p;
	    p  = qtc[v00p].phi*gtc[v000].u[2] + qtc[v00m].phi*conj(gtc[v00m].u[2]);
	    p += in[tcp0][v0p0].phi*gtc[v000].u[1];
	    p += qtc[v0m0].phi*conj(gtc[v0m0].u[1]);
	    p += qtc[vp00].phi*gtc[v000].u[0] + qtc[vm00].phi*conj(gtc[vm00].u[0]);
	    ptc[v000].phi = NDx2*qtc[v000].phi - p;
	  }
	}
	/*
	   Corner ty = 0, tx = 0
	*/
	{
	  int v000 = z0;
	  int vp00 = zp;
	  int vm00 = zm;
	  int v0p0 = z0 + Sx;
	  int v00p = z0 + 1;
	  /* v0m0 and v00m are on neighboring blocks, with ty = Sy-1
	     and tx = Sx-1 resp.
	  */
	  int v00m = z0 + Sx-1;
	  int v0m0 = z0 + (Sy-1)*Sx;
	  _Complex float p;
	  p  = qtc[v00p].phi*gtc[v000].u[2];
	  p += in[tc0m][v00m].phi*conj(g[tc0m][v00m].u[2]);
	  p += qtc[v0p0].phi*gtc[v000].u[1];
	  p += in[tcm0][v0m0].phi*conj(g[tcm0][v0m0].u[1]);
	  p += qtc[vp00].phi*gtc[v000].u[0] + qtc[vm00].phi*conj(gtc[vm00].u[0]);
	  ptc[v000].phi = NDx2*qtc[v000].phi - p;	  
	}
	/*
	   Corner ty = 0, tx = Sx - 1
	*/
	{
	  int v000 = z0 + Sx-1;
	  int vp00 = zp + Sx-1;
	  int vm00 = zm + Sx-1;
	  int v0p0 = z0 + Sx + Sx-1;
	  int v00m = z0 + Sx-2;
	  /* v0m0 and v00p are on neighboring blocks, with ty = Sy-1
	     and tx = 0 resp.
	  */
	  int v00p = z0;
	  int v0m0 = z0 + (Sy-1)*Sx + Sx-1;
	  _Complex float p;
	  p  = in[tc0p][v00p].phi*gtc[v000].u[2];
	  p += qtc[v00m].phi*conj(gtc[v00m].u[2]);
	  p += qtc[v0p0].phi*gtc[v000].u[1];
	  p += in[tcm0][v0m0].phi*conj(g[tcm0][v0m0].u[1]);
	  p += qtc[vp00].phi*gtc[v000].u[0] + qtc[vm00].phi*conj(gtc[vm00].u[0]);
	  ptc[v000].phi = NDx2*qtc[v000].phi - p;
	}
	
	/*
	   Corner ty = Sy-1, tx = 0
	*/
	{
	  int y0 = (Sy-1)*Sx;
	  int v000 = z0 + y0;
	  int vp00 = zp + y0;
	  int vm00 = zm + y0;
	  int v00p = z0 + y0 + 1;
	  int v0m0 = z0 + (Sy-2)*Sx;
	  /* v0p0 and v00m are on neighboring blocks, with ty = 0
	     and tx = Sx-1 resp.
	  */
	  int v0p0 = z0;
	  int v00m = z0 + y0 + Sx-1;
	  _Complex float p;
	  p  = qtc[v00p].phi*gtc[v000].u[2];
	  p += in[tc0m][v00m].phi*conj(g[tc0m][v00m].u[2]);
	  p += in[tcp0][v0p0].phi*gtc[v000].u[1];
	  p += qtc[v0m0].phi*conj(gtc[v0m0].u[1]);
	  p += qtc[vp00].phi*gtc[v000].u[0] + qtc[vm00].phi*conj(gtc[vm00].u[0]);
	  ptc[v000].phi = NDx2*qtc[v000].phi - p;
	}	
	
	/*
	   Corner ty = Sy-1, tx = Sx-1
	*/
	{
	  int y0 = (Sy-1)*Sx;
	  int v000 = z0 + y0 + Sx-1;
	  int vp00 = zp + y0 + Sx-1;
	  int vm00 = zm + y0 + Sx-1;
	  int v0m0 = z0 + (Sy-2)*Sx + Sx-1;
	  int v00m = z0 + y0 + Sx-2;
	  /* v0p0 and v00p are on neighboring blocks, with ty = 0
	     and tx = 0 resp.
	  */
	  int v0p0 = z0 + Sx-1;
	  int v00p = z0 + y0;
	  _Complex float p;
	  p  = in[tc0p][v00p].phi*gtc[v000].u[2];
	  p += qtc[v00m].phi*conj(gtc[v00m].u[2]);
	  p += in[tcp0][v0p0].phi*gtc[v000].u[1];
	  p += qtc[v0m0].phi*conj(gtc[v0m0].u[1]);
	  p += qtc[vp00].phi*gtc[v000].u[0] + qtc[vm00].phi*conj(gtc[vm00].u[0]);
	  ptc[v000].phi = NDx2*qtc[v000].phi - p;
	}	
      }
    }
  return;
}

/*
 * returns y = x^H x for vector x of length L*L*L
 */
double
xdotx(latparams s, field **x)
{
  size_t L = s.L;
  size_t Sx = s.Sx;
  size_t Sy = s.Sy;
  size_t lx = s.lx;
  size_t ly = s.ly;
  double y = 0; 
  for(int i=0; i<lx*ly; i++)
    for(int j=0; j<L*Sx*Sy; j++) {
      _Complex float phi = x[i][j].phi;
      y += creal(phi)*creal(phi) + cimag(phi)*cimag(phi);
    }
  return y;
}

/*
 * returns z = x^H y for vectors x, y of length L*L*L
 */
_Complex double
xdoty(latparams s, field **x, field **y)
{
  size_t L = s.L;
  size_t Sx = s.Sx;
  size_t Sy = s.Sy;
  size_t lx = s.lx;
  size_t ly = s.ly;
  _Complex double z = 0;
  for(int i=0; i<lx*ly; i++)
    for(int j=0; j<L*Sx*Sy; j++) {  
      z += conj(x[i][j].phi)*y[i][j].phi;
    }
  return z;
}

/*
 * returns y = x - y for vectors y, x, of length L*L*L
 */
void
xmy(latparams s, field **x, field **y)
{
  size_t L = s.L;
  size_t Sx = s.Sx;
  size_t Sy = s.Sy;
  size_t lx = s.lx;
  size_t ly = s.ly;
  for(int i=0; i<lx*ly; i++)
    for(int j=0; j<L*Sx*Sy; j++) {
      y[i][j].phi = x[i][j].phi - y[i][j].phi;
    }
  return;
}

/*
 * returns y = x for vectors y, x, of length L*L*L
 */
void
xeqy(latparams s, field **x, field **y)
{
  size_t L = s.L;
  size_t Sx = s.Sx;
  size_t Sy = s.Sy;
  size_t lx = s.lx;
  size_t ly = s.ly;
  for(int i=0; i<lx*ly; i++)
    for(int j=0; j<L*Sx*Sy; j++)
      x[i][j].phi = y[i][j].phi;
  
  return;
}

/*
 * returns y = a*x+y for vectors y, x, of length L*L*L and scalar a
 */
void
axpy(latparams s, _Complex float a, field **x, field **y)
{
  size_t L = s.L;
  size_t Sx = s.Sx;
  size_t Sy = s.Sy;
  size_t lx = s.lx;
  size_t ly = s.ly;
  for(int i=0; i<lx*ly; i++)
    for(int j=0; j<L*Sx*Sy; j++)
      y[i][j].phi = a*x[i][j].phi + y[i][j].phi;

  return;
}

/*
 * returns y = x+a*y for vectors y, x, of length L*L*L and scalar a
 */
void
xpay(latparams s, field **x, _Complex float a, field **y)
{
  size_t L = s.L;
  size_t Sx = s.Sx;
  size_t Sy = s.Sy;
  size_t lx = s.lx;
  size_t ly = s.ly;
  for(int i=0; i<lx*ly; i++)
    for(int j=0; j<L*Sx*Sy; j++)
      y[i][j].phi = x[i][j].phi + a*y[i][j].phi;
  
  return;
}

/*
 * Solves lapl(u) x = b, for x, given b, using Conjugate Gradient
 */
void
cg(latparams lp, field **x, field **b, link **g)
{
  size_t L = lp.L;
  int max_iter = 100;
  float tol = 1e-9;

  /* Temporary fields needed for CG */
  field **r = new_field(lp);
  field **p = new_field(lp);
  field **Ap = new_field(lp);

  /* Initial residual and p-vector */
  lapl(lp, r, x, g);
  xmy(lp, b, r);
  xeqy(lp, p, r);

  /* Initial r-norm and b-norm */
  float rr = xdotx(lp, r);  
  float bb = xdotx(lp, b);
  double t_lapl = 0;
  int iter = 0;
  for(iter=0; iter<max_iter; iter++) {
    printf(" %6d, res = %+e\n", iter, rr/bb);
    if(sqrt(rr/bb) < tol)
      break;
    double t = stop_watch(0);
    lapl(lp, Ap, p, g);
    t_lapl += stop_watch(t);
    float pAp = xdoty(lp, p, Ap);
    float alpha = rr/pAp;
    axpy(lp, alpha, p, x);
    axpy(lp, -alpha, Ap, r);
    float r1r1 = xdotx(lp, r);
    float beta = r1r1/rr;
    xpay(lp, r, beta, p);
    rr = r1r1;
  }

  /* Recompute residual after convergence */
  lapl(lp, r, x, g);
  xmy(lp, b, r);
  rr = xdotx(lp, r);

  double beta_fp = 50*((double)L*L*L)/(t_lapl/(double)iter)*1e-9;
  double beta_io = 40*((double)L*L*L)/(t_lapl/(double)iter)*1e-9;
  printf(" Converged after %6d iterations, res = %+e\n", iter, rr/bb);  
  printf(" Time in lapl(): %+6.3e sec/call, %4.2e Gflop/s, %4.2e GB/s\n",
	 t_lapl/(double)iter, beta_fp, beta_io);  

  del_field(r);
  del_field(p);
  del_field(Ap);
  return;
}

/*
 * Usage info
 */
void
usage(char *argv[])
{
  fprintf(stderr, " Usage: %s L Sy Sx\n", argv[0]);
  return;
}

int
main(int argc, char *argv[])
{
  if(argc != 4) {
    usage(argv);
    exit(1);
  }

  char *e;
  size_t L = (int)strtoul(argv[1], &e, 10);
  if(*e != '\0') {
    usage(argv);
    exit(2);
  }
  
  size_t Sy = (int)strtoul(argv[2], &e, 10);
  if(*e != '\0') {
    usage(argv);
    exit(2);
  }
  
  size_t Sx = (int)strtoul(argv[3], &e, 10);
  if(*e != '\0') {
    usage(argv);
    exit(2);
  }

  latparams lp = init_latparams(L, Sy, Sx);
  field **b = new_field(lp);
  field **x = new_field(lp);
  link **g = new_links(lp);

  rand_links(lp, g);
  rand_field(lp, b);
  zero_field(lp, x);

  cg(lp, x, b, g);
  
  del_field(b);
  del_field(x);
  del_links(g);
  return 0;
}
