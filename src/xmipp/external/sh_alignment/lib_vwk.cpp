/*********************************************************************
 *                           L I B _ V W K                            *
 **********************************************************************
 * Library is part of the Situs package URL: situs.biomachina.org     *
 * (c) Willy Wriggers, Pablo Chacon, and Paul Boyle 2001-2011         *
 **********************************************************************
 *                                                                    *
 * Map and kernel creation / manipulation tools.                      *
 *                                                                    *
 **********************************************************************
 * See legal statement for terms of distribution                      *
 *********************************************************************/

#include "situs.h"
#include "lib_vwk.h"
#include "lib_vec.h"
#include "lib_err.h"
#include "lib_std.h"

#define BARL 70  /* available space for histogram bars */


/*====================================================================*/
/* generic index function z>y>x for cubic maps */
unsigned long gidz_cube(int k, int j, int i, unsigned ext)
{
  return ext * ext * k + ext * j + i;
}

/*====================================================================*/
/* generic index function z>y>x */
unsigned long gidz_general(int k, int j, int i, unsigned ey, unsigned ex)
{
  return ex * ey * k + ex * j + i;
}

/*====================================================================*/
/* adds specified margin of zero density voxels to boundary */
/* outmap is allocated and new output map parameters are returned */
void create_padded_map(double **outmap, unsigned *out_extx, unsigned *out_exty, unsigned *out_extz,
                       double *out_origx, double *out_origy, double *out_origz, unsigned long *out_nvox,
                       double *inmap, unsigned in_extx, unsigned in_exty, unsigned in_extz,
                       double in_origx, double in_origy, double in_origz,
                       double widthx, double widthy, double widthz,
                       unsigned margin[3])
{
  unsigned indz, indx, indy;
  unsigned long index_old, index_new;

  *out_nvox = (in_extx + margin[0] * 2) * (in_exty + margin[1] * 2) * (in_extz + margin[2] * 2);
  do_vect(outmap, *out_nvox);

  for (indz = 0; indz < in_extz; indz++)
    for (indy = 0; indy < in_exty; indy++)
      for (indx = 0; indx < in_extx; indx++) {
        index_old = in_extx * in_exty * indz + in_extx * indy + indx;
        index_new = ((in_extx + 2 * margin[0]) * (in_exty + 2 * margin[1]) *
                     (indz + margin[2]) + (in_extx + 2 * margin[0]) *
                     (indy + margin[1]) + (indx + margin[0]));
        *(*outmap + index_new) = *(inmap + index_old);
      }

  *out_extx = in_extx + margin[0] * 2; /* may destroy in_extx if out_extx points to it */
  *out_exty = in_exty + margin[1] * 2; /* may destroy in_exty if out_exty points to it */
  *out_extz = in_extz + margin[2] * 2; /* may destroy in_extz if out_extz points to it */
  *out_origx = in_origx - margin[0] * widthx; /* may destroy in_origx if out_origx points to it */
  *out_origy = in_origy - margin[1] * widthy; /* may destroy in_origy if out_origy points to it */
  *out_origz = in_origz - margin[2] * widthz; /* may destroy in_origz if out_origz points to it */

  /* input variables are no longer used since they may have been overwritten */
  printf("lib_vwk> Map size expanded from %d x %d x %d to %d x %d x %d by zero-padding.\n",
         *out_extx - margin[0] * 2, *out_exty - margin[1] * 2, *out_extz - margin[2] * 2,
         *out_extx, *out_exty, *out_extz);
  printf("lib_vwk> New map origin (coord of first voxel): (%.3f,%.3f,%.3f)\n", *out_origx, *out_origy, *out_origz);
}



/*====================================================================*/
/* change voxel spacings and bring map origin into register with coordinate system origin */
/* outmap is allocated and new output map parameters are returned */
void interpolate_map(double **outmap, unsigned *out_extx, unsigned *out_exty, unsigned *out_extz,
                     double *out_origx, double *out_origy, double *out_origz,
                     double out_widthx, double out_widthy, double out_widthz,
                     double *inmap, unsigned in_extx, unsigned in_exty, unsigned in_extz,
                     double in_origx, double in_origy, double in_origz,
                     double in_widthx, double in_widthy, double in_widthz)
{

  int isx_out, isy_out, isz_out;
  int iex_out, iey_out, iez_out;
  double deltax, deltay, deltaz;
  unsigned long out_nvox;
  unsigned indx, indy, indz;
  unsigned sx, sy, sz;
  double xpos, ypos, zpos, gx, gy, gz, a, b, c;
  int x0, y0, z0, x1, y1, z1;

  /* output start index rel. to coordinate system origin, asserting that outmap is fully embedded in inmap */
  isx_out = (int)ceil(in_origx / out_widthx);
  isy_out = (int)ceil(in_origy / out_widthy);
  isz_out = (int)ceil(in_origz / out_widthz);

  /* output end index rel. to coordinate system origin, asserting that outmap is fully embedded in inmap */
  iex_out = (int)floor((in_origx + in_widthx * (in_extx - 1)) / out_widthx);
  iey_out = (int)floor((in_origy + in_widthy * (in_exty - 1)) / out_widthy);
  iez_out = (int)floor((in_origz + in_widthz * (in_extz - 1)) / out_widthz);

  /* assign output grid size */
  sx = in_extx;
  sy = in_exty;
  sz = in_extz; /* save input size parameters in case they get over written */
  *out_extx = iex_out - isx_out + 1; /* may destroy in_extx if out_extx points to it */
  *out_exty = iey_out - isy_out + 1; /* may destroy in_exty if out_exty points to it */
  *out_extz = iez_out - isz_out + 1; /* may destroy in_extz if out_extz points to it */
  if (*out_extx < 2 || *out_exty < 2 || *out_extz < 2) {
    error_underflow(17005, "lib_vwk");
  }

  /* save origin shift */
  deltax = isx_out * out_widthx - in_origx;
  deltay = isy_out * out_widthy - in_origy;
  deltaz = isz_out * out_widthz - in_origz;

  /* now allocate outmap */
  out_nvox = (*out_extx) * (*out_exty) * (*out_extz);
  do_vect(outmap, out_nvox);

  /* loop through outmap */
  for (indz = 0; indz < *out_extz; indz++)
    for (indy = 0; indy < *out_exty; indy++)
      for (indx = 0; indx < *out_extx; indx++) {

        /* determine position of outmap voxel relative to start of inmap */
        xpos = deltax + indx * out_widthx;
        ypos = deltay + indy * out_widthy;
        zpos = deltaz + indz * out_widthz;

        /* compute position in inmap voxel units */
        gx = (xpos / in_widthx);
        gy = (ypos / in_widthy);
        gz = (zpos / in_widthz);

        /* compute bounding box voxel indices and linear distances */
        x0 = (int) floor(gx);
        y0 = (int) floor(gy);
        z0 = (int) floor(gz);
        x1 = (int) ceil(gx);
        y1 = (int) ceil(gy);
        z1 = (int) ceil(gz);
        a = gx - x0;
        b = gy - y0;
        c = gz - z0;

        /* interpolate */
        *(*outmap + gidz_general(indz, indy, indx, *out_exty, *out_extx)) =
          a * b * c * *(inmap + gidz_general(z1, y1, x1, sy, sx)) +
          (1 - a) * b * c * *(inmap + gidz_general(z1, y1, x0, sy, sx)) +
          a * (1 - b) * c * *(inmap + gidz_general(z1, y0, x1, sy, sx)) +
          a * b * (1 - c) * *(inmap + gidz_general(z0, y1, x1, sy, sx)) +
          a * (1 - b) * (1 - c) * *(inmap + gidz_general(z0, y0, x1, sy, sx)) +
          (1 - a) * b * (1 - c) * *(inmap + gidz_general(z0, y1, x0, sy, sx)) +
          (1 - a) * (1 - b) * c * *(inmap + gidz_general(z1, y0, x0, sy, sx)) +
          (1 - a) * (1 - b) * (1 - c) * *(inmap + gidz_general(z0, y0, x0, sy, sx));
      }

  *out_origx = in_origx + deltax; /* may destroy in_origx if out_origx points to it */
  *out_origy = in_origy + deltay; /* may destroy in_origy if out_origy points to it */
  *out_origz = in_origz + deltaz; /* may destroy in_origz if out_origz points to it */

  /* don't use in_origx, in_origy, in_origz below this line since they may have been overwritten */
  if (sx != *out_extx || sy != *out_exty || sz != *out_extz)
    printf("lib_vwk> Map interpolated from %d x %d x %d to %d x %d x %d.\n",
           sx, sy, sz, *out_extx, *out_exty, *out_extz);
  if (in_widthx != out_widthx || in_widthy != out_widthy || in_widthz != out_widthz)
    printf("lib_vwk> Voxel spacings interpolated from (%.3f,%.3f,%.3f) to (%.3f,%.3f,%.3f).\n",
           in_widthx, in_widthy, in_widthz, out_widthx, out_widthy, out_widthz);
  if (deltax != 0.0 || deltay != 0.0 || deltaz != 0.0)
    printf("lib_vwk> New map origin (coord of first voxel), in register with coordinate system origin: (%.3f,%.3f,%.3f)\n", *out_origx, *out_origy, *out_origz);
}


/*====================================================================*/
/* project inmap to reference lattice, allocating outmap */
void project_map_lattice(double **outmap, unsigned ref_extx, unsigned ref_exty, unsigned ref_extz,
                         double ref_origx, double ref_origy, double ref_origz,
                         double ref_widthx, double ref_widthy, double ref_widthz,
                         double *inmap, unsigned in_extx, unsigned in_exty, unsigned in_extz,
                         double in_origx, double in_origy, double in_origz,
                         double in_widthx, double in_widthy, double in_widthz)
{

  double deltax, deltay, deltaz;
  unsigned long out_nvox;
  unsigned indx, indy, indz;
  double xpos, ypos, zpos, gx, gy, gz, a, b, c;
  int x0, y0, z0, x1, y1, z1;

  if (ref_extx < 2 || ref_exty < 2 || ref_extz < 2) {
    error_underflow(17010, "lib_vwk");
  }
  /* save origin shift */
  deltax = ref_origx - in_origx;
  deltay = ref_origy - in_origy;
  deltaz = ref_origz - in_origz;

  /* now allocate outmap */
  out_nvox = (ref_extx) * (ref_exty) * (ref_extz);
  do_vect(outmap, out_nvox);

  /* loop through outmap */
  for (indz = 0; indz < ref_extz; indz++)
    for (indy = 0; indy < ref_exty; indy++)
      for (indx = 0; indx < ref_extx; indx++) {

        /* determine position of outmap voxel relative to start of inmap */
        xpos = deltax + indx * ref_widthx;
        ypos = deltay + indy * ref_widthy;
        zpos = deltaz + indz * ref_widthz;

        /* compute position in inmap index units */
        gx = (xpos / in_widthx);
        gy = (ypos / in_widthy);
        gz = (zpos / in_widthz);

        /* compute probe box voxel indices and linear distances */
        x0 = (int) floor(gx);
        y0 = (int) floor(gy);
        z0 = (int) floor(gz);
        x1 = (int) ceil(gx);
        y1 = (int) ceil(gy);
        z1 = (int) ceil(gz);

        /* if probe box inside inmap interpolate */
        if (x0 >= 0 && x1 < (int)in_extx && y0 >= 0 && y1 < (int)in_exty && z0 >= 0 && z1 < (int)in_extz) {
          a = gx - x0;
          b = gy - y0;
          c = gz - z0;
          *(*outmap + gidz_general(indz, indy, indx, ref_exty, ref_extx)) =
            a * b * c * *(inmap + gidz_general(z1, y1, x1, in_exty, in_extx)) +
            (1 - a) * b * c * *(inmap + gidz_general(z1, y1, x0, in_exty, in_extx)) +
            a * (1 - b) * c * *(inmap + gidz_general(z1, y0, x1, in_exty, in_extx)) +
            a * b * (1 - c) * *(inmap + gidz_general(z0, y1, x1, in_exty, in_extx)) +
            a * (1 - b) * (1 - c) * *(inmap + gidz_general(z0, y0, x1, in_exty, in_extx)) +
            (1 - a) * b * (1 - c) * *(inmap + gidz_general(z0, y1, x0, in_exty, in_extx)) +
            (1 - a) * (1 - b) * c * *(inmap + gidz_general(z1, y0, x0, in_exty, in_extx)) +
            (1 - a) * (1 - b) * (1 - c) * *(inmap + gidz_general(z0, y0, x0, in_exty, in_extx));
        }
      }

  /* print some info */
  if (in_extx != ref_extx || in_exty != ref_exty || in_extz != ref_extz)
    printf("lib_vwk> %d x %d x %d map projected to %d x %d x %d lattice.\n",
           in_extx, in_exty, in_extz, ref_extx, ref_exty, ref_extz);
  if (in_widthx != ref_widthx || in_widthy != ref_widthy || in_widthz != ref_widthz)
    printf("lib_vwk> Voxel spacings interpolated from (%.3f,%.3f,%.3f) to (%.3f,%.3f,%.3f).\n",
           in_widthx, in_widthy, in_widthz, ref_widthx, ref_widthy, ref_widthz);
  if (deltax != 0.0 || deltay != 0.0 || deltaz != 0.0)
    printf("lib_vwk> New map origin (coord of first voxel): (%.3f,%.3f,%.3f)\n", ref_origx, ref_origy, ref_origz);
}


/*====================================================================*/
/* shrink inmap and replace with smaller *outmap that has odd intervals */
/* outmap is allocated and new output map parameters are returned */
void shrink_margin(double **outmap, unsigned *out_extx, unsigned *out_exty, unsigned *out_extz,
                   double *out_origx, double *out_origy, double *out_origz, unsigned long *out_nvox,
                   double *inmap, unsigned in_extx, unsigned in_exty, unsigned in_extz,
                   double in_origx, double in_origy, double in_origz,
                   double widthx, double widthy, double widthz)
{
  unsigned m, p, q, sx, sy, sz;
  unsigned minx, miny, minz, maxx, maxy, maxz;
  int margin[6];
  unsigned indz, indx, indy;
  unsigned long index_old, index_new;
  const char *program = "lib_vwk";

  maxx = 0;
  maxy = 0;
  maxz = 0;
  minx = in_extx - 1;
  miny = in_exty - 1;
  minz = in_extz - 1;

  for (q = 0; q < in_extz; q++)
    for (m = 0; m < in_exty; m++)
      for (p = 0; p < in_extx; p++)
        if (*(inmap + gidz_general(q, m, p, in_exty, in_extx)) > 0) {
          if (p <= minx) minx = p;
          if (p >= maxx) maxx = p;
          if (m <= miny) miny = m;
          if (m >= maxy) maxy = m;
          if (q <= minz) minz = q;
          if (q >= maxz) maxz = q;
        }

  if (maxx < minx) {
    error_density(17020, program);
  }

  margin[1] = in_extx - maxx - 1;
  margin[3] = in_exty - maxy - 1;
  margin[5] = in_extz - maxz - 1;
  margin[0] = minx;
  margin[2] = miny;
  margin[4] = minz;

  /* compute new grid size */
  p = in_extx - (margin[0] + margin[1]);
  m = in_exty - (margin[2] + margin[3]);
  q = in_extz - (margin[4] + margin[5]);

  /* number of map intervals to be odd, if necessary they are increase */
  if (2 * (p / 2) == p) {
    p++;
    if (margin[0] > 0)margin[0]--;
  }
  if (2 * (m / 2) == m) {
    m++;
    if (margin[2] > 0)margin[2]--;
  }
  if (2 * (q / 2) == q) {
    q++;
    if (margin[4] > 0)margin[4]--;
  }


  *out_nvox = p * m * q;
  do_vect(outmap, *out_nvox);

  for (indz = margin[4]; indz < in_extz - margin[5]; indz++)
    for (indy = margin[2]; indy < in_exty - margin[3]; indy++)
      for (indx = margin[0]; indx < in_extx - margin[1]; indx++) {
        index_new = p * m * (indz - margin[4]) + p * (indy - margin[2]) + (indx - margin[0]);
        index_old = in_extx * in_exty * indz + in_extx * indy + indx;
        *(*outmap + index_new) = *(inmap + index_old);
      }
  sx = in_extx;
  sy = in_exty;
  sz = in_extz;
  *out_extx = p;/* may destroy in_extx if out_extx points to it */
  *out_exty = m;/* may destroy in_exty if out_exty points to it */
  *out_extz = q;/* may destroy in_extz if out_extz points to it */
  *out_origx = in_origx + margin[0] * widthx; /* may destroy in_origx if out_origx points to it */
  *out_origy = in_origy + margin[2] * widthy; /* may destroy in_origy if out_origy points to it */
  *out_origz = in_origz + margin[4] * widthz; /* may destroy in_origz if out_origz points to it */

  /* input variables are no longer used since they may have been overwritten */
  printf("lib_vwk> Map size changed from %d x %d x %d to %d x %d x %d.\n", sx, sy, sz, *out_extx, *out_exty, *out_extz);
  printf("lib_vwk> New map origin (coord of first voxel): (%.3f,%.3f,%.3f)\n", *out_origx, *out_origy, *out_origz);
}


/*====================================================================*/
/* returns total density in map */
double calc_total(double *phi, unsigned long nvox)
{
  unsigned long m;
  double total_density;
  total_density = 0.0;
  for (m = 0; m < nvox; m++) total_density += (*(phi + m));
  return total_density;
}

/*====================================================================*/
/* returns average density */
double calc_average(double *phi, unsigned long nvox)
{
  double total_density;
  unsigned long m;
  total_density = 0.0;
  for (m = 0; m < nvox; m++) {
    total_density += (*(phi + m));
  }
  return total_density / ((double)nvox);
}

/*====================================================================*/
/* returns sigma */
double calc_sigma(double *phi, unsigned long nvox)
{
  unsigned long m;
  double ave, varsum;

  ave = calc_average(phi, nvox);
  varsum = 0.0;
  for (m = 0; m < nvox; m++) {
    varsum += (*(phi + m) - ave) * (*(phi + m) - ave);
  }
  return sqrt((varsum) / ((double)nvox));
}

/*====================================================================*/
/* returns norm */
double calc_norm(double *phi, unsigned long nvox)
{
  unsigned long m;
  double varsum;

  varsum = 0.0;
  for (m = 0; m < nvox; m++) {
    varsum += (*(phi + m)) * (*(phi + m));
  }
  return sqrt((varsum) / ((double)nvox));
}

/*====================================================================*/
/* returns above zero average density */
double calc_gz_average(double *phi, unsigned long nvox)
{
  double total_density = 0.0;
  unsigned long m = 0, t = 0;

  for (m = 0; m < nvox; m++) {
    if (*(phi + m) > 0) {
      total_density += (*(phi + m));
      ++t;
    }
  }
  if (t > 0) return total_density / ((double)t);
  else return 0.0;
}

/*====================================================================*/
/* returns above zero sigma */
double calc_gz_sigma(double *phi, unsigned long nvox)
{
  unsigned long m = 0, t = 0;
  double ave = 0.0, varsum = 0.0;

  ave = calc_gz_average(phi, nvox);

  for (m = 0; m < nvox; m++) {
    if (*(phi + m) > 0) {
      varsum += (*(phi + m) - ave) * (*(phi + m) - ave);
      ++t;
    }
  }
  if (t > 0) return sqrt((varsum) / ((double)t));
  else return 0.0;
}

/*====================================================================*/
/* returns above zero density norm */
double calc_gz_norm(double *phi, unsigned long nvox)
{
  unsigned long m = 0, t = 0;
  double varsum = 0.0;

  for (m = 0; m < nvox; m++) {
    if (*(phi + m) > 0) {
      varsum += (*(phi + m)) * (*(phi + m));
      ++t;
    }
  }
  if (t > 0) return sqrt((varsum) / ((double)t));
  else return 0.0;
}

/*====================================================================*/
/* returns maximum density */
double calc_max(double *phi, unsigned long nvox)
{
  unsigned long m;
  double maxdens;

  maxdens = -1e20;
  for (m = 0; m < nvox; m++) {
    if (*(phi + m) > maxdens) maxdens = *(phi + m);
  }
  return maxdens;
}

/*====================================================================*/
/* returns minimum density */
double calc_min(double *phi, unsigned long nvox)
{
  unsigned long m;
  double mindens;

  mindens = 1e20;
  for (m = 0; m < nvox; m++) {
    if (*(phi + m) < mindens) mindens = *(phi + m);
  }
  return mindens;
}

/*====================================================================*/
/* efficiently computes all four map density distribution parameters */
void calc_map_info(double *phi, unsigned long nvox, double *maxdens, double *mindens, double *ave, double *sig)
{
  unsigned long m;

  *maxdens = -1e20;
  *mindens = 1e20;
  *ave = 0.0;
  for (m = 0; m < nvox; m++) {
    if (*(phi + m) > *maxdens) *maxdens = *(phi + m);
    if (*(phi + m) < *mindens) *mindens = *(phi + m);
    *ave += *(phi + m);
  }
  *ave /= (double)nvox;
  *sig = 0.0;
  for (m = 0; m < nvox; m++) if (*(phi + m) > 0) *sig += (*(phi + m) - *ave) * (*(phi + m) - *ave);
  *sig /= (double)nvox;
  *sig = sqrt(*sig);
}


/*====================================================================*/
/* outputs info about map density distribution */
void print_map_info(double *phi, unsigned long nvox)
{
  double maxdens, mindens, sig, ave;

  calc_map_info(phi, nvox, &maxdens, &mindens, &ave, &sig);
  printf("lib_vwk> Map density info: max %f, min %f, ave %f, sig %f.\n", maxdens, mindens, ave, sig);
}


/*====================================================================*/
/* set densities below limit to zero */
void threshold(double *phi, unsigned long nvox, double limit)
{
  unsigned long m, v;

  v = nvox;
  for (m = 0; m < nvox; m++) if (*(phi + m) < limit) {
      *(phi + m) = 0.0;
      --v;
    }
  printf("lib_vwk> Setting density values below %f to zero.\n", limit);
  printf("lib_vwk> Remaining occupied volume: %lu voxels.\n", v);
}


/*====================================================================*/
/* set densities below limit to zero and equal or above to one */
void step_threshold(double *phi, unsigned long nvox, double limit)
{
  unsigned long m, v;

  v = nvox;
  for (m = 0; m < nvox; m++) {
    if (*(phi + m) < limit) {
      *(phi + m) = 0.0;
      --v;
    } else *(phi + m) = 1.0;
  }
  printf("lib_vwk> Setting density values below %f to zero and equal or above to one.\n", limit);
  printf("lib_vwk> Remaining occupied volume: %lu voxels.\n", v);
}

/*====================================================================*/
/* boost densities above limit by factor scale */
void boost_factor_high(double *phi, unsigned long nvox, double limit, double scale)
{
  unsigned long m;

  for (m = 0; m < nvox; m++) if (*(phi + m) > limit) *(phi + m) *= scale;
}

/*====================================================================*/
/* boost densities above limit by exponent lexpo, keeping densities continuosly differentiable at limit */
void boost_power_high(double *phi, unsigned long nvox, double limit, double lexpo)
{
  unsigned long m;
  double lnorm, lconst;

  if (limit == 0) for (m = 0; m < nvox; m++) if (*(phi + m) > limit) *(phi + m) = pow(*(phi + m), lexpo);
      else {
        lnorm = 1.0 / (lexpo * pow(limit, (lexpo - 1.0)));
        lconst = (1.0 - 1.0 / lexpo) * limit;
        for (m = 0; m < nvox; m++) if (*(phi + m) > limit) *(phi + m) = lconst + pow(*(phi + m), lexpo) * lnorm;
      }
}

/*====================================================================*/
/* divide density values by factor */
void normalize(double *phi, unsigned long nvox, double factor)
{
  unsigned i;
  const char *program = "lib_vwk";

  if (factor == 0.0) {
    error_normalize(17130, program);
  }
  for (i = 0; i < nvox; i++) *(phi + i) /= factor;
}

/*====================================================================*/
/* subtract value from map density */
void floatshift(double *phi, unsigned long nvox, double dens)
{
  unsigned i;

  for (i = 0; i < nvox; i++) *(phi + i) -= dens;
}

/*====================================================================*/
/* clip map density, returning 1 of clipping occurs, zero if not */
int clipped(double *phi, unsigned long nvox, double max, double min)
{
  unsigned i;
  int clipping = 0;

  for (i = 0; i < nvox; i++) {
    if (*(phi + i) < min) {
      *(phi + i) = min;
      clipping = 1;
    } else if (*(phi + i) > max) {
      *(phi + i) = max;
      clipping = 1;
    }
  }
  return clipping;
}



/*====================================================================*/
/* allocates and generates truncated Gaussian 3D kernel with */
/* sigma1D = sigmap, within sigma_factor*sigmap*/
void create_gaussian(double **phi, unsigned long *nvox, unsigned *ext,
                     double sigmap, double sigma_factor)
{
  int exth, indx, indy, indz;
  double dsqu;
  double mscale;
  unsigned long count;
  double bvalue, cvalue;

  /* truncate at sigma_factor * sigma1D */
  exth = (int) ceil(sigma_factor * sigmap);
  *ext = 2 * exth - 1;
  *nvox = *ext * *ext * *ext;

  printf("lib_vwk> Generating Gaussian kernel with %d^3 = %lu voxels.\n", *ext, *nvox);
  do_vect(phi, *nvox);

  /* write Gaussian within sigma_factor * sigma-1D to map */
  bvalue = -1.0 / (2.0 * sigmap * sigmap);
  cvalue = sigma_factor * sigma_factor * sigmap * sigmap;

  mscale = 0.0;
  for (indz = 0; indz < (int)*ext; indz++)
    for (indy = 0; indy < (int)*ext; indy++)
      for (indx = 0; indx < (int)*ext; indx++) {
        dsqu = (indx - exth + 1) * (indx - exth + 1) +
               (indy - exth + 1) * (indy - exth + 1) +
               (indz - exth + 1) * (indz - exth + 1);
        if (dsqu <= cvalue)
          *(*phi + gidz_cube(indz, indy, indx, *ext)) = exp(dsqu * bvalue);
        mscale += *(*phi + gidz_cube(indz, indy, indx, *ext));
      }
  for (count = 0; count < *nvox; count++) *(*phi + count) /= mscale;
}

/*====================================================================*/
/* shrink input kernel inmap and replace with smaller *outmap within sigma_factor*sigmap*/
/* that has odd intervals; outmap is allocated and new output map parameters are returned */
void shrink_to_sigma_factor(double **outmap, unsigned *out_ext, double *inmap,
                            unsigned in_ext, double sigmap, double sigma_factor)
{
  int exth, indx, indy, indz;
  unsigned long nvox, index_old, index_new;
  unsigned margin;
  double cvalue, dsqu;
  const char *program = "lib_vwk";

  /* truncate at sigma_factor * sigma1D */
  exth = (int) ceil(sigma_factor * sigmap);
  *out_ext = 2 * exth - 1;
  if (*out_ext > in_ext || 2 * (in_ext / 2) == in_ext) {
    error_kernels(17030, program);
  }
  nvox = *out_ext * *out_ext * *out_ext;
  cvalue = sigma_factor * sigma_factor * sigmap * sigmap;

  printf("lib_vwk> Generating kernel with %d^3 = %lu voxels.\n", *out_ext, nvox);
  do_vect(outmap, nvox);

  margin = (in_ext - *out_ext) / 2;
  for (indz = margin; indz < (int)(in_ext - margin); indz++)
    for (indy = margin; indy < (int)(in_ext - margin); indy++)
      for (indx = margin; indx < (int)(in_ext - margin); indx++) {
        index_new = *out_ext * *out_ext * (indz - margin) + *out_ext * (indy - margin) + (indx - margin);
        index_old = in_ext * in_ext * indz + in_ext * indy + indx;
        *(*outmap + index_new) = *(inmap + index_old);
      }

  /* make the kernel spherical */
  for (indz = 0; indz < (int)*out_ext; indz++)
    for (indy = 0; indy < (int)*out_ext; indy++)
      for (indx = 0; indx < (int)*out_ext; indx++) {
        index_new = (*out_ext) * (*out_ext) * indz + (*out_ext) * indy + indx;
        dsqu = (indx - exth + 1) * (indx - exth + 1) +
               (indy - exth + 1) * (indy - exth + 1) +
               (indz - exth + 1) * (indz - exth + 1);
        if (dsqu > cvalue) *(*outmap + index_new) = 0.0;
      }
}

/*====================================================================*/
/* creates identity 1x1x1 kernel */
void create_identity(double **phi, unsigned long *nvox, unsigned *ext)
{
  *ext = 1;
  *nvox = 1;
  do_vect(phi, *nvox);
  *(*phi + 0) = 1.0;
}

/*====================================================================*/
/* creates Laplacian 3x3x3 kernel */
void create_laplacian(double **phi, unsigned long *nvox, unsigned *ext)
{
  int indx, indy, indz;
  double lap_discrete[3][3][3] = {
    { {0.0, 0.0, 0.0}, {0.0, 1. / 12.0, 0.0}, {0.0, 0.0, 0.0} },
    { {0.0, 1.0 / 12.0, 0.0}, {1.0 / 12.0, -6.0 / 12.0, 1.0 / 12.0}, {0.0, 1.0 / 12.0, 0.0} },
    { {0.0, 0.0, 0.0}, {0.0, 1.0 / 12.0, 0.0}, {0.0, 0.0, 0.0} }
  };

  *ext = 3;
  *nvox = *ext * *ext * *ext;

  printf("lib_vwk> Generating Laplacian kernel with %d^3 = %lu voxels.\n", *ext, *nvox);
  do_vect(phi, *nvox);

  for (indz = 0; indz < (int)*ext; indz++)
    for (indy = 0; indy < (int)*ext; indy++)
      for (indx = 0; indx < (int)*ext; indx++)
        *(*phi + gidz_cube(indz, indy, indx, *ext)) = lap_discrete[indx][indy][indz];
}

/*====================================================================*/
/* relaxes shell of width radius, just outside thresholded area, by the Poisson equation */
/* assumes radius is smaller than previously applied zero padding */
/* assumes that phi has been thresholded and contains some non-zero densities */
void relax_laplacian(double **phi, unsigned extx, unsigned exty, 
                     unsigned extz, unsigned ignored[3], double radius)
{
  double average[27] = {0.0, 0.0, 0.0, 0.0, 1.0 / 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 6.0, 0.0, 1.0 / 6.0, 0.0, 1.0 / 6.0, 0.0, 1.0 / 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 6.0, 0.0, 0.0, 0.0, 0.0};
  double *nextphi, diff, norm;
  unsigned long nvox, indv, indw, threscount, maskcount;
  char *mask;
  unsigned indx, indy, indz;
  int indx2, indy2, indz2;
  int margx, margy, margz, margin;
  const char *program = "lib_vwk";

  margx = (int)(ignored[0] + radius);
  margy = (int)(ignored[1] + radius);
  margz = (int)(ignored[2] + radius);
  margin = (int) ceil(radius);
  printf("lib_vwk> Relaxing %d voxel thick shell about thresholded density... \n", margin);

  /* allocate phi mask */
  nvox = extx * exty * extz;
  mask = (char *) alloc_vect(nvox, sizeof(char));
  for (indv = 0; indv < nvox; indv++)
    *(mask + indv) = 1;

  /* assign phi mask value based on distance to thresholded map */
  for (indz = margz; indz < extz - margz; indz++)
    for (indy = margy; indy < exty - margy; indy++)
      for (indx = margx; indx < extx - margx; indx++) {
        indv = gidz_general(indz, indy, indx, exty, extx);
        if (*(*phi + indv) != 0)
          for (indz2 = -margin; indz2 <= margin; indz2++)
            for (indy2 = -margin; indy2 <= margin; indy2++)
              for (indx2 = -margin; indx2 <= margin; indx2++) {
                indw = gidz_general(indz + indz2, indy + indy2, indx + indx2, exty, extx);
                if (*(*phi + indw) == 0.0 && indz2 * indz2 + indy2 * indy2 + indx2 * indx2 < radius * radius) *(mask + indw) = 0;
              }
      }
  /* compute norm */
  maskcount = 0;
  threscount = 0;
  norm = 0.0;
  for (indv = 0; indv < nvox; indv++) {
    if (*(*phi + indv) != 0.0) {
      ++threscount;
      norm += *(*phi + indv);
    } else if (*(mask + indv) == 0) ++maskcount;
  }
  if (0 == threscount) {
    printf("threscount was 0 (zero)\n");
    exit(-1);
  }
  norm /= (double)threscount; /* average density for thresholded volume, assuming threscount>0 */
  norm *= maskcount;/* density total one would get if mask=0 was filled with average */

  /* iterate on original lattice, no focusing */
  do_vect(&nextphi, nvox);
  do {
    convolve_kernel_inside_fast(&nextphi, *phi, extx, exty, extz, average, 3, 1.0, ignored);
    diff = 0.0;
    for (indz = ignored[2]; indz < extz - ignored[2]; indz++)
      for (indy = ignored[1]; indy < exty - ignored[1]; indy++)
        for (indx = ignored[0]; indx < extx - ignored[0]; indx++) {
          indv = gidz_general(indz, indy, indx, exty, extx);
          if (*(mask + indv) == 0) {
            diff += fabs(*(nextphi + indv) - * (*phi + indv));
            *(*phi + indv) = *(nextphi + indv);
          }
        }
  } while (diff > 1E-8 * norm);
  free_vect_and_zero_ptr((void**)&nextphi);
  free_vect_and_zero_ptr((void**)&mask);
}

/*====================================================================*/
/* convolves inmap with kernel inside, and writes to same-size outmap*/
/* sufficient memory for *outmap must already be allocated */
/* allows *outmap and inmap to point to same array - not speed optimized */
void convolve_kernel_inside(double **outmap, double *inmap,
                            unsigned in_extx, unsigned in_exty, unsigned in_extz,
                            double *kernel, unsigned kernel_size)
{
  double dval;
  unsigned long out_nvox;
  unsigned indx, indy, indz;
  int indx2, indy2, indz2, margin;
  double *tmpmap;
  const char *program = "lib_vwk";

  if (kernel_size < 1 || 2 * ((kernel_size + 1) / 2) - kernel_size - 1 != 0) {
    error_kernel_size(17140, program, kernel_size);
  }

  margin = (kernel_size - 1) / 2;
  out_nvox = in_extx * in_exty * in_extz;

  do_vect(&tmpmap, out_nvox);
  cp_vect(&tmpmap, &inmap, out_nvox); /* save inmap in case it gets overwritten */
  zero_vect(*outmap, out_nvox);

  for (indz = margin; indz < in_extz - margin; indz++)
    for (indy = margin; indy < in_exty - margin; indy++)
      for (indx = margin; indx < in_extx - margin; indx++) {
        dval = *(tmpmap + gidz_general(indz, indy, indx, in_exty, in_extx));
        if (dval != 0.0) for (indz2 = -margin; indz2 <= margin; indz2++)
            for (indy2 = -margin; indy2 <= margin; indy2++)
              for (indx2 = -margin; indx2 <= margin; indx2++) {
                *(*outmap + gidz_general(indz + indz2, indy + indy2, indx + indx2, in_exty, in_extx))
                += *(kernel + gidz_cube(indz2 + margin, indy2 + margin, indx2 + margin, kernel_size)) * dval;
              }
      }
  free_vect_and_zero_ptr((void**)&tmpmap);
}

/*====================================================================*/
/* convolves inmap with kernel inside "ignored" boundary, and writes to same-size outmap */
/* memory for *outmap must be allocated, and *outmap and inmap must point to different array */
/* speed-optimized version, no memory allocation, no error checks, norm variable */
void convolve_kernel_inside_fast(double **outmap, double *inmap,
                                 unsigned in_extx, unsigned in_exty, unsigned in_extz,
                                 double *kernel, unsigned kernel_size, double normfac, unsigned ignored[3])
{
  double dval, inv_normfac;
  unsigned long out_nvox;
  unsigned indx, indy, indz;
  int indx2, indy2, indz2, margin, marginx, marginy, marginz;

  margin = (kernel_size - 1) / 2;
  marginx = margin + ignored[0];
  marginy = margin + ignored[1];
  marginz = margin + ignored[2];
  out_nvox = in_extx * in_exty * in_extz;
  inv_normfac = 1.0 / normfac;

  zero_vect(*outmap, out_nvox);

  for (indz = marginz; indz < in_extz - marginz; indz++)
    for (indy = marginy; indy < in_exty - marginy; indy++)
      for (indx = marginx; indx < in_extx - marginx; indx++) {
        dval = (*(inmap + gidz_general(indz, indy, indx, in_exty, in_extx))) * inv_normfac;
        if (dval != 0.0) for (indz2 = -margin; indz2 <= margin; indz2++)
            for (indy2 = -margin; indy2 <= margin; indy2++)
              for (indx2 = -margin; indx2 <= margin; indx2++) {
                *(*outmap + gidz_general(indz + indz2, indy + indy2, indx + indx2, in_exty, in_extx))
                += *(kernel + gidz_cube(indz2 + margin, indy2 + margin, indx2 + margin, kernel_size)) * dval;
              }
      }
}

/*====================================================================*/
/* convolves inmap with kernel inside, and writes to same-size outmap*/
/* filters out convolutions where kernel hits zero density in *inmap */
/* to avoid cutoff edge effects when using finite difference kernel*/
/* sufficient memory for *outmap must already be allocated */
/* allows *outmap and inmap to point to same array - not speed optimized */
void convolve_kernel_inside_erode(double **outmap, double *inmap,
                                  unsigned in_extx, unsigned in_exty, unsigned in_extz,
                                  double *kernel, unsigned kernel_size)
{
  unsigned long out_nvox;
  unsigned indx, indy, indz;
  int indx2, indy2, indz2, margin;
  double *tmpmap;
  double dval, dval2;
  unsigned skip;
  const char *program = "lib_vwk";

  if (kernel_size < 1 || 2 * ((kernel_size + 1) / 2) - kernel_size - 1 != 0) {
    error_kernel_size(17150, program, kernel_size);
  }

  margin = (kernel_size - 1) / 2;
  out_nvox = in_extx * in_exty * in_extz;

  do_vect(&tmpmap, out_nvox);
  cp_vect(&tmpmap, &inmap, out_nvox); /* save inmap in case it gets overwritten */
  zero_vect(*outmap, out_nvox);

  for (indz = margin; indz < in_extz - margin; indz++)
    for (indy = margin; indy < in_exty - margin; indy++)
      for (indx = margin; indx < in_extx - margin; indx++) {
        skip = 0;
        for (indz2 = -margin; skip == 0 && indz2 <= margin; indz2++)
          for (indy2 = -margin; skip == 0 && indy2 <= margin; indy2++)
            for (indx2 = -margin; skip == 0 && indx2 <= margin; indx2++) { /* check if kernel hits zero density */
              dval = *(tmpmap + gidz_general(indz + indz2, indy + indy2, indx + indx2, in_exty, in_extx));
              dval2 = *(kernel + gidz_cube(margin - indz2, margin - indy2, margin - indx2, kernel_size));
              if (dval == 0.0 && dval2 != 0.0) skip = 1;
            }
        if (skip == 0) {
          for (indz2 = -margin; indz2 <= margin; indz2++)
            for (indy2 = -margin; indy2 <= margin; indy2++)
              for (indx2 = -margin; indx2 <= margin; indx2++) {
                dval = *(tmpmap + gidz_general(indz + indz2, indy + indy2, indx + indx2, in_exty, in_extx));
                dval2 = *(kernel + gidz_cube(margin - indz2, margin - indy2, margin - indx2, kernel_size));
                *(*outmap + gidz_general(indz, indy, indx, in_exty, in_extx)) += dval * dval2;
              }
        }
      }
  free_vect_and_zero_ptr((void**)&tmpmap);
}

/*====================================================================*/
/* convolves inmap with kernel overlapping the border, and writes to larger-size outmap */
/* sufficient memory for *outmap must already be allocated*/
/* allows *outmap and inmap to point to same array - not speed optimized*/
void convolve_kernel_outside(double **outmap, unsigned *out_extx, unsigned *out_exty, unsigned *out_extz,
                             double *out_origx, double *out_origy, double *out_origz,
                             double *inmap, unsigned in_extx, unsigned in_exty, unsigned in_extz,
                             double in_origx, double in_origy, double in_origz,
                             double widthx, double widthy, double widthz,
                             double *kernel, unsigned kernel_size)
{
  double dval;
  unsigned long out_nvox;
  unsigned long in_nvox;
  unsigned indx, indy, indz, tmp_extx, tmp_exty;
  int indx2, indy2, indz2, margin;
  double *tmpmap;
  const char *program = "lib_vwk";

  if (kernel_size < 1 || 2 * ((kernel_size + 1) / 2) - kernel_size - 1 != 0) {
    error_kernel_size(17160, program, kernel_size);
  }

  margin = (kernel_size - 1) / 2;
  tmp_extx = in_extx;
  tmp_exty = in_exty;
  *out_extx = kernel_size - 1 + in_extx; /* may overwrite in_extx */
  *out_exty = kernel_size - 1 + in_exty; /* may overwrite in_exty */
  *out_extz = kernel_size - 1 + in_extz; /* may overwrite in_extz */

  in_nvox = in_extx * in_exty * in_extz;
  out_nvox = (*out_extx) * (*out_exty) * (*out_extz);

  do_vect(&tmpmap, in_nvox);
  cp_vect(&tmpmap, &inmap, in_nvox); /* save inmap in case it gets overwritten */
  zero_vect(*outmap, out_nvox);

  for (indz = margin; indz < (*out_extz) - margin; indz++)
    for (indy = margin; indy < (*out_exty) - margin; indy++)
      for (indx = margin; indx < (*out_extx) - margin; indx++) {
        dval = *(tmpmap + gidz_general(indz - margin, indy - margin, indx - margin, tmp_exty, tmp_extx));
        if (dval != 0.0) for (indz2 = -margin; indz2 <= margin; indz2++)
            for (indy2 = -margin; indy2 <= margin; indy2++)
              for (indx2 = -margin; indx2 <= margin; indx2++) {
                *(*outmap + gidz_general(indz + indz2, indy + indy2, indx + indx2, (*out_exty), (*out_extx)))
                += *(kernel + gidz_cube(indz2 + margin, indy2 + margin, indx2 + margin, kernel_size)) * dval;
              }
      }
  free_vect_and_zero_ptr((void**)&tmpmap);
  *out_origx = in_origx - widthx * margin;
  *out_origy = in_origy - widthy * margin;
  *out_origz = in_origz - widthz * margin;
}


/*====================================================================*/
/* prints histogram of density map, nbins<=0 triggers interactive query, returns new nbins */

int print_histogram(unsigned *extx, unsigned *exty, unsigned *extz, 
                    double **phi, int nbins)
{


  unsigned long nvox, count;
  double maxdensity, mindensity, scale;
  int *his, *phis;
  int currp, currnp, current;
  int peak, nextpeak, i, j;
  double histotal;
  double *hisc;
  double *pphi;

  nvox = *extx * *exty * *extz;
  pphi = *phi;

  mindensity = calc_min(pphi, nvox);
  maxdensity = calc_max(pphi, nvox);
  printf("lib_vwk> Density information. min: %f max: %f ave: %f sig: %f norm: %f \n", mindensity, maxdensity, calc_average(pphi, nvox), calc_sigma(pphi, nvox), calc_norm(pphi, nvox));
  printf("lib_vwk> Above zero density information. ave: %f sig: %f norm: %f \n", calc_gz_average(pphi, nvox), calc_gz_sigma(pphi, nvox), calc_gz_norm(pphi, nvox));

  if (maxdensity > mindensity) {

    if (nbins <= 0) {
      printf("lib_vwk> Please enter the of number histogram bins: ");
      nbins = readln_int();
    }

    his = (int *) alloc_vect(nbins, sizeof(int));
    phis = (int *) alloc_vect(nbins, sizeof(int));
    hisc = (double *) alloc_vect(nbins, sizeof(double));

    printf("lib_vwk> Printing voxel histogram, %d histogram bins\n", nbins);
    printf("lib_vwk> (density value; voxel count; top-down cumulative volume fraction):\n");

    for (j = 0; j < nbins; j++) {
      his[j] = 0;
      hisc[j] = 0.0;
    }
    scale = ((nbins - 1) / (maxdensity - mindensity));
    for (count = 0; count < nvox; count++) {
      current = (int) floor(0.5 + (*(pphi + count) - mindensity) * scale);
      ++his[current];
    }

    /* determine two fullest bins for normalization */
    currp = 0;
    currnp = 0;
    nextpeak = 0;
    peak = 0;
    for (j = 0; j < nbins; j++) {
      if (his[j] >= currp) {
        currnp = currp;
        nextpeak = peak;
        currp = his[j];
        peak = j;
      } else if (his[j] >= currnp) {
        currnp = his[j];
        nextpeak = j;
      }
    }

    histotal = 0.0;
    for (j = 0; j < nbins; j++) histotal += his[j];
    hisc[0] = 1.0;

    if (histotal > 0) for (j = 1; j < nbins; j++) hisc[j] = hisc [j - 1] - (his[j - 1] / histotal);
    else for (j = 1; j < nbins; j++) hisc[j] = 0;

    if (his[nextpeak] > 0) {
      for (j = 0; j < nbins; j++) {
        phis[j] = (int) floor((BARL * his[j] / his[nextpeak]) + 0.5);
        if (phis[j] > BARL) phis[j] = BARL + 1;
      }
    } else {
      if (his[peak] > 0) {
        for (j = 0; j < nbins; j++) {
          phis[j] = (int)floor((BARL * his[j] / his[peak]) + 0.5);
          if (phis[j] > BARL) phis[j] = BARL + 1;
        }
      } else for (j = 0; j < nbins; j++) phis[j] = 0;
    }

    /* print histogram */
    for (j = 0; j < nbins; j++) {
      printf("%8.3f |", mindensity + (j / (nbins - 1.0)) * (maxdensity - mindensity));
      for (i = 0; i < phis[j]; ++i) printf("=");
      if (phis[j] > BARL) printf("->");
      printf(" %d ", his[j]);
      if (his[peak] > 0) {
        if (his[j] > 0) {
          if (phis[j] <= BARL) for (i = (int)(floor(log10(his[j])) - floor(log10(his[peak])) - BARL - 3 + phis[j]); i < 0; ++i) {
              if ((j % 2) || (i % 4)) printf(" ");
              else printf(".");
            }
        } else {
          for (i = -(int)floor(log10(his[peak])) - BARL - 3 + phis[j]; i < 0; ++i) {
            if ((j % 2) || (i % 4)) printf(" ");
            else printf(".");
          }
        }
      }
      printf("| %5.3e\n", hisc[j]);
    }
    printf("lib_vwk> Maximum at density value %5.3f\n", mindensity + (peak / (nbins - 1.0)) * (maxdensity - mindensity));
  }
  return nbins;
}

/*====================================================================*/
/* prints centered histogram of (difference) density map, blocking zero density */

void print_diff_histogram(unsigned *extx, unsigned *exty, unsigned *extz, 
                          double **phi, int nbins)
{


  unsigned long nvox, count;
  double maxdensity, mindensity, scale;
  int *his, *phis;
  int currp, current;
  int peak, i, j;
  double *pphi;

  nvox = *extx * *exty * *extz;
  pphi = *phi;

  /* make nbins odd */
  nbins = (1 + 2 * (nbins / 2));
  printf("lib_vwk> Printing centered difference histogram (%d histogram bins used):\n", nbins);


  mindensity = calc_min(pphi, nvox);
  maxdensity = calc_max(pphi, nvox);
  printf("lib_vwk> Difference density range: %f to %f, ", mindensity, maxdensity);
  if (maxdensity > -mindensity) {
    maxdensity = -mindensity;
    printf("upper tail clipped, ");
  } else {
    mindensity = -maxdensity;
    printf("lower tail clipped, ");
  }
  printf("showing %d histogram bins about center:\n", nbins);

  if (maxdensity > mindensity) {

    his = (int *) alloc_vect(nbins, sizeof(int));
    phis = (int *) alloc_vect(nbins, sizeof(int));


    /* set up histogram, ignoring zero values */
    for (j = 0; j < nbins; j++) his[j] = 0;
    scale = ((nbins - 1) / (maxdensity - mindensity));
    for (count = 0; count < nvox; count++) {
      if (*(pphi + count) != 0) {
        current = (int) floor(0.5 + (*(pphi + count) - mindensity) * scale);
        ++his[current];
      }
    }

    /* determine fullest bin for normalization */
    currp = 0;
    peak = 0;
    for (j = 0; j < nbins; j++) {
      if (his[j] >= currp) {
        currp = his[j];
        peak = j;
      }
    }
    if (his[peak] > 0) {
      for (j = 0; j < nbins; j++) {
        phis[j] = (int) floor((BARL * his[j] / his[peak]) + 0.5);
      }
    }

    /* print histogram */
    for (j = 0; j < nbins; j++) {
      printf("%8.3f |", mindensity + (j / (nbins - 1.0)) * (maxdensity - mindensity));
      for (i = 0; i < phis[j]; ++i) printf("=");
      printf(" %d ", his[j]);
      if (his[peak] > 0) {
        if (his[j] > 0) {
          if (phis[j] <= BARL) for (i = (int)(floor(log10(his[j])) - floor(log10(his[peak])) - BARL - 3 + phis[j]); i <= 0; ++i) {
              if (j == (nbins - 1) / 2) printf(".");
              else {
                if ((j % 2) || (i % 4)) printf(" ");
                else printf(".");
              }
            }
        } else {
          for (i = -(int)floor(log10(his[peak])) - BARL - 3 + phis[j]; i <= 0; ++i) {
            if (j == (nbins - 1) / 2) printf(".");
            else {
              if ((j % 2) || (i % 4)) printf(" ");
              else printf(".");
            }
          }
        }
      }
      printf("\n");
    }
  }
  return;
}

