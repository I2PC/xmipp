/*********************************************************************
*                           L I B _ P W K                            *
**********************************************************************
* Library is part of the Situs package URL: situs.biomachina.org     *
* (c) Pablo Chacon, Jochen Heyd and Willy Wriggers, 2001-2009        *
**********************************************************************
*                                                                    *
* PDB structure manipulation tools.                                  *
*                                                                    *
**********************************************************************
* See legal statement for terms of distribution                      *
*********************************************************************/

#include "situs.h"
#include "lib_pio.h"
#include "lib_pwk.h"
#include "lib_vwk.h"
#include "lib_vec.h"
#include "lib_err.h"
#include "lib_eul.h"


/*====================================================================*/
/* copies ktot atoms from *pdb_original to *pdb_duplicate */
void copy_atoms(PDB *pdb_original, PDB *pdb_duplicate,
                int io, int id, int ktot)
{
  int j, k;
  for (k = 0; k < ktot; ++k) {
    pdb_duplicate[id + k].weight = pdb_original[io].weight;
    pdb_duplicate[id + k].x = pdb_original[io].x;
    pdb_duplicate[id + k].y = pdb_original[io].y;
    pdb_duplicate[id + k].z = pdb_original[io].z;
    for (j = 0; j < 4; ++j) pdb_duplicate[id + k].segid[j] = pdb_original[io].segid[j];
    pdb_duplicate[id + k].serial = id + k + 1;
    for (j = 0; j < 7; ++j) pdb_duplicate[id + k].recd[j] = pdb_original[io].recd[j];
    for (j = 0; j < 3; ++j) pdb_duplicate[id + k].type[j] = pdb_original[io].type[j];
    for (j = 0; j < 3; ++j) pdb_duplicate[id + k].loc[j] = pdb_original[io].loc[j];
    for (j = 0; j < 2; ++j) pdb_duplicate[id + k].alt[j] = pdb_original[io].alt[j];
    for (j = 0; j < 5; ++j) pdb_duplicate[id + k].res[j] = pdb_original[io].res[j];
    for (j = 0; j < 2; ++j) pdb_duplicate[id + k].chain[j] = pdb_original[io].chain[j];
    pdb_duplicate[id + k].seq = pdb_original[io].seq;
    for (j = 0; j < 2; ++j) pdb_duplicate[id + k].icode[j] = pdb_original[io].icode[j];
    pdb_duplicate[id + k].occupancy = pdb_original[io].occupancy;
    pdb_duplicate[id + k].beta = pdb_original[io].beta;
    pdb_duplicate[id + k].footnote = pdb_original[io].footnote;
    for (j = 0; j < 3; ++j) pdb_duplicate[id + k].element[j] = pdb_original[io].element[j];
    for (j = 0; j < 3; ++j) pdb_duplicate[id + k].charge[j] = pdb_original[io].charge[j];
  }
  return;
}


/*====================================================================*/
/* structure must be centered, rotates about X, Y, or Z axis */
void rot_axis(PDB *pdb_original, PDB *pdb_rotate, unsigned num_atoms,
              char axis, double angle)
{
  unsigned id;
  double x, y, z;
  double sint = sin(angle);
  double cost = cos(angle);

  switch (axis) {
    case ('X'):
      for (id = 0; id < num_atoms; ++id) {
        y = pdb_original[id].y;
        z = pdb_original[id].z;
        pdb_rotate[id].x = pdb_original[id].x;
        pdb_rotate[id].y = (cost * y + sint * z);
        pdb_rotate[id].z = (cost * z - sint * y);
      }
      break;
    case ('Y'):
      for (id = 0; id < num_atoms; ++id) {
        x = pdb_original[id].x;
        z = pdb_original[id].z;
        pdb_rotate[id].y = pdb_original[id].y;
        pdb_rotate[id].x = (cost * x + sint * z);
        pdb_rotate[id].z = (cost * z - sint * x);
      }
      break;
    case ('Z'):
      for (id = 0; id < num_atoms; ++id) {
        x = pdb_original[id].x;
        y = pdb_original[id].y;
        pdb_rotate[id].z = pdb_original[id].z;
        pdb_rotate[id].x = (cost * x + sint * y);
        pdb_rotate[id].y = (cost * y - sint * x);
      }
      break;
  }
}


/*====================================================================*/
/* structure must be centered, rotates by Euler angles */
void rot_euler(PDB *pdb_original, PDB *pdb_rotate,
               unsigned num_atoms, double psi, double theta, double phi)
{
  unsigned id;
  double rot_matrix[3][3], currx, curry, currz;

  get_rot_matrix(rot_matrix, psi, theta, phi);

  for (id = 0; id < num_atoms; ++id) {
    currx = pdb_original[id].x;
    curry = pdb_original[id].y;
    currz = pdb_original[id].z;
    pdb_rotate[id].x = currx * rot_matrix[0][0] +
                       curry * rot_matrix[0][1] +
                       currz * rot_matrix[0][2];
    pdb_rotate[id].y = currx * rot_matrix[1][0] +
                       curry * rot_matrix[1][1] +
                       currz * rot_matrix[1][2];
    pdb_rotate[id].z = currx * rot_matrix[2][0] +
                       curry * rot_matrix[2][1] +
                       currz * rot_matrix[2][2];
  }
}


/*====================================================================*/
/* translates *pdb_original and stores in *pdb_move */
void translate(PDB *pdb_original, PDB *pdb_move,
               unsigned num_atoms, double x0, double y0, double z0)
{
  unsigned id;

  for (id = 0; id < num_atoms; ++id) {
    pdb_move[id].x = pdb_original[id].x + x0;
    pdb_move[id].y = pdb_original[id].y + y0;
    pdb_move[id].z = pdb_original[id].z + z0;
  }
}


/*====================================================================*/
/* computes geometric center of structure */
void calc_center(PDB *pdb0, unsigned num_atoms,
                 double *cx, double *cy, double *cz)
{
  unsigned id;
  const char *program = "lib_pwk";
  *cx = 0.0;
  *cy = 0.0;
  *cz = 0.0;
  for (id = 0; id < num_atoms; ++id) {
    *cx += pdb0[id].x;
    *cy += pdb0[id].y;
    *cz += pdb0[id].z;
  }
  if (num_atoms > 0) {
    *cx /= (num_atoms * 1.0);
    *cy /= (num_atoms * 1.0);
    *cz /= (num_atoms * 1.0);
  } else {
    error_divide_zero(16010, program);
  }
}


/*====================================================================*/
/* returns total mass of structure */
double calc_mass(PDB *pdb0, unsigned num_atoms)
{
  unsigned id;
  double mtot;

  mtot = 0.0;
  for (id = 0; id < num_atoms; ++id)
    mtot += pdb0[id].weight;
  return mtot;
}


/*====================================================================*/
/* computes COM and returns total mass of structure */
double calc_center_mass(PDB *pdb0, unsigned num_atoms,
                        double *cx, double *cy, double *cz)
{
  unsigned id;
  double mtot;
  const char *program = "lib_pwk";

  *cx = 0.0;
  *cy = 0.0;
  *cz = 0.0;
  mtot = 0.0;
  for (id = 0; id < num_atoms; ++id) {
    mtot += pdb0[id].weight;
    *cx += pdb0[id].x * pdb0[id].weight;
    *cy += pdb0[id].y * pdb0[id].weight;
    *cz += pdb0[id].z * pdb0[id].weight;
  }
  if (mtot > 0.0) {
    *cx /= mtot;
    *cy /= mtot;
    *cz /= mtot;
  } else {
    error_divide_zero(16020, program);
  }
  return mtot;
}


/*====================================================================*/
/* computes bounding radius of structure relative to input center */
double calc_sphere(PDB *pdb0, unsigned num_atoms,
                   double cx, double cy, double cz)
{
  unsigned id;
  double maxradius, currradius;
  const char *program = "lib_pwk";
  const char *shape = "sphere";

  maxradius = -1e20;
  for (id = 0; id < num_atoms; ++id) {
    currradius = ((pdb0[id].x - cx) * (pdb0[id].x - cx) +
                  (pdb0[id].y - cy) * (pdb0[id].y - cy) +
                  (pdb0[id].z - cz) * (pdb0[id].z - cz));
    if (currradius > maxradius) maxradius = currradius;
  }

  if (maxradius >= 0.0) {
    maxradius = sqrt(maxradius);
    return maxradius;
  } else {
    error_no_bounding(16030, program, shape);
    return 0;
  }
}


/*====================================================================*/
/* computes bounding box of structure */
void calc_box(PDB *pdb0, unsigned num_atoms,
              double *minx, double *miny, double *minz,
              double *maxx, double *maxy, double *maxz)
{
  unsigned id;
  const char *program = "lib_pwk";
  const char *shape = "box";

  *minx = 1e20;
  *miny = 1e20;
  *minz = 1e20;
  *maxx = -1e20;
  *maxy = -1e20;
  *maxz = -1e20;
  for (id = 0; id < num_atoms; ++id) {
    if (*minx > pdb0[id].x) *minx = pdb0[id].x;
    if (*maxx < pdb0[id].x) *maxx = pdb0[id].x;
    if (*miny > pdb0[id].y) *miny = pdb0[id].y;
    if (*maxy < pdb0[id].y) *maxy = pdb0[id].y;
    if (*minz > pdb0[id].z) *minz = pdb0[id].z;
    if (*maxz < pdb0[id].z) *maxz = pdb0[id].z;
  }
  if (*minx < 1e20 && *miny < 1e20 && *minz < 1e20 && *maxx > -1e20 && *maxy > -1e20 && *maxz > -1e20) return;
  else {
    error_no_bounding(16040, program, shape);
  }
}


/*====================================================================*/
/* projects PDB to lattice using mass-weighting for colores and collage */
/* assumes both structure and map are centered (before any shifting) */
void project_mass(double **outmap, unsigned long nvox, double widthx, double widthy, double widthz, unsigned extx, unsigned exty, unsigned extz,
                  PDB *inpdb, unsigned num_atoms, double shift[3], unsigned ignored[3])
{
  int x0, y0, z0, x1, y1, z1, i;
  double gx, gy, gz;
  double a, b, c;
  double ab, ab1, a1b, a1b1;
  int marglx, margly, marglz, margux, marguy, marguz;

  /* lower margins */
  marglx = ignored[0];
  margly = ignored[1];
  marglz = ignored[2];

  /* one voxel safety buffer for boundary test below */
  if (marglx == 0) marglx = 1;
  if (margly == 0) margly = 1;
  if (marglz == 0) marglz = 1;

  /* upper margins */
  margux = extx - marglx;
  marguy = exty - margly;
  marguz = extz - marglz;

  zero_vect(*outmap, nvox);
  for (i = 0; i < num_atoms; ++i) {

    /* compute position within grid in voxel units */

    gx = extx / 2.0 + (inpdb[i].x + shift[0]) / widthx;
    gy = exty / 2.0 + (inpdb[i].y + shift[1]) / widthy;
    gz = extz / 2.0 + (inpdb[i].z + shift[2]) / widthz;
    x0 = floor(gx);
    y0 = floor(gy);
    z0 = floor(gz);
    x1 = x0 + 1;
    y1 = y0 + 1;
    z1 = z0 + 1;

    /* boundary check considers if probe cube overlaps at least partially with allowed region (stripped of ignored) */
    if (x1 >= marglx && x0 < margux && y1 >= margly && y0 < marguy && z1 >= marglz && z0 < marguz) {

      /* interpolate */
      a = x1 - gx;
      b = y1 - gy;
      c = z1 - gz;
      ab = a * b;
      ab1 = a * (1 - b);
      a1b = (1 - a) * b;
      a1b1 = (1 - a) * (1 - b);
      a = (1 - c);
      *(*outmap + gidz_general(z0, y0, x0, exty, extx)) += ab * c * inpdb[i].weight;
      *(*outmap + gidz_general(z1, y0, x0, exty, extx)) += ab * a * inpdb[i].weight;
      *(*outmap + gidz_general(z0, y1, x0, exty, extx)) += ab1 * c * inpdb[i].weight;
      *(*outmap + gidz_general(z1, y1, x0, exty, extx)) += ab1 * a * inpdb[i].weight;
      *(*outmap + gidz_general(z0, y0, x1, exty, extx)) += a1b * c * inpdb[i].weight;
      *(*outmap + gidz_general(z1, y0, x1, exty, extx)) += a1b * a * inpdb[i].weight;
      *(*outmap + gidz_general(z0, y1, x1, exty, extx)) += a1b1 * c * inpdb[i].weight;
      *(*outmap + gidz_general(z1, y1, x1, exty, extx)) += a1b1 * a * inpdb[i].weight;
    }
  }
}


/*====================================================================*/
/* projects PDB to lattice using mass-weighting and carries out */
/* fast (inside) kernel convolution and correlation calculation in one step */
/* assumes both structure and map are centered (before any shifting) */

void project_mass_convolve_kernel_corr(double widthx, double widthy,  double widthz,
                                       unsigned extx, unsigned exty, unsigned extz,
                                       PDB *inpdb, unsigned num_atoms, double shift[3],
                                       double *kernel, unsigned kernel_size, double normfac, unsigned ignored[3],
                                       double *lowmap, double *corr_hi_low)
{
  int x0, y0, z0, x1, y1, z1, i;
  double gx, gy, gz;
  double a, b, c;
  double ab, ab1, a1b, a1b1;
  double dval1, dval2, dval3, dval4, dval5, dval6, dval7, dval8;
  int indx2, indy2, indz2, margin;
  double inv_normfac;
  double *idx_kern;
  double *idx_low1, *idx_low2, *idx_low3, *idx_low4;
  int marglx, margly, marglz, margux, marguy, marguz;

  inv_normfac = 1.0 / normfac;
  margin = (kernel_size - 1) / 2;

  /* lower margins */
  marglx = margin + ignored[0];
  margly = margin + ignored[1];
  marglz = margin + ignored[2];

  /* one voxel safety buffer for boundary test below */
  if (marglx == margin) marglx += 1;
  if (margly == margin) margly += 1;
  if (marglz == margin) marglz += 1;

  /* upper margins */
  margux = extx - marglx;
  marguy = exty - margly;
  marguz = extz - marglz;

  for (i = 0; i < num_atoms; ++i) {

    /* compute position of probe cube within grid in voxel units */

    gx = extx / 2.0 + (inpdb[i].x + shift[0]) / widthx;
    gy = exty / 2.0 + (inpdb[i].y + shift[1]) / widthy;
    gz = extz / 2.0 + (inpdb[i].z + shift[2]) / widthz;
    x0 = floor(gx);
    y0 = floor(gy);
    z0 = floor(gz);
    x1 = x0 + 1;
    y1 = y0 + 1;
    z1 = z0 + 1;

    /* boundary check considers if probe cube overlaps at least partially with original map (stripped of margin + ignored) */
    if (x1 >= marglx && x0 < margux && y1 >= margly && y0 < marguy && z1 >= marglz && z0 < marguz) {

      /* interpolate */
      a = x1 - gx;
      b = y1 - gy;
      c = z1 - gz;
      ab = a * b;
      ab1 = a * (1 - b);
      a1b = (1 - a) * b;
      a1b1 = (1 - a) * (1 - b);
      a = (1 - c);

      dval1 = inv_normfac * ab   * c * inpdb[i].weight;
      dval2 = inv_normfac * a1b  * c * inpdb[i].weight;
      dval3 = inv_normfac * ab1  * c * inpdb[i].weight;
      dval4 = inv_normfac * a1b1 * c * inpdb[i].weight;
      dval5 = inv_normfac * ab   * a * inpdb[i].weight;
      dval6 = inv_normfac * a1b  * a * inpdb[i].weight;
      dval7 = inv_normfac * ab1  * a * inpdb[i].weight;
      dval8 = inv_normfac * a1b1 * a * inpdb[i].weight;

      idx_kern = kernel;
      for (indz2 = -margin; indz2 <= margin; indz2++) {
        for (indy2 = -margin; indy2 <= margin; indy2++) {
          idx_low1 = lowmap + (exty * extx * (z0 + indz2) + extx * (y0 + indy2) + x0);
          idx_low2 = lowmap + (exty * extx * (z0 + indz2) + extx * (y1 + indy2) + x0);
          idx_low3 = lowmap + (exty * extx * (z1 + indz2) + extx * (y0 + indy2) + x0);
          idx_low4 = lowmap + (exty * extx * (z1 + indz2) + extx * (y1 + indy2) + x0);
          for (indx2 = -margin; indx2 <= margin; indx2++) {
            *corr_hi_low  += *(idx_kern) *
                             (dval1 * *(idx_low1 + indx2)
                              + dval2 * *(idx_low1 + indx2 + 1)
                              + dval3 * *(idx_low2 + indx2)
                              + dval4 * *(idx_low2 + indx2 + 1)
                              + dval5 * *(idx_low3 + indx2)
                              + dval6 * *(idx_low3 + indx2 + 1)
                              + dval7 * *(idx_low4 + indx2)
                              + dval8 * *(idx_low4 + indx2 + 1));
            idx_kern++;
          }
        }
      }  /* end of loop over kernel */

    } /* end of boundary check */

  } /* end of loop over atoms */

}


/*====================================================================*/
/* checks if at least a fraction of all atoms inside comparison map */
/* assumes both structure and map are centered (before any shifting) */
int check_if_inside(double fraction, double widthx, double widthy, double widthz,
                    unsigned extx, unsigned exty, unsigned extz,
                    PDB *inpdb, unsigned num_atoms, double shift[3])
{

  int x0, y0, z0, x1, y1, z1, i;
  double gx, gy, gz;
  int outside_error = 0;

  for (i = 0; i < num_atoms; ++i) {
    /* compute position within grid  */
    gx = extx / 2.0 + (inpdb[i].x + shift[0]) / widthx;
    gy = exty / 2.0 + (inpdb[i].y + shift[1]) / widthy;
    gz = extz / 2.0 + (inpdb[i].z + shift[2]) / widthz;
    x0 = floor(gx);
    y0 = floor(gy);
    z0 = floor(gz);
    x1 = x0 + 1;
    y1 = y0 + 1;
    z1 = z0 + 1;
    if (x0 < 0) ++outside_error;
    if (y0 < 0) ++outside_error;
    if (z0 < 0) ++outside_error;
    if (x1 >= extx) ++outside_error;
    if (y1 >= exty) ++outside_error;
    if (z1 >= extz) ++outside_error;
  }
  if (outside_error >= (1 - fraction)*num_atoms) return outside_error;
  else return 0;
}
