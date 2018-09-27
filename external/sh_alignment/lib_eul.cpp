/*********************************************************************
*                           L I B _ E U L                            *
**********************************************************************
* Library is part of the Situs package URL: situs.biomachina.org     *
* (c) Jochen Heyd, Valerio Mariani, Pablo Chacon,                    *
* and Willy Wriggers, 2001-2007                                      *
**********************************************************************
*                                                                    *
* Euler angle - related routines.                                    *
*                                                                    *
**********************************************************************
* See legal statement for terms of distribution                      *
*********************************************************************/

/*

Euler angle order: (psi,theta,phi) <. (0,1,2)
Euler angle convention: Goldstein or "X-convention":

       A=BCD

rotation components:

   |  cos_psi  sin_psi 0 |
B= | -sin_psi  cos_psi 0 |
   |    0        0     1 |

   |  1      0          0      |
C= |  0   cos_theta  sin_theta |
   |  0  -sin_theta  cos_theta |

   |  cos_phi  sin_phi 0 |
D= | -sin_phi  cos_phi 0 |
   |    0        0     1 |


rotation matrix:

|a_11 a_12 a_13|
A=|a_21 a_22 a_23|
|a_31 a_22 a_33|

a_11 = cos_psi * cos_phi - cos_theta * sin_phi * sin_psi;
a_12 = cos_psi * sin_phi + cos_theta * cos_phi * sin_psi;
a_13 = sin_psi * sin_theta;
a_21 = -sin_psi * cos_phi- cos_theta * sin_phi * cos_psi;
a_22 = -sin_psi * sin_phi+ cos_theta * cos_phi * cos_psi;
a_23 =cos_psi * sin_theta;
a_31 =sin_theta * sin_phi;
a_32 = -sin_theta * cos_phi;
a_33 =cos_theta;

See http://mathworld.wolfram.com/
*/

#include "situs.h"
#include "lib_eul.h"
#include "lib_err.h"
#include "lib_pwk.h"
#include "lib_vec.h"

/*=====================================================================*/
/* writes range, step_size, and all angles to specified filename */
void write_eulers(char *eu_file, unsigned long eu_count, float *eu_store, 
                  double eu_range[3][2], double delta)
{
  FILE *out;
  const char *program = "lib_eul";
  unsigned i;

  if ((out = fopen(eu_file, "w")) == NULL) {
    error_open_filename(15010, program, eu_file);
  }

  for (i = 0; i < eu_count; i++)
    fprintf(out, "%10.4f%10.4f%10.4f\n",
            *(eu_store + 3 * i + 0) / ROT_CONV, *(eu_store + 3 * i + 1) / ROT_CONV, *(eu_store + 3 * i + 2) / ROT_CONV);
  fclose(out);
}

/*=====================================================================*/
/* reads Euler angles from in_file */
void read_eulers(char *in_file, unsigned long *eu_count, float **eu_store)
{
  FILE *in;
  unsigned long i;
  double psi, theta, phi;
  const char *program = "lib_eul";

  /* count number of Euler angle sets */

  if ((in = fopen(in_file, "r")) == NULL) {
    error_open_filename(15020, program, in_file);
  }
  i = 0;
  while (!feof(in)) {
    if (fscanf(in, "%lf %lf %lf\n", &psi, &theta, &phi) != 3) break;
    else i++;
  }
  fclose(in);

  /* allocate eu_store */
  *eu_store = (float *) alloc_vect(i * 3, sizeof(float));
  if ((in = fopen(in_file, "r")) == NULL) {
    error_open_filename(15050, program, in_file);
  }
  i = 0;
  while (!feof(in)) {
    if (fscanf(in, "%lf %lf %lf\n", &psi, &theta, &phi) != 3) break;
    else {
      *(*eu_store + i * 3 + 0) = psi * ROT_CONV;
      *(*eu_store + i * 3 + 1) = theta * ROT_CONV;
      *(*eu_store + i * 3 + 2) = phi * ROT_CONV;
      i++;
    }
  }
  fclose(in);
  *eu_count = i;
  printf("lib_eul> %lu Euler angles read from file %s\n", i, in_file);
}

/* remaps Euler angles using rot-matrix invariant transformations*/
/* and attempts to make angles fall in the intervalls: */
/* [psi_ref, psi_ref + 2*PI] */
/* [theta_ref, theta_ref + PI] */
/* [phi_ref, phi_ref + 2*PI] */
/* Note that this is strictly guaranteed only for*/
/* (psi_ref, theta_ref, phi_ref) == (0,0,0)*/
/* For non-zero reference we may have for small number of angles */
/* theta_ref + PI <= theta <= theta_ref + 2 * PI */
/*=====================================================================*/
void remap_eulers(double *psi_out, double *theta_out, double *phi_out,
                  double psi_in, double theta_in, double phi_in,
                  double psi_ref, double theta_ref, double phi_ref)
{
  double curr_psi, curr_theta, curr_phi;
  double new_psi, new_theta, new_phi;

  /* bring psi, theta, phi, within 2 PI of reference */
  curr_psi = psi_in - psi_ref;
  if (curr_psi >= 0) new_psi = fmod(curr_psi, 2 * PI) + psi_ref;
  else new_psi = 2 * PI - fmod(-curr_psi, 2 * PI) + psi_ref;

  curr_theta = theta_in - theta_ref;
  if (curr_theta >= 0) new_theta = fmod(curr_theta, 2 * PI) + theta_ref;
  else new_theta = 2 * PI - fmod(-curr_theta, 2 * PI) + theta_ref;

  curr_phi = phi_in - phi_ref;
  if (curr_phi >= 0) new_phi = fmod(curr_phi, 2 * PI) + phi_ref;
  else new_phi = 2 * PI - fmod(-curr_phi, 2 * PI) + phi_ref;

  /* if theta is not within PI, we use invariant transformations */
  /* and attempt to map to above intervals */
  /* this works in most cases even if the reference is not zero*/
  if (new_theta - theta_ref > PI) { /* theta overflow */
    /* theta . 2 PI - theta */
    if (new_theta >= 0) curr_theta = fmod(new_theta, 2 * PI);
    else curr_theta = 2 * PI - fmod(-new_theta, 2 * PI);
    new_theta -= 2 * curr_theta;

    /* remap to [0, 2 PI] interval */
    curr_theta = new_theta - theta_ref;
    if (curr_theta >= 0) new_theta = fmod(curr_theta, 2 * PI) + theta_ref;
    else new_theta = 2 * PI - fmod(-curr_theta, 2 * PI) + theta_ref;

    /* we have flipped theta so we need to flip psi and phi as well */
    /* to keep rot-matrix invariant */
    /* psi . psi + PI */
    if (new_psi - psi_ref > PI) new_psi -= PI;
    else new_psi += PI;

    /* phi . phi + PI */
    if (new_phi - phi_ref > PI) new_phi -= PI;
    else new_phi += PI;
  }

  *psi_out = new_psi;
  *theta_out = new_theta;
  *phi_out = new_phi;
}

/* computes rotation matrix based on Euler angles psi, theta, phi in radians */
/*=====================================================================*/
void get_rot_matrix(double dump[3][3], double psi, double theta, double phi)
{
  double sin_psi = sin(psi);
  double cos_psi = cos(psi);
  double sin_theta = sin(theta);
  double cos_theta = cos(theta);
  double sin_phi = sin(phi);
  double cos_phi = cos(phi);

  /* use Goldstein convention */
  dump[0][0] = cos_psi * cos_phi - cos_theta * sin_phi * sin_psi;
  dump[0][1] = cos_psi * sin_phi + cos_theta * cos_phi * sin_psi;
  dump[0][2] = sin_psi * sin_theta;
  dump[1][0] = -sin_psi * cos_phi - cos_theta * sin_phi * cos_psi;
  dump[1][1] = -sin_psi * sin_phi + cos_theta * cos_phi * cos_psi;
  dump[1][2] = cos_psi * sin_theta;
  dump[2][0] = sin_theta * sin_phi;
  dump[2][1] = -sin_theta * cos_phi;
  dump[2][2] = cos_theta;
}

/* Spiral algorithm for Euler angle computation in specified range.*/
/* The idea behind the algorithm is that one can cut the globe with n*/
/* horizontal planes spaced 2/(n-1) units apart, forming n circles of*/
/* latitude on the sphere, each latitude containing one spiral point.*/
/* To obtain the kth spiral point, one proceeds upward from the*/
/* (k-1)st point (theta(k-1), psi(k-1)) along a great circle to the*/
/* next latitude and travels counterclockwise along it for a fixed */
/* distance to arrive at the kth point (theta(k), psi(k)). */
/* */
/* Refs: (1) E.B. Saff and A.B.J. Kuijlaars, Distributing Many */
/* Points on a Sphere, The Mathematical Intelligencer, */
/* 19(1), Winter (1997). */
/* (2) "Computational Geometry in C." Joseph O'Rourke*/
/*=====================================================================*/
void eu_spiral(double eu_range[3][2], double delta, 
               unsigned long *eu_count, float **eu_store)
{
  unsigned long i, j;
  int phi_steps, n, k;
  double phi, phi_tmp, psi_old, psi_new, psi_tmp, theta, theta_tmp, h;
  const char *program = "lib_eul";

  /* rough estimate of number of points on sphere that give a surface*/
  /* density that is the squared linear density of angle increments*/
  n = (int)ceil(360.0 * 360.0 / (delta * delta * PI));

  /* total nr. points = nr. of points on the sphere * phi increments */
  phi_steps = (int)ceil((eu_range[2][1] - eu_range[2][0]) / delta);
  if (phi_steps < 1) {
    error_negative_euler(15080, program);
  }

  /* count number of points on the (theta,psi) sphere */
  j = 0;

  /* lower pole on (theta,psi) sphere, k=0 */
  theta = PI;
  psi_new = (eu_range[0][1] + eu_range[0][0]) * 0.5 * ROT_CONV;
  remap_eulers(&psi_new, &theta, &phi_tmp, psi_new, theta, 0.0,
               eu_range[0][0]*ROT_CONV, eu_range[1][0]*ROT_CONV, eu_range[2][0]*ROT_CONV);
  if (eu_range[1][0]*ROT_CONV <= theta && eu_range[1][1]*ROT_CONV >= theta) j++;


  /* intermediate sphere latitudes theta */
  psi_old = 0; /* longitude */
  for (k = 1; k < n - 1; k++) {
    h = -1 + 2 * k / (n - 1.0); /* linear distance from pole */
    theta = acos(h);
    psi_new = psi_old + 3.6 / (sqrt((double)n * (1 - h * h)));
    psi_old = psi_new;

    remap_eulers(&psi_new, &theta, &phi_tmp, psi_new, theta, 0.0,
                 eu_range[0][0]*ROT_CONV, eu_range[1][0]*ROT_CONV, eu_range[2][0]*ROT_CONV);

    if (eu_range[0][0]*ROT_CONV <= psi_new && eu_range[0][1]*ROT_CONV >= psi_new && eu_range[1][0]*ROT_CONV <= theta && eu_range[1][1]*ROT_CONV >= theta) j++;
  }

  /* upper pole on (theta,psi) sphere, k=n-1 */
  theta = 0.0;
  psi_new = (eu_range[0][1] + eu_range[0][0]) * 0.5 * ROT_CONV;
  remap_eulers(&psi_new, &theta, &phi_tmp, psi_new, theta, 0.0,
               eu_range[0][0]*ROT_CONV, eu_range[1][0]*ROT_CONV, eu_range[2][0]*ROT_CONV);
  if (eu_range[1][0]*ROT_CONV <= theta && eu_range[1][1]*ROT_CONV >= theta) j++;

  i = phi_steps * j;
  *eu_count = i;
  printf("lib_eul> Spiral Euler angle distribution, total number %lu (delta = %f deg.)\n", i, delta);

  /* allocate memory */

  *eu_store = (float *) alloc_vect(i * 3, sizeof(float));

  j = 0;
  /* lower pole on (theta,psi) sphere, k=0 */
  theta = PI;
  psi_new = (eu_range[0][1] + eu_range[0][0]) * 0.5 * ROT_CONV;
  remap_eulers(&psi_new, &theta, &phi_tmp, psi_new, theta, 0.0,
               eu_range[0][0]*ROT_CONV, eu_range[1][0]*ROT_CONV, eu_range[2][0]*ROT_CONV);
  if (eu_range[1][0]*ROT_CONV <= theta && eu_range[1][1]*ROT_CONV >= theta) {
    for (phi = eu_range[2][0]; phi <= eu_range[2][1]; phi += delta) {
      if (phi >= 360) break;
      remap_eulers(&psi_tmp, &theta_tmp, &phi_tmp, psi_new, theta, phi * ROT_CONV,
                   0.0, 0.0, 0.0);
      *(*eu_store + j + 0) = psi_tmp;
      *(*eu_store + j + 1) = theta_tmp;
      *(*eu_store + j + 2) = phi_tmp;
      j += 3;
    }
  }

  /* intermediate sphere latitudes theta */
  psi_old = 0; /* longitude */
  for (k = 1; k < n - 1; k++) {
    h = -1 + 2 * k / (n - 1.0); /* linear distance from pole */
    theta = acos(h);
    psi_new = psi_old + 3.6 / (sqrt((double)n * (1 - h * h)));
    psi_old = psi_new;
    remap_eulers(&psi_new, &theta, &phi_tmp, psi_new, theta, 0.0,
                 eu_range[0][0]*ROT_CONV, eu_range[1][0]*ROT_CONV, eu_range[2][0]*ROT_CONV);
    if (eu_range[0][0]*ROT_CONV <= psi_new && eu_range[0][1]*ROT_CONV >= psi_new && eu_range[1][0]*ROT_CONV <= theta && eu_range[1][1]*ROT_CONV >= theta) {
      for (phi = eu_range[2][0]; phi <= eu_range[2][1]; phi += delta) {
        if (phi >= 360) break;
        remap_eulers(&psi_tmp, &theta_tmp, &phi_tmp, psi_new, theta, phi * ROT_CONV,
                     0.0, 0.0, 0.0);
        *(*eu_store + j + 0) = psi_tmp;
        *(*eu_store + j + 1) = theta_tmp;
        *(*eu_store + j + 2) = phi_tmp;
        j += 3;
      }
    }
  }

  /* upper pole on (theta,psi) sphere, k=n-1 */
  theta = 0.0;
  psi_new = (eu_range[0][1] + eu_range[0][0]) * 0.5 * ROT_CONV;
  remap_eulers(&psi_new, &theta, &phi_tmp, psi_new, theta, 0.0,
               eu_range[0][0]*ROT_CONV, eu_range[1][0]*ROT_CONV, eu_range[2][0]*ROT_CONV);
  if (eu_range[1][0]*ROT_CONV <= theta && eu_range[1][1]*ROT_CONV >= theta) {
    for (phi = eu_range[2][0]; phi <= eu_range[2][1]; phi += delta) {
      if (phi >= 360) break;
      remap_eulers(&psi_tmp, &theta_tmp, &phi_tmp, psi_new, theta, phi * ROT_CONV,
                   0.0, 0.0, 0.0);
      *(*eu_store + j + 0) = psi_tmp;
      *(*eu_store + j + 1) = theta_tmp;
      *(*eu_store + j + 2) = phi_tmp;
      j += 3;
    }
  }
}

/* This subroutine generates a nondegenerate set of Euler angles */
/* in the specified scan range. The angles are first sampled with*/
/* equal spacing in psi, theta, phi. Finally, the distribution is*/
/* sparsed to reduce the density at the poles. */
/*=====================================================================*/
void eu_sparsed(double eu_range[3][2], double delta, 
                unsigned long *eu_count, float **eu_store)
{
  double psi, psi_tmp, theta, theta_tmp, phi, phi_tmp;
  char *degenerate;
  int psi_steps, theta_steps, phi_steps;
  int h;
  unsigned long i, j, n, k;
  unsigned long partial_count, curr_count;
  float *curr_eulers;
  const char *program = "lib_eul";
  PDB *pdb1;
  PDB *pdb2;
  PDB *pdb3;

  psi_steps = (int)ceil((eu_range[0][1] - eu_range[0][0]) / delta);
  theta_steps = (int)ceil((eu_range[1][1] - eu_range[1][0]) / delta);
  phi_steps = (int)ceil((eu_range[2][1] - eu_range[2][0]) / delta);

  if (psi_steps < 1 || theta_steps < 1 || phi_steps < 1) {
    error_negative_euler(15115, program);
  }

  partial_count = (psi_steps + 1) * (theta_steps + 1);

  /* allocate memory */
  curr_eulers = (float *) alloc_vect(partial_count * 3, sizeof(float));
  degenerate = (char *) alloc_vect(partial_count, sizeof(char));
  pdb1 = (PDB *) alloc_vect(1, sizeof(PDB));
  pdb2 = (PDB *) alloc_vect(1, sizeof(PDB));
  pdb3 = (PDB *) alloc_vect(1, sizeof(PDB));

  pdb1[0].x = 0.0;
  pdb1[0].y = 0.0;
  pdb1[0].z = 1.0;

  j = 0;
  n = 0;
  phi = 0;
  for (psi = 0; (float)psi <= psi_steps * delta && (float)psi < 360 ; psi += delta) {
    for (theta = 0; (float)theta <= theta_steps * delta && (float)theta < 360 ; theta += delta)  {
      remap_eulers(&psi_tmp, &theta_tmp, &phi_tmp,
                   (eu_range[0][0] + psi)*ROT_CONV,
                   (eu_range[1][0] + theta)*ROT_CONV,
                   (eu_range[2][0] + phi)*ROT_CONV,
                   0.0, 0.0, 0.0);
      *(curr_eulers + j + 0) = psi_tmp;
      *(curr_eulers + j + 1) = theta_tmp;
      *(curr_eulers + j + 2) = phi_tmp;
      n++;
      j += 3;
    }
  }

  partial_count = n;

  printf("lib_eul> Euler angles, initial combinations of psi and phi angles %lu (delta = %f deg.)\n", partial_count, delta);

  for (j = 0; j < partial_count; j++) *(degenerate + j) = 0;

  k = 0;
  printf("lib_eul> Searching for redundant orientations ");
  for (j = 0; j < partial_count; j++) if (*(degenerate + j) == 0) {
      if (fmod(j + 1, 1000) == 0.0)printf(".");
      rot_euler(pdb1, pdb2, 1, *(curr_eulers + j * 3 + 0), *(curr_eulers + j * 3 + 1), *(curr_eulers + j * 3 + 2));
      for (i = j + 1; i < partial_count; i++) if (*(degenerate + i) == 0) {
          rot_euler(pdb1, pdb3, 1, *(curr_eulers + i * 3 + 0), *(curr_eulers + i * 3 + 1), *(curr_eulers + i * 3 + 2));
          if ((pdb2[0].x - pdb3[0].x) * (pdb2[0].x - pdb3[0].x) + (pdb2[0].y - pdb3[0].y) * (pdb2[0].y - pdb3[0].y) +
              (pdb2[0].z - pdb3[0].z) * (pdb2[0].z - pdb3[0].z) < ((2.0 * sin(0.7 * delta * ROT_CONV / 2.0)) * (2.0 * sin(0.7 * delta * ROT_CONV / 2.0)))) {
            k++;
            *(degenerate + i) = 1;
          }
        }
    }

  printf("\n");
  *eu_count = (partial_count - k) * (phi_steps + 1);
  curr_count = (partial_count - k) * (phi_steps + 1);

  /* allocate memors */
  *eu_store = (float *) alloc_vect(curr_count * phi_steps * 3, sizeof(float));
  h = 0;
  printf("lib_eul> Adding psi angle to reduced set...\n");
  for (j = 0; j < partial_count; j++)  {
    if (*(degenerate + j) != 1) {
      for (phi = 0; (float)phi <= phi_steps * delta && (float)phi < 360 ; phi += delta)  {
        *(*eu_store + h * 3 + 0) = *(curr_eulers + j * 3 + 0);
        *(*eu_store + h * 3 + 1) = *(curr_eulers + j * 3 + 1);
        *(*eu_store + h * 3 + 2) = phi * ROT_CONV;
        h++;
      }
    }
  }

  *eu_count = h;

  printf("lib_eul> Euler angles, final number %lu (delta = %f deg.)\n", *eu_count, delta);

  free_vect_and_zero_ptr((void**)&pdb1);
  free_vect_and_zero_ptr((void**)&pdb2);
  free_vect_and_zero_ptr((void**)&pdb3);
  free_vect_and_zero_ptr((void**)&curr_eulers);
  free_vect_and_zero_ptr((void**)&degenerate);
}

/* returns cosine of deviation angle between two rotations */
/* as defined by Gabb et al., J. Mol. Biol. 272:106-120, 1997. */
/*=====================================================================*/
double deviation_cosine(double matrix1[3][3], double matrix2[3][3])
{
  double trace;
  int i, j;

  trace = 0;
  for (i = 0; i < 3; ++i) for (j = 0; j < 3; ++j) trace += matrix1[i][j] * matrix2[i][j];
  return (trace - 1.0) / 2.0;
}

/* checks if two orientations are within 1 degree*/
/* this can be slow if one of the orientations remains the same in a loop*/
/*=====================================================================*/
char similar_eulers(double a1, double a2, double a3, double b1, double b2, double b3)
{

  double deviation_cosine;
  double trace;
  double matrix1[3][3], matrix2[3][3];
  int i, j;

  get_rot_matrix(matrix2, a1, a2, a3);
  get_rot_matrix(matrix1, b1, b2, b3);
  trace = 0;
  for (i = 0; i < 3; ++i) for (j = 0; j < 3; ++j) trace += matrix2[i][j] * matrix1[i][j];
  deviation_cosine = (trace - 1.0) / 2.0;
  if (deviation_cosine <= 0.9998477) return 0;
  else return 1;
}

/* This subroutine generates a nondegenerate set of Euler angles */
/* in the specified scan range. For the full sphere, the number  */
/* of points is almost identical to the spiral method but without */
/* the helical slant and the weirdness at the poles. Angle       */
/* generation for subintervals is far superior and even cases    */
/* where the range can't be evenly devided by delta are handled  */
/* gracefully.                                                   */
/* The method works by dividing the sphere into slices and       */
/* determining how many psi angles need to be in a slice to re- */
/* produce the same spacing as on the equator.                  */
/*=====================================================================*/
void eu_proportional(double eu_range[3][2], double delta, 
                     unsigned long *eu_count, float **eu_store)
{
  double psi, theta, phi;
  double psi_ang_dist, psi_real_dist;
  double theta_real_dist, phi_real_dist;
  double psi_steps, theta_steps, phi_steps;
  double psi_range, theta_range, phi_range;
  unsigned long u, j;
  const char *program = "lib_eul";

  if ((eu_range[0][1] - eu_range[0][0]) / delta < -1  ||
      (eu_range[1][1] - eu_range[1][0]) / delta < -1  ||
      (eu_range[2][1] - eu_range[2][0]) / delta < -1) {
    error_negative_euler(15115, program);
  }

  psi_range   = eu_range[0][1] - eu_range[0][0];
  theta_range = eu_range[1][1] - eu_range[1][0];
  phi_range   = eu_range[2][1] - eu_range[2][0];

  /* Use rounding instead of CEIL to avoid rounding up at x.001 */

  phi_steps       = rint((phi_range / delta) + 0.499);
  phi_real_dist   = phi_range / phi_steps;

  theta_steps     = rint((theta_range / delta) + 0.499);
  theta_real_dist = theta_range / theta_steps;

  /* Computes the number of angles that will be generated */

  u = 0;
  for (phi = eu_range[2][0]; phi < 360.0 &&
       phi <= eu_range[2][1];  phi += phi_real_dist)  {
    for (theta = eu_range[1][0]; theta <= 180.0 &&
         theta <= eu_range[1][1];  theta += theta_real_dist)  {
      if (theta == 0.0 || theta == 180.0) {
        psi_steps = 1;
      } else {
        psi_steps = rint(360.0 * cos((90.0 - theta) * ROT_CONV) / delta);
      }
      psi_ang_dist  = 360.0 / psi_steps;
      psi_real_dist = psi_range / (ceil(psi_range / psi_ang_dist));
      for (psi = eu_range[0][0]; psi < 360.0 &&
           psi <= eu_range[0][1];  psi += psi_real_dist)  {
        u++;
      }
    }
  }

  *eu_count = u;

  /* allocate memory */
  *eu_store = (float *) alloc_vect(*eu_count * 3, sizeof(float));

  j = 0;
  for (phi = eu_range[2][0]; phi < 360.0 &&
       phi <= eu_range[2][1];  phi += phi_real_dist)  {
    for (theta = eu_range[1][0]; theta <= 180.0 &&
         theta <= eu_range[1][1];  theta += theta_real_dist)  {
      if (theta == 0.0 || theta == 180.0) {
        psi_steps = 1;
      } else {
        psi_steps = rint(360.0 * cos((90.0 - theta) * ROT_CONV) / delta);
      }
      psi_ang_dist  = 360.0 / psi_steps;
      psi_real_dist = psi_range / (ceil(psi_range / psi_ang_dist));
      for (psi = eu_range[0][0]; psi < 360.0 &&
           psi <= eu_range[0][1];  psi += psi_real_dist)  {

        *(*eu_store + j * 3 + 0) = (float)psi * ROT_CONV;
        *(*eu_store + j * 3 + 1) = (float)theta * ROT_CONV;
        *(*eu_store + j * 3 + 2) = (float)phi * ROT_CONV;

        j++;
      }
    }
  }

  printf("lib_eul> Proportional Euler angles distribution, total number %lu (delta = %f deg.)\n", *eu_count, delta);

  return;

}











