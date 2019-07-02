/*********************************************************************
*                            L I B _ V I O                           *
**********************************************************************
* Library is part of the Situs package (c) Willy Wriggers, 1998-2012 *
* URL: situs.biomachina.org                                          *
**********************************************************************
*                                                                    *
* Tools for map input and output.                                    *
*                                                                    *
**********************************************************************
* See legal statement for terms of distribution                      *
*********************************************************************/

#include "situs.h"
#include "lib_vio.h"
#include "lib_vwk.h"
#include "lib_vec.h"
#include "lib_std.h"
#include "lib_err.h"
#include <math.h>

#define FLENGTH 1000  /* file name length */

/* read volume in Situs-compatible format, currently CCP4/MRC or Situs, autodetected on read */
void read_vol(char *vol_file, double *width, double *origx, double *origy, 
              double *origz, unsigned *extx, unsigned *exty, unsigned *extz, 
              double **phi)
{
  int n_range_viol0, n_range_viol1, n_range_viol2;
  int ordermode = 7, cubic = 1, orom = 1;
  unsigned nc, nr, ns;
  unsigned nx, ny, nz;
  int nxstart = 0, nystart = 0, nzstart = 0;
  int ncstart = 0, nrstart = 0, nsstart = 0;
  double widthx, widthy, widthz;
  double xorigin, yorigin, zorigin;
  double alpha, beta, gamma;
  double *phi_raw;

  n_range_viol0 = test_mrc(vol_file, 0);
  n_range_viol1 = test_mrc(vol_file, 1);
  n_range_viol2 = test_situs(vol_file);

  if (n_range_viol2 < n_range_viol0 && n_range_viol2 < n_range_viol1) 
    read_situs(vol_file, width, origx, origy, origz, extx, exty, extz, phi);
  else {
    read_mrc(vol_file, &orom, &cubic, &ordermode, &nc, &nr, &ns, &ncstart, 
             &nrstart, &nsstart, &widthx, &widthy, &widthz, &xorigin, &yorigin, 
             &zorigin, &alpha, &beta, &gamma, &phi_raw);
    permute_map(ordermode, nc, nr, ns, &nx, &ny, &nz, ncstart, nrstart, nsstart,
                &nxstart, &nystart, &nzstart, phi_raw, phi);
    permute_print_info(ordermode, nc, nr, ns, nx, ny, nz, ncstart, nrstart, 
                       nsstart, nxstart, nystart, nzstart);
    assert_cubic_map(orom, cubic, alpha, beta, gamma, widthx, widthy, widthz, 
                     nx, ny, nz, nxstart, nystart, nzstart, xorigin, yorigin, 
                     zorigin, width, origx, origy, origz, extx, exty, extz, phi);
    printf("lib_vio> \n");
  }

  return;
}


/* write volume in Situs-compatible format, currently Situs or CCP4/MRC, format selected based on suffix of vol_file */
void write_vol(char *vol_file, double width, double origx, double origy, 
               double origz, unsigned extx, unsigned exty, unsigned extz, 
               double *phi)
{

  if (have_situs_suffix(vol_file)) 
    write_situs(vol_file, width, origx, origy, origz, extx, exty, extz, phi);
  else 
    write_mrc(1, vol_file, width, origx, origy, origz, extx, exty, extz, phi);

  return;
}


/* read Situs format map */
void read_situs(char *vol_file, double *width, double *origx, double *origy, 
                double *origz, unsigned *extx, unsigned *exty, unsigned *extz, 
                double **phi)
{
  unsigned long nvox, count;
  double dorigx, dorigy, dorigz, dwidth;
  double phitest, dtemp;
  const char *program = "lib_vio";
  FILE *fin;

  fin = fopen(vol_file, "r");
  if (fin == NULL) {
    error_open_filename(13010, program, vol_file);
  }

  /* read header and print information */
  if (7 != fscanf(fin, "%le %le %le %le %d %d %d", &dwidth, &dorigx, &dorigy, &dorigz, extx, exty, extz)) error_fscanf(program, vol_file);
  *width = dwidth;
  *origx = dorigx;
  *origy = dorigy;
  *origz = dorigz;
  printf("lib_vio> Situs formatted map file %s - Header information: \n", vol_file);
  printf("lib_vio> Columns, rows, and sections: x=1-%d, y=1-%d, z=1-%d\n", *extx, *exty, *extz);
  printf("lib_vio> 3D coordinates of first voxel: (%f,%f,%f)\n", *origx, *origy, *origz);
  printf("lib_vio> Voxel size in Angstrom: %f \n", *width);

  nvox = *extx * *exty * *extz;

  /* allocate memory and read data */
  printf("lib_vio> Reading density data... \n");
  *phi = (double *) alloc_vect(nvox, sizeof(double));

  for (count = 0; count < nvox; count++) {
    if (fscanf(fin, "%le", &dtemp) != 1) {
      error_unreadable_file_long(13030, program, vol_file);
    } else *(*phi + count) = dtemp;
  }
  if (fscanf(fin, "%le", &phitest) != EOF) {
    error_unreadable_file_long(13040, program, vol_file);
  }
  fclose(fin);
  printf("lib_vio> Volumetric data read from file %s\n", vol_file);
  return;
}


/* write Situs format map */
void write_situs(char *vol_file, double width, double origx, double origy, 
                 double origz, unsigned extx, unsigned exty, unsigned extz, 
                 double *phi)
{
  unsigned long nvox, count;
  const char *program = "lib_vio";
  FILE *fout;

  nvox = extx * exty * extz;
  fout = fopen(vol_file, "w");
  if (fout == NULL) {
    error_open_filename(13210, program, vol_file);
  }

  printf("lib_vio> Writing density data... \n");
  fprintf(fout, "%f %f %f %f %d %d %d\n", width, origx, origy, origz, extx, exty, extz);
  fprintf(fout, "\n");

  for (count = 0; count < nvox; count++) {
    if ((count + 1) % 10 == 0) fprintf(fout, " %10.6f \n", *(phi + count));
    else fprintf(fout, " %10.6f ", *(phi + count));
  }
  fclose(fout);
  printf("lib_vio> Volumetric data written in Situs format to file %s \n", vol_file);

  /* header information */
  printf("lib_vio> Situs formatted map file %s - Header information: \n", vol_file);
  printf("lib_vio> Columns, rows, and sections: x=1-%d, y=1-%d, z=1-%d\n", extx, exty, extz);
  printf("lib_vio> 3D coordinates of first voxel: (%f,%f,%f)\n", origx, origy, origz);
  printf("lib_vio> Voxel size in Angstrom: %f \n", width);

  return;
}


/* reads ASCII stream */
void read_ascii(char *vol_file, unsigned long nvox, double **fphi)
{
  unsigned long count;
  FILE *fin;
  float currfloat;

  fin = fopen(vol_file, "r");
  if (fin == NULL) {
    error_open_filename(70310, "lib_vio", vol_file);
  }
  printf("lib_vio> Reading ASCII data... \n");
  do_vect(fphi, nvox);
  for (count = 0; count < nvox; count++) {
    if (fscanf(fin, "%e", &currfloat) != 1) {
      error_unreadable_file_short(70330 , "lib_vio", vol_file);
    } else {
      *(*fphi + count) = currfloat;
    }
  }
  if (fscanf(fin, "%e", &currfloat) != EOF) {
    error_unreadable_file_long(70340 , "lib_vio", vol_file);
  }
  printf("lib_vio> Volumetric data read from file %s\n", vol_file);
  fclose(fin);
}


/* reads X-PLOR ASCII file */
void read_xplor(char *vol_file, int *orom, int *cubic,
                double *widthx, double *widthy, double *widthz,
                double *alpha, double *beta, double *gamma,
                int *nxstart, int *nystart, int *nzstart,
                unsigned *extx, unsigned *exty, unsigned *extz, double **fphi)
{

  unsigned long nvox;
  FILE *fin;
  int idummy, done;
  char *nextline;
  float a_tmp, b_tmp, g_tmp, currfloat;
  long mx, my, mz, mxend, myend, mzend;
  long mxstart, mystart, mzstart;
  int testa, testb, testg;
  float xlen, ylen, zlen;
  int indx, indy, indz;

  nextline = (char *) alloc_vect(FLENGTH, sizeof(char));

  fin = fopen(vol_file, "r");
  if (fin == NULL) {
    error_open_filename(70420, "lib_vio", vol_file);
  }

  /* ignore header length line */
  xplor_skip_to_number(&fin, &nextline);

  /* read index line */
  xplor_skip_to_number(&fin, &nextline);
  if (sscanf(nextline, "%8ld%8ld%8ld%8ld%8ld%8ld%8ld%8ld%8ld", &mx, &mxstart, &mxend, &my, &mystart, &myend, &mz, &mzstart, &mzend) != 9) {
    error_xplor_file_indexing(70430, "lib_vio");
  }
  *extx = mxend - mxstart + 1;
  *exty = myend - mystart + 1;
  *extz = mzend - mzstart + 1;
  nvox = *extx * *exty * *extz;

  printf("lib_vio> X-PLOR map indexing (counting from 0): \n");
  printf("lib_vio>       NA = %8ld  (# of X intervals in unit cell) \n", mx);
  printf("lib_vio>     AMIN = %8ld  (start index X) \n", mxstart);
  printf("lib_vio>     AMAX = %8ld  (end index X) \n", mxend);
  printf("lib_vio>       NB = %8ld  (# of Y intervals in unit cell) \n", my);
  printf("lib_vio>     BMIN = %8ld  (start index Y) \n", mystart);
  printf("lib_vio>     BMAX = %8ld  (end index Y) \n", myend);
  printf("lib_vio>       NC = %8ld  (# of Z intervals in unit cell) \n", mz);
  printf("lib_vio>     CMIN = %8ld  (start index Z) \n", mzstart);
  printf("lib_vio>     CMAX = %8ld  (end index Z) \n", mzend);

  /* read unit cell info and determine grid width and origin */
  xplor_skip_to_number(&fin, &nextline);
  if (sscanf(nextline, "%12f%12f%12f%12f%12f%12f", &xlen, &ylen, &zlen, &a_tmp, &b_tmp, &g_tmp) != 6) {
    error_xplor_file_unit_cell(70440, "lib_vio");
  } else {
    *alpha = a_tmp;
    *beta = b_tmp;
    *gamma = g_tmp;
  }

  printf("lib_vio> X-PLOR unit cell info: \n");
  printf("lib_vio>        A = %8.3f  (unit cell dimension) \n", xlen);
  printf("lib_vio>        B = %8.3f  (unit cell dimension) \n", ylen);
  printf("lib_vio>        C = %8.3f  (unit cell dimension) \n", zlen);
  printf("lib_vio>    ALPHA = %8.3f  (unit cell angle) \n", *alpha);
  printf("lib_vio>     BETA = %8.3f  (unit cell angle) \n", *beta);
  printf("lib_vio>    GAMMA = %8.3f  (unit cell angle) \n", *gamma);

  /* assign voxel spacing parameters */
  *widthx = xlen / (double) mx;
  *widthy = ylen / (double) my;
  *widthz = zlen / (double) mz;
  *nxstart = mxstart;
  *nystart = mystart;
  *nzstart = mzstart;

  /* test for orthogonal and cubic lattice */
  testa = (int)floor(100 * *alpha + 0.5);
  testb = (int)floor(100 * *beta + 0.5);
  testg = (int)floor(100 * *gamma + 0.5);
  if (testa != 9000 || testb != 9000 || testg != 9000) *orom = 0;
  if (*orom == 0 || floor((*widthx - *widthy) * 1000 + 0.5) != 0 || 
      floor((*widthy - *widthz) * 1000 + 0.5) != 0 || 
      floor((*widthx - *widthz) * 1000 + 0.5) != 0) 
    *cubic = 0;

  /* read ZYX info */
  for (done = 0; done == 0;) { /* read next line and check if it contains ZYX */
    if (fgets(nextline, FLENGTH, fin) == NULL) {
      error_EOF_ZYX_mode(70450, "lib_vio");
    }
    if (*nextline == 'Z' && *(nextline + 1) == 'Y' && *(nextline + 2) == 'X') {
      done = 1;
      break;
    }
    if (*nextline == 'z' && *(nextline + 1) == 'y' && *(nextline + 2) == 'x') {
      done = 1;
      break;
    }
  }

  /* read sections */
  do_vect(fphi, nvox);
  for (indz = 0; indz < (int)*extz; indz++) {
    /* skip section header and read section number */
    xplor_skip_to_number(&fin, &nextline);
    if (sscanf(nextline, "%8d", &idummy) != 1) {
      error_xplor_file_map_section(70470, "lib_vio");
    }
    if (idummy != indz) {
      error_xplor_file_map_section_number(70480, "lib_vio");
    }

    /* read section data */
    for (indy = 0; indy < (int)*exty; indy++) for (indx = 0; indx < (int)*extx; indx++) {
        if (fscanf(fin, "%12f", &currfloat) != 1) {
          error_unreadable_file_short(70490, "lib_vio", vol_file);
        } else {
          *(*fphi + gidz_general(indz, indy, indx, *exty, *extx)) = currfloat;
        }
      }
  }

  /* read end of data marker */
  xplor_skip_to_number(&fin, &nextline);
  if (sscanf(nextline, "%8d", &idummy) != 1 || idummy != -9999) {
    error_xplor_maker("lib_vio");
  }

  fclose(fin);
  free_vect_and_zero_ptr((void**)&nextline);
  printf("lib_vio> Volumetric data read from file %s\n", vol_file);
}


/* reads lines from current position in opened X-PLOR file */
/* stops if current line contains numbers before any '!' */
void xplor_skip_to_number(FILE **fin, char **nextline)
{
  int i, done, foundletter;

  for (done = 0; done == 0;) { /* read next line */
    if (fgets(*nextline, FLENGTH, *fin) == NULL) {
      error_EOF(70510, "lib_vio");
    }
    for (i = 0; i < FLENGTH; ++i) 
      if ((*(*nextline + i) == '!') || (*(*nextline + i) == '\n')) {
        *(*nextline + i) = '\0';
        break;
      }
    foundletter = 0;
    for (i = 0; * (*nextline + i) != '\0'; ++i) { /* check if it contains ABC FGHIJKLMNOPQRSTUVWXYZ (or lower case) */
      if (*(*nextline + i) > 64 && *(*nextline + i) < 68) {
        foundletter = 1;
        break;
      }
      if (*(*nextline + i) > 69 && *(*nextline + i) < 91) {
        foundletter = 1;
        break;
      }
      if (*(*nextline + i) > 96 && *(*nextline + i) < 100) {
        foundletter = 1;
        break;
      }
      if (*(*nextline + i) > 101 && *(*nextline + i) < 123) {
        foundletter = 1;
        break;
      }
    }
    if (foundletter == 0) for (i = 0; * (*nextline + i) != '\0'; ++i)
        if (*(*nextline + i) >= '0' && *(*nextline + i) <= '9') {
          done = 1;
          break;
        }
  }
}


/* reads MRC or CCP4 binary file and swaps bytes automatically, also automatically detects MODE 0 signed-ness */
void read_mrc(char *vol_file, int *orom, int *cubic, int *ordermode,
              unsigned *nc, unsigned *nr, unsigned *ns,
              int *ncstart, int *nrstart, int *nsstart,
              double *widthx, double *widthy, double *widthz,
              double *xorigin, double *yorigin, double *zorigin,
              double *alpha, double *beta, double *gamma,
              double **fphi)
{

  unsigned long count, nvox;
  FILE *fin;
  int nc_tmp, nr_tmp, ns_tmp, mx, my, mz;
  int mode;
  float a_tmp, b_tmp, g_tmp;
  float x_tmp, y_tmp, z_tmp;
  int testa, testb, testg;
  float xlen, ylen, zlen;
  int i, swap, header_ok = 1;
  int mapc, mapr, maps;
  float dmin, dmax, dmean, dummy, currfloat, cfunsigned, prevfloat = 0.0f, pfunsigned = 0.0f, rms;
  double totdiffsigned = 0.0, totdiffunsigned = 0.0;
  int n_range_viol0, n_range_viol1;
  char mapchar1, mapchar2, mapchar3, mapchar4;
  char machstchar1, machstchar2, machstchar3, machstchar4;
  int ispg, nsymbt, lskflg;
  float skwmat11, skwmat12, skwmat13, skwmat21, skwmat22, skwmat23, skwmat31, skwmat32, skwmat33, skwtrn1, skwtrn2, skwtrn3;

  n_range_viol0 = test_mrc(vol_file, 0);
  n_range_viol1 = test_mrc(vol_file, 1);

  if (n_range_viol0 < n_range_viol1) { /* guess endianism */
    swap = 0;
    if (n_range_viol0 > 0) {
      printf("lib_vio> Warning: %i header field range violations detected in file %s \n", n_range_viol0, vol_file);
    }
  } else {
    swap = 1;
    if (n_range_viol1 > 0) {
      printf("lib_vio> Warning: %i header field range violations detected in file %s \n", n_range_viol1, vol_file);
    }
  }

  /* read header */
  fin = fopen(vol_file, "rb");
  if (fin == NULL) {
    error_open_filename(70620, "lib_vio", vol_file);
  }
  printf("lib_vio> Reading header information from MRC or CCP4 file %s \n", vol_file);
  header_ok *= read_int(&nc_tmp, fin, swap);
  header_ok *= read_int(&nr_tmp, fin, swap);
  header_ok *= read_int(&ns_tmp, fin, swap);
  *nc = nc_tmp;
  *nr = nr_tmp;
  *ns = ns_tmp;
  header_ok *= read_int(&mode, fin, swap);
  header_ok *= read_int(ncstart, fin, swap);
  header_ok *= read_int(nrstart, fin, swap);
  header_ok *= read_int(nsstart, fin, swap);
  header_ok *= read_int(&mx, fin, swap);
  header_ok *= read_int(&my, fin, swap);
  header_ok *= read_int(&mz, fin, swap);
  header_ok *= read_float(&xlen, fin, swap);
  header_ok *= read_float(&ylen, fin, swap);
  header_ok *= read_float(&zlen, fin, swap);
  header_ok *= read_float(&a_tmp, fin, swap);
  header_ok *= read_float(&b_tmp, fin, swap);
  header_ok *= read_float(&g_tmp, fin, swap);
  *alpha = a_tmp;
  *beta = b_tmp;
  *gamma = g_tmp;
  header_ok *= read_int(&mapc, fin, swap);
  header_ok *= read_int(&mapr, fin, swap);
  header_ok *= read_int(&maps, fin, swap);
  header_ok *= read_float(&dmin, fin, swap);
  header_ok *= read_float(&dmax, fin, swap);
  header_ok *= read_float(&dmean, fin, swap);
  header_ok *= read_int(&ispg, fin, swap);
  header_ok *= read_int(&nsymbt, fin, swap);
  header_ok *= read_int(&lskflg, fin, swap);
  header_ok *= read_float(&skwmat11, fin, swap);
  header_ok *= read_float(&skwmat12, fin, swap);
  header_ok *= read_float(&skwmat13, fin, swap);
  header_ok *= read_float(&skwmat21, fin, swap);
  header_ok *= read_float(&skwmat22, fin, swap);
  header_ok *= read_float(&skwmat23, fin, swap);
  header_ok *= read_float(&skwmat31, fin, swap);
  header_ok *= read_float(&skwmat32, fin, swap);
  header_ok *= read_float(&skwmat33, fin, swap);
  header_ok *= read_float(&skwtrn1, fin, swap);
  header_ok *= read_float(&skwtrn2, fin, swap);
  header_ok *= read_float(&skwtrn3, fin, swap);
  for (i = 38; i < 50; ++i) header_ok *= read_float(&dummy, fin, swap);
  header_ok *= read_float(&x_tmp, fin, swap);
  header_ok *= read_float(&y_tmp, fin, swap);
  header_ok *= read_float(&z_tmp, fin, swap);
  *xorigin = x_tmp;
  *yorigin = y_tmp;
  *zorigin = z_tmp;
  header_ok *= read_char(&mapchar1, fin);
  header_ok *= read_char(&mapchar2, fin);
  header_ok *= read_char(&mapchar3, fin);
  header_ok *= read_char(&mapchar4, fin);
  header_ok *= read_char(&machstchar1, fin);
  header_ok *= read_char(&machstchar2, fin);
  header_ok *= read_char(&machstchar3, fin);
  header_ok *= read_char(&machstchar4, fin);
  header_ok *= read_float(&rms, fin, swap);

  if (header_ok == 0) {
    error_file_header(70650, "lib_vio", vol_file);
  }

  /* print some info */
  printf("lib_vio>       NC = %8d  (# columns)\n", *nc);
  printf("lib_vio>       NR = %8d  (# rows)\n", *nr);
  printf("lib_vio>       NS = %8d  (# sections)\n", *ns);
  printf("lib_vio>     MODE = %8d  (data type: 0: 8-bit, 1: 16-bit; 2: 32-bit)\n", mode);
  printf("lib_vio>  NCSTART = %8d  (index of first column, counting from 0)\n", *ncstart);
  printf("lib_vio>  NRSTART = %8d  (index of first row, counting from 0)\n", *nrstart);
  printf("lib_vio>  NSSTART = %8d  (index of first section, counting from 0)\n", *nsstart);
  printf("lib_vio>       MX = %8d  (# of X intervals in unit cell)\n", mx);
  printf("lib_vio>       MY = %8d  (# of Y intervals in unit cell)\n", my);
  printf("lib_vio>       MZ = %8d  (# of Z intervals in unit cell)\n", mz);
  printf("lib_vio> X length = %8.3f  (unit cell dimension)\n", xlen);
  printf("lib_vio> Y length = %8.3f  (unit cell dimension)\n", ylen);
  printf("lib_vio> Z length = %8.3f  (unit cell dimension)\n", zlen);
  printf("lib_vio>    Alpha = %8.3f  (unit cell angle)\n", *alpha);
  printf("lib_vio>     Beta = %8.3f  (unit cell angle)\n", *beta);
  printf("lib_vio>    Gamma = %8.3f  (unit cell angle)\n", *gamma);
  printf("lib_vio>     MAPC = %8d  (columns axis: 1=X,2=Y,3=Z)\n", mapc);
  printf("lib_vio>     MAPR = %8d  (rows axis: 1=X,2=Y,3=Z)\n", mapr);
  printf("lib_vio>     MAPS = %8d  (sections axis: 1=X,2=Y,3=Z)\n", maps);
  printf("lib_vio>     DMIN = %8.3f  (minimum density value - ignored)\n", dmin);
  printf("lib_vio>     DMAX = %8.3f  (maximum density value - ignored)\n", dmax);
  printf("lib_vio>    DMEAN = %8.3f  (mean density value - ignored)\n", dmean);
  printf("lib_vio>     ISPG = %8d  (space group number - ignored)\n", ispg);
  printf("lib_vio>   NSYMBT = %8d  (# bytes storing symmetry operators)\n", nsymbt);
  printf("lib_vio>   LSKFLG = %8d  (skew matrix flag: 0:none, 1:follows)\n", lskflg);
  if (lskflg != 0) {
    printf("lib_vio> Warning: Will compute skew parameters from the above header info.\n");
    printf("lib_vio> The following skew parameters in the header will be ignored:\n");
    printf("lib_vio>      S11 = %8.3f  (skew matrix element 11)\n", skwmat11);
    printf("lib_vio>      S12 = %8.3f  (skew matrix element 12)\n", skwmat12);
    printf("lib_vio>      S13 = %8.3f  (skew matrix element 13)\n", skwmat13);
    printf("lib_vio>      S21 = %8.3f  (skew matrix element 21)\n", skwmat21);
    printf("lib_vio>      S22 = %8.3f  (skew matrix element 22)\n", skwmat22);
    printf("lib_vio>      S23 = %8.3f  (skew matrix element 23)\n", skwmat23);
    printf("lib_vio>      S31 = %8.3f  (skew matrix element 31)\n", skwmat31);
    printf("lib_vio>      S32 = %8.3f  (skew matrix element 32)\n", skwmat32);
    printf("lib_vio>      S33 = %8.3f  (skew matrix element 33)\n", skwmat33);
    printf("lib_vio>       T1 = %8.3f  (skew translation element 1)\n", skwtrn1);
    printf("lib_vio>       T2 = %8.3f  (skew translation element 2)\n", skwtrn2);
    printf("lib_vio>       T3 = %8.3f  (skew translation element 3)\n", skwtrn3);
    printf("lib_vio> End of skew parameters warning.\n");
  }
  printf("lib_vio>  XORIGIN = %8.3f  (X origin - MRC2000 only)\n", *xorigin);
  printf("lib_vio>  YORIGIN = %8.3f  (Y origin - MRC2000 only)\n", *yorigin);
  printf("lib_vio>  ZORIGIN = %8.3f  (Z origin - MRC2000 only)\n", *zorigin);
  printf("lib_vio>      MAP =      %c%c%c%c (map string)\n", mapchar1, mapchar2, mapchar3, mapchar4);
  printf("lib_vio>   MACHST = %d %d %d %d (machine stamp - ignored)\n", machstchar1, machstchar2, machstchar3, machstchar4);
  printf("lib_vio>      RMS = %8.3f  (density rms deviation -ignored)\n", rms);
  if (mapchar1 != 'M' || mapchar2 != 'A' || mapchar3 != 'P') {
    printf("lib_vio> Warning: MAP string not detected. It appears this is an older (pre-2000) or unconventional MRC format.\n");
    printf("lib_vio> If in doubt use the map2map tool in manual mode and/or inspect the results.\n");
  }

  /* extract data based on file type mode, currently supports 8-bit, 16-bit, and 32-bit */
  /* note: fphi will contain data in the order of the input file, no axis permutation yet */
  nvox = *nc * *nr * *ns;
  switch (mode) {
    case 0: /* char - converted to float, testing for signed-ness*/
      rewind(fin);
      do_vect(fphi, nvox);
      for (count = 0; count < 256; ++count) if (read_float_empty(fin) == 0) error_file_convert(70642, "lib_vio", vol_file);
      for (count = 0; count < (unsigned long)nsymbt; ++count) if (read_char_float(&currfloat, fin) == 0) error_file_convert(70643, "lib_vio", vol_file);
      for (count = 0; count < nvox; ++count) {
        if (read_char_float(&currfloat, fin) == 0) error_file_convert(70651, "lib_vio", vol_file);
        else {
          *(*fphi + count) = currfloat;
          if (currfloat < 0) cfunsigned = currfloat + 256.0;
          else cfunsigned = currfloat;
          totdiffsigned += fabs(currfloat - prevfloat);
          totdiffunsigned += fabs(cfunsigned - pfunsigned);
          prevfloat = currfloat;
          pfunsigned = cfunsigned;
        }
      }
      fclose(fin);
      if (totdiffsigned > totdiffunsigned) {
        for (count = 0; count < nvox; count++) if (*(*fphi + count) < 0) *(*fphi + count) += 256.0;
        printf("lib_vio> Warning: It appears the MODE 0 data type uses the unsigned 8-bit convention, adding 256 to negative values.\n");
        printf("lib_vio> If in doubt use the map2map tool in manual mode and/or inspect the results.\n");
      }
      break;
    case 1: /* 16-bit float */
      rewind(fin);
      do_vect(fphi, nvox);
      for (count = 0; count < 256; ++count) if (read_float_empty(fin) == 0) {
          error_file_convert(70644, "lib_vio", vol_file);
        }
      for (count = 0; count < (unsigned long)nsymbt; ++count) if (read_char_float(&currfloat, fin) == 0) {
          error_file_convert(70645, "lib_vio", vol_file);
        }

      for (count = 0; count < nvox; ++count) {
        if (read_short_float(&currfloat, fin, swap) == 0) {
          error_file_convert(70652, "lib_vio", vol_file);
        } else {
          *(*fphi + count) = currfloat;
        }
      }
      fclose(fin);
      break;
    case 2: /* 32-bit float */
      rewind(fin);
      do_vect(fphi, nvox);
      for (count = 0; count < 256; ++count) if (read_float_empty(fin) == 0) {
          error_file_convert(70646, "lib_vio", vol_file);
        }
      for (count = 0; count < (unsigned long)nsymbt; ++count) if (read_char_float(&currfloat, fin) == 0) {
          error_file_convert(70647, "lib_vio", vol_file);
        }
      for (count = 0; count < nvox; ++count) {
        if (read_float(&currfloat, fin, swap) == 0) {
          error_file_convert(70653, "lib_vio", vol_file);
        } else {
          *(*fphi + count) = currfloat;
        }
      }
      fclose(fin);
      break;
    default:
      error_file_float_mode(70654, "lib_vio", vol_file);
  }

  /* assign voxel spacing parameters */
  *widthx = xlen / (double) mx;
  *widthy = ylen / (double) my;
  *widthz = zlen / (double) mz;

  /* test for orthogonal and cubic lattice */
  testa = (int)floor(100 * *alpha + 0.5);
  testb = (int)floor(100 * *beta + 0.5);
  testg = (int)floor(100 * *gamma + 0.5);
  if (testa != 9000 || testb != 9000 || testg != 9000) *orom = 0;
  if (*orom == 0 || floor((*widthx - *widthy) * 1000 + 0.5) != 0 || floor((*widthy - *widthz) * 1000 + 0.5) != 0 || floor((*widthx - *widthz) * 1000 + 0.5) != 0) *cubic = 0;

  /* set axis permutation mode */
  *ordermode = 7;
  if (mapc == 1 && mapr == 2 && maps == 3) {
    *ordermode = 1;
  }
  if (mapc == 1 && mapr == 3 && maps == 2) {
    *ordermode = 2;
  }
  if (mapc == 2 && mapr == 1 && maps == 3) {
    *ordermode = 3;
  }
  if (mapc == 2 && mapr == 3 && maps == 1) {
    *ordermode = 4;
  }
  if (mapc == 3 && mapr == 1 && maps == 2) {
    *ordermode = 5;
  }
  if (mapc == 3 && mapr == 2 && maps == 1) {
    *ordermode = 6;
  }
  if (*ordermode == 7) {
    error_axis_assignment(70680, "lib_vio");
  }
  printf("lib_vio> Volumetric data read from file %s\n", vol_file);
}


/* reads SPIDER binary file and swaps bytes automatically */
void read_spider(char *vol_file, unsigned *extx, unsigned *exty, unsigned *extz, double **fphi)
{
  unsigned long nvox;
  FILE *fin;
  int i, swap, header_ok = 1, headlen;
  unsigned long count;
  float dummy, currfloat;
  float nslice, nrow, iform, imami, fmax, fmin, av, sig, nsam, headrec;
  float iangle, phi, theta, gamma, xoff, yoff, zoff, scale, labbyt, lenbyt;
  float istack, inuse, maxim, kangle, phi1, theta1, psi1, phi2, theta2, psi2;
  int n_range_viol0, n_range_viol1;

  n_range_viol0 = test_spider(vol_file, 0);
  n_range_viol1 = test_spider(vol_file, 1);

  if (n_range_viol0 < n_range_viol1) { /* guess endianism */
    swap = 0;
    if (n_range_viol0 > 0) {
      printf("lib_vio> Warning: %i header field range violations detected \n", n_range_viol0);
    }
  } else {
    swap = 1;
    if (n_range_viol1 > 0) {
      printf("lib_vio> Warning: %i header field range violations detected \n", n_range_viol1);
    }
  }

  /* read header */
  fin = fopen(vol_file, "rb");
  if (fin == NULL) {
    error_open_filename(70820, "lib_vio", vol_file);
  }
  printf("lib_vio> Reading header information from SPIDER file %s \n", vol_file);
  header_ok *= read_float(&nslice, fin, swap);
  header_ok *= read_float(&nrow, fin, swap);
  header_ok *= read_float(&dummy, fin, swap);
  header_ok *= read_float(&dummy, fin, swap);
  header_ok *= read_float(&iform, fin, swap);
  header_ok *= read_float(&imami, fin, swap);
  header_ok *= read_float(&fmax, fin, swap);
  header_ok *= read_float(&fmin, fin, swap);
  header_ok *= read_float(&av, fin, swap);
  header_ok *= read_float(&sig, fin, swap);
  header_ok *= read_float(&dummy, fin, swap);
  header_ok *= read_float(&nsam, fin, swap);
  header_ok *= read_float(&headrec, fin, swap);
  header_ok *= read_float(&iangle, fin, swap);
  header_ok *= read_float(&phi, fin, swap);
  header_ok *= read_float(&theta, fin, swap);
  header_ok *= read_float(&gamma, fin, swap);
  header_ok *= read_float(&xoff, fin, swap);
  header_ok *= read_float(&yoff, fin, swap);
  header_ok *= read_float(&zoff, fin, swap);
  header_ok *= read_float(&scale, fin, swap);
  header_ok *= read_float(&labbyt, fin, swap);
  header_ok *= read_float(&lenbyt, fin, swap);
  header_ok *= read_float(&istack, fin, swap);
  header_ok *= read_float(&inuse, fin, swap);
  header_ok *= read_float(&maxim, fin, swap);
  for (i = 0; i < 4; ++i) header_ok *= read_float(&dummy, fin, swap);
  header_ok *= read_float(&kangle, fin, swap);
  header_ok *= read_float(&phi1, fin, swap);
  header_ok *= read_float(&theta1, fin, swap);
  header_ok *= read_float(&psi1, fin, swap);
  header_ok *= read_float(&phi2, fin, swap);
  header_ok *= read_float(&theta2, fin, swap);
  header_ok *= read_float(&psi2, fin, swap);
  if (header_ok == 0) {
    error_file_header(70850, "lib_vio", vol_file);
  }

  /* print some info */
  printf("lib_vio>   NSLICE = %8.f  (# sections)\n", nslice);
  printf("lib_vio>     NROW = %8.f  (# rows)\n", nrow);
  printf("lib_vio>    IFORM = %8.f  (file type specifier - ignored)\n", iform);
  printf("lib_vio>    IMAMI = %8.f  (flag: 1 = the maximum and minimum are computed - ignored)\n", imami);
  printf("lib_vio>     FMAX = %8.3f  (maximum density value - ignored)\n", fmax);
  printf("lib_vio>     FMIN = %8.3f  (minimum density value - ignored)\n", fmin);
  printf("lib_vio>       AV = %8.3f  (average density value - ignored)\n", av);
  printf("lib_vio>      SIG = %8.3f  (density rms deviation - ignored)\n", sig);
  printf("lib_vio>     NSAM = %8.f  (# columns)\n", nsam);
  printf("lib_vio>  HEADREC = %8.f  (number of records in file header)\n", headrec);
  printf("lib_vio>   IANGLE = %8.f  (flag: 1 = tilt angles filled - ignored)\n", iangle);
  printf("lib_vio>      PHI = %8.3f  (tilt angle - ignored)\n", phi);
  printf("lib_vio>    THETA = %8.3f  (tilt angle - ignored)\n", theta);
  printf("lib_vio>    GAMMA = %8.3f  (tilt angle - ignored)\n", gamma);
  printf("lib_vio>     XOFF = %8.3f  (X offset - ignored)\n", xoff);
  printf("lib_vio>     YOFF = %8.3f  (Y offset - ignored)\n", yoff);
  printf("lib_vio>     ZOFF = %8.3f  (Z offset - ignored)\n", zoff);
  printf("lib_vio>    SCALE = %8.3f  (scale factor - ignored)\n", scale);
  printf("lib_vio>   LABBYT = %8.f  (total number of bytes in header)\n", labbyt);
  printf("lib_vio>   LENBYT = %8.f  (record length in bytes)\n", lenbyt);
  printf("lib_vio>   ISTACK = %8.f  (flag; file contains a stack of images - ignored)\n", istack);
  printf("lib_vio>    INUSE = %8.f  (flag; this image in stack is used - ignored)\n", inuse);
  printf("lib_vio>    MAXIM = %8.f  (maximum image used in stack - ignored)\n", maxim);
  printf("lib_vio>   KANGLE = %8.f  (flag: additional angles set - ignored)\n", kangle);
  printf("lib_vio>     PHI1 = %8.3f  (additional rotation - ignored)\n", phi1);
  printf("lib_vio>   THETA1 = %8.3f  (additional rotation - ignored)\n", theta1);
  printf("lib_vio>     PSI1 = %8.3f  (additional rotation - ignored)\n", psi1);
  printf("lib_vio>     PHI2 = %8.3f  (additional rotation - ignored)\n", phi2);
  printf("lib_vio>   THETA2 = %8.3f  (additional rotation - ignored)\n", theta2);
  printf("lib_vio>     PSI2 = %8.3f  (additional rotation - ignored)\n", psi2);

  *extx = (unsigned int)nsam;
  *exty = (unsigned int)nrow;
  *extz = (unsigned int)nslice;
  nvox = *extx * *exty * *extz;
  headlen = *extx * (int)ceil(256 / (*extx * 1.0));
  do_vect(fphi, nvox);
  rewind(fin);
  for (count = 0; count < (unsigned long)headlen; ++count) if (read_float_empty(fin) == 0) {
      error_file_convert(70841, "lib_vio", vol_file);
    }
  for (count = 0; count < nvox; ++count) if (read_float(&currfloat, fin, swap) == 0) {
      error_file_convert(70842, "lib_vio", vol_file);
    } else {
      *(*fphi + count) = currfloat;
    }
  fclose(fin);

  /* notes and error checks */
  if (((int) floor(100 * (sizeof(float)*headlen - labbyt) + 0.5)) != 0) {
    error_spider_header(70860, "lib_vio");
  }
  printf("lib_vio> Volumetric data read from file %s\n", vol_file);

}


/* ASCII dumps binary file, flips bytes if swap==1 */
void dump_binary_and_exit(char *in_file, char *out_file, int swap)
{
  FILE *fin, *fout;
  float *phi;
  unsigned long nfloat;
  unsigned long count;

  fin = fopen(in_file, "rb");
  if (fin == NULL) {
    error_open_filename(70910, "lib_vio", in_file);
  }

  nfloat = count_floats(&fin);

  phi = (float *) alloc_vect(nfloat, sizeof(float));

  for (count = 0; count < nfloat; count++)
    read_float(phi + count, fin, swap);

  printf("lib_vio> Binary data read as 32-bit 'float' type from file %s\n", in_file);
  if (swap) printf("lib_vio> The byte order (endianism) has been swapped.\n");
  fclose(fin);

  fout = fopen(out_file, "w");
  if (fout == NULL) {
    error_open_filename(70940, "lib_vio", out_file);
  }

  printf("lib_vio> Writing ASCII data... \n");
  for (count = 0; count < nfloat; count++) {
    if ((count + 1) % 10 == 0) fprintf(fout, " %10.6f \n", *(phi + count));
    else fprintf(fout, " %10.6f ", *(phi + count));
  }
  fclose(fout);

  printf("lib_vio> ASCII dump of binary file %s written to file %s. \n", in_file, out_file);
  printf("lib_vio> Open / check ASCII file with text editor and extract map densities. \n");
  printf("lib_vio> Then convert with map2map tool, option 1: ASCII.\n");
  exit(1);
}


/* reads 4-byte float and swaps bytes if swap==1 */
int read_float(float *currfloat, FILE *fin, int swap)
{
  unsigned char *cptr, tmp;

  if (fread(currfloat, 4, 1, fin) != 1) return 0;
  if (swap == 1) {
    cptr = (unsigned char *)currfloat;
    tmp = cptr[0];
    cptr[0] = cptr[3];
    cptr[3] = tmp;
    tmp = cptr[1];
    cptr[1] = cptr[2];
    cptr[2] = tmp;
  }
  return 1;
}


/* reads 2-byte float and swaps bytes if swap==1 */
int read_short_float(float *currfloat, FILE *fin, int swap)
{
  unsigned char *cptr, tmp;
  short currshort;

  if (fread(&currshort, 2, 1, fin) != 1) return 0;
  if (swap == 1) {
    cptr = (unsigned char *)&currshort;
    tmp = cptr[0];
    cptr[0] = cptr[1];
    cptr[1] = tmp;
  }
  *currfloat = (float)currshort;
  return 1;
}


/* reads header float and does nothing */
int read_float_empty(FILE *fin)
{
  float currfloat;

  if (fread(&currfloat, 4, 1, fin) != 1) return 0;
  return 1;
}

/* reads char and assigns to float */
int read_char_float(float *currfloat, FILE *fin)
{
  char currchar;

  if (fread(&currchar, 1, 1, fin) != 1) return 0;
  *currfloat = (float)currchar;
  return 1;
}


/* reads char and assigns to char */
int read_char(char *currchar, FILE *fin)
{

  if (fread(currchar, 1, 1, fin) != 1) return 0;
  return 1;
}


/* reads int and swaps bytes if swap==1 */
int read_int(int *currlong, FILE *fin, int swap)
{
  unsigned char *cptr, tmp;

  if (fread(currlong, 4, 1, fin) != 1) return 0;
  if (swap == 1) {
    cptr = (unsigned char *)currlong;
    tmp = cptr[0];
    cptr[0] = cptr[3];
    cptr[3] = tmp;
    tmp = cptr[1];
    cptr[1] = cptr[2];
    cptr[2] = tmp;
  }
  return 1;
}


/* counts number of 4-byte floats of open file */
unsigned long count_floats(FILE **fin)
{
  unsigned long finl = 0;

  rewind(*fin);
  for (; ;) {
    if (fgetc(*fin) == EOF) break;
    ++finl;
  }
  rewind(*fin);
  return finl / 4;
}


/* checks for registration of grid with origin of coordinate system */
int test_registration(float origx, float origy, float origz, float width)
{
  float xreg, xreg1, yreg, yreg1, zreg, zreg1;

  xreg = fabs(fmod(origx + 0.00001 * width, width));
  xreg1 = fabs(fmod(origx - 0.00001 * width, width));
  yreg = fabs(fmod(origy + 0.00001 * width, width));
  yreg1 = fabs(fmod(origy - 0.00001 * width, width));
  zreg = fabs(fmod(origz + 0.00001 * width, width));
  zreg1 = fabs(fmod(origz - 0.00001 * width, width));
  if (xreg1 < xreg) xreg = xreg1;
  if (yreg1 < yreg) yreg = yreg1;
  if (zreg1 < zreg) zreg = zreg1;
  if (xreg + yreg + zreg > 0.0001 * width) return 0;
  else return 1;
}


/* tests Situs header and returns (double) a number of range violations */
/* weight factor 2 takes into account that fewer header params in Situs file */
int test_situs(char *vol_file)
{

  FILE *fin;
  double width, origx, origy, origz;
  unsigned extx, exty, extz;
  int header_ok = 1, n_range_viols = 0;

  fin = fopen(vol_file, "r");
  if (fin == NULL) {
    error_open_filename(71010, "lib_vio", vol_file);
  }

  /* read header */
  if (fscanf(fin, "%le", &width) != 1) header_ok *= 0;
  if (fscanf(fin, "%le", &origx) != 1) header_ok *= 0;
  if (fscanf(fin, "%le", &origy) != 1) header_ok *= 0;
  if (fscanf(fin, "%le", &origz) != 1) header_ok *= 0;
  if (fscanf(fin, "%d", &extx) != 1) header_ok *= 0;
  if (fscanf(fin, "%d", &exty) != 1) header_ok *= 0;
  if (fscanf(fin, "%d", &extz) != 1) header_ok *= 0;
  fclose(fin);

  if (header_ok == 0) return 999999; /* worst case, can't read data */
  else {
    n_range_viols += (extx > 5000);
    n_range_viols += (extx < 0);
    n_range_viols += (exty > 5000);
    n_range_viols += (exty < 0);
    n_range_viols += (extz > 5000);
    n_range_viols += (extz < 0);
    n_range_viols += (width > 2000);
    n_range_viols += (width < 0);
    n_range_viols += (origx > 1e6);
    n_range_viols += (origx < -1e6);
    n_range_viols += (origy > 1e6);
    n_range_viols += (origy < -1e6);
    n_range_viols += (origz > 1e6);
    n_range_viols += (origz < -1e6);
    return 2 * n_range_viols;
  }
}

/* checks if vol_file has Situs file name suffix */
int have_situs_suffix(char *vol_file)
{

  int vl;
  int situs_format = 0;

  vl = strlen(vol_file);
  if (vl > 4) {
    if (strstr(vol_file + (vl - 4), ".sit") != NULL) situs_format = 1;
    if (strstr(vol_file + (vl - 4), ".SIT") != NULL) situs_format = 1;
  }
  if (vl > 6) {
    if (strstr(vol_file + (vl - 6), ".situs") != NULL) situs_format = 1;
    if (strstr(vol_file + (vl - 6), ".SITUS") != NULL) situs_format = 1;
  }
  return situs_format;
}


/* tests Situs file name and returns (double) of a number of range violations */
int test_situs_header_and_suffix(char *vol_file)
{

  if (have_situs_suffix(vol_file) == 0) return 999999;
  else return test_situs(vol_file);
}


/* tests MRC / CCP4 header and returns a (non-exhaustive) number of range violations */
int test_mrc(char *vol_file, int swap)
{
  FILE *fin;
  int nc, nr, ns, mx, my, mz;
  int mode, ncstart, nrstart, nsstart;
  float xlen, ylen, zlen;
  int i, header_ok = 1, n_range_viols = 0;
  int mapc, mapr, maps;
  float alpha, beta, gamma;
  float dmin, dmax, dmean, dummy, xorigin, yorigin, zorigin;

  fin = fopen(vol_file, "rb");
  if (fin == NULL) {
    error_open_filename(71010, "lib_vio", vol_file);
  }

  /* read header info */
  header_ok *= read_int(&nc, fin, swap);
  header_ok *= read_int(&nr, fin, swap);
  header_ok *= read_int(&ns, fin, swap);
  header_ok *= read_int(&mode, fin, swap);
  header_ok *= read_int(&ncstart, fin, swap);
  header_ok *= read_int(&nrstart, fin, swap);
  header_ok *= read_int(&nsstart, fin, swap);
  header_ok *= read_int(&mx, fin, swap);
  header_ok *= read_int(&my, fin, swap);
  header_ok *= read_int(&mz, fin, swap);
  header_ok *= read_float(&xlen, fin, swap);
  header_ok *= read_float(&ylen, fin, swap);
  header_ok *= read_float(&zlen, fin, swap);
  header_ok *= read_float(&alpha, fin, swap);
  header_ok *= read_float(&beta, fin, swap);
  header_ok *= read_float(&gamma, fin, swap);
  header_ok *= read_int(&mapc, fin, swap);
  header_ok *= read_int(&mapr, fin, swap);
  header_ok *= read_int(&maps, fin, swap);
  header_ok *= read_float(&dmin, fin, swap);
  header_ok *= read_float(&dmax, fin, swap);
  header_ok *= read_float(&dmean, fin, swap);
  for (i = 23; i < 50; ++i) header_ok *= read_float(&dummy, fin, swap);
  header_ok *= read_float(&xorigin, fin, swap);
  header_ok *= read_float(&yorigin, fin, swap);
  header_ok *= read_float(&zorigin, fin, swap);
  fclose(fin);
  if (header_ok == 0) {
    error_file_header(71020, "lib_vio", vol_file);
  }

  n_range_viols += (nc > 5000);
  n_range_viols += (nc < 0);
  n_range_viols += (nr > 5000);
  n_range_viols += (nr < 0);
  n_range_viols += (ns > 5000);
  n_range_viols += (ns < 0);
  n_range_viols += (ncstart > 5000);
  n_range_viols += (ncstart < -5000);
  n_range_viols += (nrstart > 5000);
  n_range_viols += (nrstart < -5000);
  n_range_viols += (nsstart > 5000);
  n_range_viols += (nsstart < -5000);
  n_range_viols += (mx > 5000);
  n_range_viols += (mx < 0);
  n_range_viols += (my > 5000);
  n_range_viols += (my < 0);
  n_range_viols += (mz > 5000);
  n_range_viols += (mz < 0);
  n_range_viols += (alpha > 360.0f);
  n_range_viols += (alpha < -360.0f);
  n_range_viols += (beta > 360.0f);
  n_range_viols += (beta < -360.0f);
  n_range_viols += (gamma > 360.0f);
  n_range_viols += (gamma < -360.0f);

  return n_range_viols;
}


/* tests SPIDER header and returns double of a (non-exhaustive) number of range violations */
/* weight factor 2 takes into account that fewer header params in SPIDER file */
int test_spider(char *vol_file, int swap)
{
  FILE *fin;
  int i, header_ok = 1, n_range_viols = 0, headlen;
  float dummy;
  float nslice, nrow, iform, imami, fmax, fmin, av, sig, nsam, headrec;
  float iangle, phi, theta, gamma, xoff, yoff, zoff, scale, labbyt, lenbyt;
  float istack, inuse, maxim, kangle, phi1, theta1, psi1, phi2, theta2, psi2;

  fin = fopen(vol_file, "rb");
  if (fin == NULL) {
    error_open_filename(71210, "lib_vio", vol_file);
  }

  /* read header info */
  header_ok *= read_float(&nslice, fin, swap);
  header_ok *= read_float(&nrow, fin, swap);
  header_ok *= read_float(&dummy, fin, swap);
  header_ok *= read_float(&dummy, fin, swap);
  header_ok *= read_float(&iform, fin, swap);
  header_ok *= read_float(&imami, fin, swap);
  header_ok *= read_float(&fmax, fin, swap);
  header_ok *= read_float(&fmin, fin, swap);
  header_ok *= read_float(&av, fin, swap);
  header_ok *= read_float(&sig, fin, swap);
  header_ok *= read_float(&dummy, fin, swap);
  header_ok *= read_float(&nsam, fin, swap);
  header_ok *= read_float(&headrec, fin, swap);
  header_ok *= read_float(&iangle, fin, swap);
  header_ok *= read_float(&phi, fin, swap);
  header_ok *= read_float(&theta, fin, swap);
  header_ok *= read_float(&gamma, fin, swap);
  header_ok *= read_float(&xoff, fin, swap);
  header_ok *= read_float(&yoff, fin, swap);
  header_ok *= read_float(&zoff, fin, swap);
  header_ok *= read_float(&scale, fin, swap);
  header_ok *= read_float(&labbyt, fin, swap);
  header_ok *= read_float(&lenbyt, fin, swap);
  header_ok *= read_float(&istack, fin, swap);
  header_ok *= read_float(&inuse, fin, swap);
  header_ok *= read_float(&maxim, fin, swap);
  for (i = 0; i < 4; ++i) header_ok *= read_float(&dummy, fin, swap);
  header_ok *= read_float(&kangle, fin, swap);
  header_ok *= read_float(&phi1, fin, swap);
  header_ok *= read_float(&theta1, fin, swap);
  header_ok *= read_float(&psi1, fin, swap);
  header_ok *= read_float(&phi2, fin, swap);
  header_ok *= read_float(&theta2, fin, swap);
  header_ok *= read_float(&psi2, fin, swap);
  fclose(fin);
  if (header_ok == 0) {
    error_file_header(71220, "lib_vio", vol_file);
  }
  headlen = (int)(nsam * ceil(256 / (nsam * 1.0)));
  n_range_viols += (((int) floor(100 * (4 * headlen - labbyt) + 0.5)) != 0);
  n_range_viols += (headrec > 1e10);
  n_range_viols += (headrec <= 1e-10);
  n_range_viols += (labbyt > 1e10);
  n_range_viols += (labbyt <= 1e-10);
  n_range_viols += (lenbyt > 1e10);
  n_range_viols += (lenbyt <= 1e-10);
  n_range_viols += (nslice > 1e10);
  n_range_viols += (nslice < 1e-10);
  n_range_viols += (nrow > 1e10);
  n_range_viols += (nrow < 1e-10);
  n_range_viols += (nsam > 1e10);
  n_range_viols += (nsam < 1e-10);
  return 2 * n_range_viols;
}


/* returns the permuted index */
/* nc, nr, ns, are the unperturbed dimensions (# colums, # rows, # sections) */
/* computes first from 'count' the indices ic, ir, is, assumed to correspond to original, unpermuted input map */
/* returns new index m = ix + iy * nx + iz * nx * ny, where the x,y,z mapping is done (implicitly) as in permute_dimensions */
/* new(m) = old(count) will create the permuted (x,y,z) map */
unsigned long permuted_index(int ordermode, unsigned long count, unsigned nc, 
                             unsigned nr, unsigned ns)
{
  unsigned ic, ir, is;
  unsigned long ncr, q;

  ncr = nc * nr;
  is = count / ncr;
  q = count - is * ncr;
  ir = q / nc;
  ic = q - ir * nc;

  switch (ordermode) {
    case 1:
      return ic + ir * nc + is * nc * nr;
    case 2:
      return ic + is * nc + ir * nc * ns;
    case 3:
      return ir + ic * nr + is * nr * nc;
    case 4:
      return is + ic * ns + ir * ns * nc;
    case 5:
      return ir + is * nr + ic * nr * ns;
    case 6:
      return is + ir * ns + ic * ns * nr;
    default:
      error_option(70210, "lib_vio");
      return 0;
  }
}


/* create permuted map and reorder dimensions */
void permute_map(int ordermode, unsigned nc, unsigned nr, unsigned ns, 
                 unsigned *nx, unsigned *ny, unsigned *nz, int ncstart, 
                 int nrstart, int nsstart, int *nxstart, int *nystart, 
                 int *nzstart, double *phi, double **pphi)
{

  unsigned long nvox, count;

  /* create permuted map pphi */
  nvox = nc * nr * ns;
  do_vect(pphi, nvox);
  for (count = 0; count < nvox; count++) 
    *(*pphi + permuted_index(ordermode, count, nc, nr, ns)) = *(phi + count);
  free_vect_and_zero_ptr((void**)&phi);

  /* permute the map dimensions */
  permute_dimensions(ordermode, nc, nr, ns, nx, ny, nz, ncstart, nrstart, 
                     nsstart, nxstart, nystart, nzstart);
}


/* create permuted dimensions */
void permute_dimensions(int ordermode, unsigned nc, unsigned nr, unsigned ns, 
                        unsigned *nx, unsigned *ny, unsigned *nz, int ncstart, 
                        int nrstart, int nsstart, int *nxstart, int *nystart, 
                        int *nzstart)
{

  switch (ordermode) {
    case 1:
      *nx = nc;
      *ny = nr;
      *nz = ns;
      *nxstart = ncstart;
      *nystart = nrstart;
      *nzstart = nsstart;
      break;
    case 2:
      *nx = nc;
      *ny = ns;
      *nz = nr;
      *nxstart = ncstart;
      *nystart = nsstart;
      *nzstart = nrstart;
      break;
    case 3:
      *nx = nr;
      *ny = nc;
      *nz = ns;
      *nxstart = nrstart;
      *nystart = ncstart;
      *nzstart = nsstart;
      break;
    case 4:
      *nx = ns;
      *ny = nc;
      *nz = nr;
      *nxstart = nsstart;
      *nystart = ncstart;
      *nzstart = nrstart;
      break;
    case 5:
      *nx = nr;
      *ny = ns;
      *nz = nc;
      *nxstart = nrstart;
      *nystart = nsstart;
      *nzstart = ncstart;
      break;
    case 6:
      *nx = ns;
      *ny = nr;
      *nz = nc;
      *nxstart = nsstart;
      *nystart = nrstart;
      *nzstart = ncstart;
      break;
    default:
      error_option(70212, "lib_vio");
  }
}


/* print information on permuted dimensions */
void permute_print_info(int ordermode, unsigned nc, unsigned nr, unsigned ns, unsigned nx, unsigned ny, unsigned nz, int ncstart, int nrstart, int nsstart, int nxstart, int nystart, int nzstart)
{

  if (ordermode > 1) {
    printf("lib_vio> Map parameters BEFORE selected axis permutation: \n");
    printf("lib_vio>       NC = %8d  (# columns)\n", nc);
    printf("lib_vio>       NR = %8d  (# rows)\n", nr);
    printf("lib_vio>       NS = %8d  (# sections)\n", ns);
    printf("lib_vio>  NCSTART = %8d  (index of first column, counting from 0)\n", ncstart);
    printf("lib_vio>  NRSTART = %8d  (index of first row, counting from 0)\n", nrstart);
    printf("lib_vio>  NSSTART = %8d  (index of first section, counting from 0)\n", nsstart);
    printf("lib_vio> Map parameters AFTER selected axis permutation: \n");
    printf("lib_vio>       NX = %8d  (# X fields)\n", nx);
    printf("lib_vio>       NY = %8d  (# Y fields)\n", ny);
    printf("lib_vio>       NZ = %8d  (# Z fields)\n", nz);
    printf("lib_vio>  NXSTART = %8d  (X index of first voxel, counting from 0)\n", nxstart);
    printf("lib_vio>  NYSTART = %8d  (Y index of first voxel, counting from 0)\n", nystart);
    printf("lib_vio>  NZSTART = %8d  (Z index of first voxel, counting from 0)\n", nzstart);
  } else {
    printf("lib_vio> C,R,S = X,Y,Z (no axis permutation). \n");
  }
}


/* sets origin based on crystallographic or MRC2000 conventions, returns 0 for crystallographic, 1 for MRC2000 style */
int set_origin_get_mode(int nxstart, int nystart, int nzstart, double widthx, double widthy, double widthz, double xorigin, double yorigin, double zorigin, double *origx, double *origy, double *origz)
{

  if (fabs(xorigin) < 0.0001 && fabs(yorigin) < 0.0001 && fabs(zorigin) < 0.0001) { /* seem to have crystallographic origin */
    printf("lib_vio> Using crystallographic (CCP4) style origin defined by unit cell start indices.\n");
    *origx = nxstart * widthx;
    *origy = nystart * widthy;
    *origz = nzstart * widthz;
    return 0;
  } else { /* seem to have MRC2000 */
    printf("lib_vio> Using MRC2000 style origin defined by [X,Y,Z]ORIGIN fields.\n");
    *origx = xorigin;
    *origy = yorigin;
    *origz = zorigin;
    return 1;
  }
}

/* if necessary, map non-cubic maps to cubic lattice by interpolation */
void assert_cubic_map(int orom, int cubic, double alpha, double beta, 
                      double gamma, double widthx, double widthy, 
                      double widthz, unsigned nx, unsigned ny, unsigned nz,
                      int nxstart, int nystart, int nzstart, double xorigin, 
                      double yorigin, double zorigin, double *pwidth, 
                      double *porigx, double *porigy,double *porigz, 
                      unsigned *pextx, unsigned *pexty, unsigned *pextz, 
                      double **pphi)
{

  unsigned long nvox;
  double *pp2;
  double origx, origy, origz;

  /* distinguish between cubic, orthonormal, and skewed */
  if (cubic) {
    set_origin_get_mode(nxstart, nystart, nzstart, widthx, widthy, widthz, 
                        xorigin, yorigin, zorigin, porigx, porigy, porigz);
    printf("lib_vio> Cubic lattice present. \n");
    *pwidth = widthx;
    *pextx = nx;
    *pexty = ny;
    *pextz = nz;
  } else {
    if (orom) {
      set_origin_get_mode(nxstart, nystart, nzstart, widthx, widthy, widthz, 
                          xorigin, yorigin, zorigin, &origx, &origy, &origz);
      printf("lib_vio> Orthogonal lattice with unequal spacings in x, y, z detected.\n");
      printf("lib_vio> Interpolating map to cubic lattice using smallest detected spacing rounded to nearest 0.1 Angstrom.\n");
      /* for interpolation, find minimum voxel spacing, rounded to nearest 0.1 Angstrom for good measure */
      *pwidth = widthx;
      if (widthy < *pwidth) *pwidth = widthy;
      if (widthz < *pwidth) *pwidth = widthz;
      *pwidth = floor(*pwidth * 10.0 + 0.5) / 10.0;
      /* the new map will have the maximum box dimensions that do not exceed the old map */
      interpolate_map(&pp2, pextx, pexty, pextz, porigx, porigy, porigz,
                      *pwidth, *pwidth, *pwidth, *pphi, nx, ny, nz, origx,
                      origy, origz, widthx, widthy, widthz);
      nvox = *pextx * *pexty * *pextz;
      cp_vect_destroy(pphi, &pp2, nvox);
    } else {
      printf("lib_vio> Skewed, non-orthogonal lattice detected.\n");
      printf("lib_vio> Interpolating skewed map to cubic lattice using smallest detected spacing rounded to nearest 0.1 Angstrom.\n");
      /* here, the new map will have the minimum box dimensions that fully enclose the old map */
      interpolate_skewed_map_to_cubic(&pp2, pextx, pexty, pextz, porigx, porigy,
                                      porigz, pwidth, *pphi, nx, ny, nz, 
                                      nxstart, nystart, nzstart, xorigin, 
                                      yorigin, zorigin, widthx, widthy, widthz,
                                      alpha, beta, gamma);
      nvox = *pextx * *pexty * *pextz;
      cp_vect_destroy(pphi, &pp2, nvox);
    }
  }

  /* check for registration of lattice with origin of coordinate system */
  if (test_registration(*porigx, *porigy, *porigz, *pwidth) == 0) 
    fprintf(stderr, "lib_vio> Input grid not in register with origin of coordinate system.\n");
}


/* interpolate skewed map to cubic lattice within rectangular bounding box */
/* pphiout is allocated and new output map parameters are returned */
void interpolate_skewed_map_to_cubic(double **pphiout, unsigned *pextx, 
                                     unsigned *pexty, unsigned *pextz,
                                     double *porigx, double *porigy, 
                                     double *porigz, double *pwidth,
                                     double *phiin, unsigned nx, unsigned ny, 
                                     unsigned nz, int nxstart, int nystart, 
                                     int nzstart, double xorigin, 
                                     double yorigin, double zorigin,
                                     double widthx, double widthy, 
                                     double widthz, double alpha, double beta, 
                                     double gamma)
{

  unsigned long pnvox;
  double ax, ay, az, bx, by, bz, cx, cy, cz, aix, aiy, aiz, bix, biy, biz, cix, ciy, ciz;
  double t1x, t1y, t2x, t2y, t3x, t3y, t4x, t4y, cdet, scz;
  double ux[8], uy[8], uz[8], uxmin, uxmax, uymin, uymax, uzmin, uzmax;
  double xpos, ypos, zpos, gx, gy, gz, a, b, c;
  int x0, y0, z0, x1, y1, z1;
  double endox, endoy, endoz;
  int i, indx, indy, indz;
  double origx, origy, origz;

  /* forward transform skewed -> rectangular (ax,ay,az,bx,by,bz,cx,cy,cz) */
  /* compute unit cell vectors a b c by intersection of projections in a,b plane; Bronstein & Semendjajew 2.6.6.1 */
  ax = 1.0;
  ay = 0.0;
  az = 0.0;
  bx = cos(PI * gamma / 180.0);
  by = sin(PI * gamma / 180.0);
  bz = 0.0;
  t1x = cos(PI * (gamma + alpha) / 180.0);
  t1y = sin(PI * (gamma + alpha) / 180.0);
  t2x = cos(PI * (gamma - alpha) / 180.0);
  t2y = sin(PI * (gamma - alpha) / 180.0);
  t3x = cos(PI * beta / 180.0);
  t3y = sin(PI * beta / 180.0);
  t4x = cos(PI * beta / 180.0);
  t4y = -1.0 * sin(PI * beta / 180.0);
  cdet = (t4y - t3y) * (t2x - t1x) - (t2y - t1y) * (t4x - t3x);
  if (fabs(cdet) < 1E-15) {
    error_divide_zero(71330, "lib_vio");
  }
  cx = ((t4x - t3x) * (t1y * t2x - t2y * t1x) - 
        (t2x - t1x) * (t3y * t4x - t4y * t3x)) / cdet;
  cy = ((t4y - t3y) * (t1y * t2x - t2y * t1x) - 
        (t2y - t1y) * (t3y * t4x - t4y * t3x)) / cdet;
  scz = 1.0 - (cx * cx + cy * cy);
  if (scz < 0.0) {
    error_sqrt_negative(71340, "lib_vio");
  }
  cz = sqrt(scz);

  /* inverse transform rectangular -> skewed (aix,aiy,aiz,bix,biy,biz,cix,ciy,ciz) */
  aix = 1.0;
  aiy = 0.0;
  aiz = 0.0;
  bix = -bx / by;
  biy = 1.0 / by;
  biz = 0.0;
  cix = (bx * cy - cx * by) / (by * cz);
  ciy = -cy / (by * cz);
  ciz = 1.0 / cz;

  /* assign origin and map it to skewed coordinates if it is not yet skewed */
  if (set_origin_get_mode(nxstart, nystart, nzstart, widthx, widthy, widthz, xorigin, yorigin, zorigin, &origx, &origy, &origz)) {
    gx = aix * origx + bix * origy + cix * origz;
    gy = aiy * origx + biy * origy + ciy * origz;
    gz = aiz * origx + biz * origy + ciz * origz;
    origx = gx;
    origy = gy;
    origz = gz;
  }

  /* compute actual x y z extent of the skewed map: */
  endox = origx + (nx - 1) * widthx;
  endoy = origy + (ny - 1) * widthy;
  endoz = origz + (nz - 1) * widthz;
  ux[0] = ax * origx + bx * origy + cx * origz;
  uy[0] = ay * origx + by * origy + cy * origz;
  uz[0] = az * origx + bz * origy + cz * origz;
  ux[1] = ax * endox + bx * origy + cx * origz;
  uy[1] = ay * endox + by * origy + cy * origz;
  uz[1] = az * endox + bz * origy + cz * origz;
  ux[2] = ax * origx + bx * endoy + cx * origz;
  uy[2] = ay * origx + by * endoy + cy * origz;
  uz[2] = az * origx + bz * endoy + cz * origz;
  ux[3] = ax * origx + bx * origy + cx * endoz;
  uy[3] = ay * origx + by * origy + cy * endoz;
  uz[3] = az * origx + bz * origy + cz * endoz;
  ux[4] = ax * endox + bx * endoy + cx * origz;
  uy[4] = ay * endox + by * endoy + cy * origz;
  uz[4] = az * endox + bz * endoy + cz * origz;
  ux[5] = ax * origx + bx * endoy + cx * endoz;
  uy[5] = ay * origx + by * endoy + cy * endoz;
  uz[5] = az * origx + bz * endoy + cz * endoz;
  ux[6] = ax * endox + bx * origy + cx * endoz;
  uy[6] = ay * endox + by * origy + cy * endoz;
  uz[6] = az * endox + bz * origy + cz * endoz;
  ux[7] = ax * endox + bx * endoy + cx * endoz;
  uy[7] = ay * endox + by * endoy + cy * endoz;
  uz[7] = az * endox + bz * endoy + cz * endoz;
  uxmin = 1E20;
  for (i = 0; i < 8; i++) if (ux[i] < uxmin) uxmin = ux[i];
  uymin = 1E20;
  for (i = 0; i < 8; i++) if (uy[i] < uymin) uymin = uy[i];
  uzmin = 1E20;
  for (i = 0; i < 8; i++) if (uz[i] < uzmin) uzmin = uz[i];
  uxmax = -1E20;
  for (i = 0; i < 8; i++) if (ux[i] > uxmax) uxmax = ux[i];
  uymax = -1E20;
  for (i = 0; i < 8; i++) if (uy[i] > uymax) uymax = uy[i];
  uzmax = -1E20;
  for (i = 0; i < 8; i++) if (uz[i] > uzmax) uzmax = uz[i];

  /* for interpolation, find minimum voxel spacing, rounded to nearest 0.1 Angstrom for good measure */
  *pwidth = widthx; /* ax = 1 */
  if ((widthy * by) < *pwidth) *pwidth = widthy * by;
  if ((widthz * cz) < *pwidth) *pwidth = widthz * cz;
  *pwidth = floor(*pwidth * 10.0 + 0.5) / 10.0;

  /* compute output origin */
  /* we start pphiout at or below lower bound of phiin, and assert the new origin is in register with origin of the orthogonal coordinate system */
  *porigx = *pwidth * floor(uxmin / *pwidth);
  *porigy = *pwidth * floor(uymin / *pwidth);
  *porigz = *pwidth * floor(uzmin / *pwidth);

  /* compute output map dimensions */
  /* we end pphiout at or above upper bound of phiin */
  *pextx = (int)(ceil(uxmax / *pwidth) - floor(uxmin / *pwidth) + 1.5);
  *pexty = (int)(ceil(uymax / *pwidth) - floor(uymin / *pwidth) + 1.5);
  *pextz = (int)(ceil(uzmax / *pwidth) - floor(uzmin / *pwidth) + 1.5);
  pnvox = *pextx * *pexty * *pextz;

  /* create output map, and loop through its cubic lattice to perform interpolation */
  do_vect(pphiout, pnvox);
  for (indz = 0; indz < (int)*pextz; indz++)
    for (indy = 0; indy < (int)*pexty; indy++)
      for (indx = 0; indx < (int)*pextx; indx++) {
        /* determine position in orthogonal coordinates */
        xpos = *porigx + indx * *pwidth;
        ypos = *porigy + indy * *pwidth;
        zpos = *porigz + indz * *pwidth;

        /* compute position of probe cube within skewed map in voxel units */
        gx = (aix * xpos + bix * ypos + cix * zpos - origx) / widthx;
        gy = (aiy * xpos + biy * ypos + ciy * zpos - origy) / widthy;
        gz = (aiz * xpos + biz * ypos + ciz * zpos - origz) / widthz;
        x0 = (int) floor(gx);
        y0 = (int) floor(gy);
        z0 = (int) floor(gz);
        x1 = x0 + 1;
        y1 = y0 + 1;
        z1 = z0 + 1;

        /* if probe cube is fully within skewed map, do interpolate */
        if (x0 >= 0 && x1 < (int)nx && y0 >= 0 && y1 < (int)ny && 
            z0 >= 0 && z1 < (int)nz) {
          a = gx - x0;
          b = gy - y0;
          c = gz - z0;
          *(*pphiout + gidz_general(indz, indy, indx, *pexty, *pextx)) =
            a * b * c * *(phiin + gidz_general(z1, y1, x1, ny, nx)) +
            (1 - a) * b * c * *(phiin + gidz_general(z1, y1, x0, ny, nx)) +
            a * (1 - b) * c * *(phiin + gidz_general(z1, y0, x1, ny, nx)) +
            a * b * (1 - c) * *(phiin + gidz_general(z0, y1, x1, ny, nx)) +
            a * (1 - b) * (1 - c) * *(phiin + gidz_general(z0, y0, x1, ny, nx)) +
            (1 - a) * b * (1 - c) * *(phiin + gidz_general(z0, y1, x0, ny, nx)) +
            (1 - a) * (1 - b) * c * *(phiin + gidz_general(z1, y0, x0, ny, nx)) +
            (1 - a) * (1 - b) * (1 - c) * *(phiin + gidz_general(z0, y0, x0, ny, nx));
        }
      }
  printf("lib_vio> Conversion to cubic lattice completed. \n");
}


/* write X-PLOR map to vol_file */
void write_xplor(char *vol_file, double pwidth, double porigx, double porigy,
                 double porigz, unsigned pextx, unsigned pexty, unsigned pextz, double *pphi)
{
  FILE *fout;
  long mxstart, mystart, mzstart, mxend, myend, mzend;
  unsigned indx, indy, indz;
  unsigned long count;
  unsigned extx2, exty2, extz2;
  double origx2, origy2, origz2;
  double *pphi2;

  fout = fopen(vol_file, "w");
  if (fout == NULL) {
    error_open_filename(71410, "lib_vio", vol_file);
  }

  /* X-PLOR does not support free origin definitions, must work within indexing constraints */
  /* if necessary, bring into register with coordinate system origin (for integer start indices) */
  if (test_registration(porigx, porigy, porigz, pwidth) == 0) {
    printf("lib_vio> Input grid not in register with origin of coordinate system.\n");
    printf("lib_vio> Data will be interpolated to fit crystallographic X-PLOR format.\n");
  }
  interpolate_map(&pphi2, &extx2, &exty2, &extz2, &origx2, &origy2, &origz2,
                  pwidth, pwidth, pwidth, pphi, pextx, pexty, pextz, porigx,
                  porigy, porigz, pwidth, pwidth, pwidth);

  /* compute indices */
  mxstart = (long) floor((origx2 / pwidth) + 0.5);
  mystart = (long) floor((origy2 / pwidth) + 0.5);
  mzstart = (long) floor((origz2 / pwidth) + 0.5);
  mxend = mxstart + extx2 - 1;
  myend = mystart + exty2 - 1;
  mzend = mzstart + extz2 - 1;

  /* write map */
  printf("lib_vio>\n");
  printf("lib_vio> Writing X-PLOR formatted (ASCII) volumetric map \n");
  fprintf(fout, " \n");
  fprintf(fout, " 2 !NTITLE \n");
  fprintf(fout, "REMARKS FILENAME=\"%s\" \n", vol_file);
  fprintf(fout, "REMARKS created by the Situs lib_vio library\n");
  fprintf(fout, "%8d%8ld%8ld%8d%8ld%8ld%8d%8ld%8ld\n", extx2, mxstart, mxend, exty2, mystart, myend, extz2, mzstart, mzend);
  fprintf(fout, "%12.5E%12.5E%12.5E%12.5E%12.5E%12.5E\n", extx2 * pwidth, exty2 * pwidth, extz2 * pwidth, 90.0, 90.0, 90.0);
  fprintf(fout, "ZYX\n");
  for (indz = 0; indz < extz2; indz++) {
    fprintf(fout, "%8d\n", indz);
    count = 0;
    for (indy = 0; indy < exty2; indy++) for (indx = 0; indx < extx2; indx++) {
        if ((count + 1) % 6 == 0) fprintf(fout, "%12.5E \n", *(pphi2 + gidz_general(indz, indy, indx, exty2, extx2)));
        else fprintf(fout, "%12.5E", *(pphi2 + gidz_general(indz, indy, indx, exty2, extx2)));
        ++count;
      }
    if ((count) % 6 != 0) fprintf(fout, " \n");
  }
  fprintf(fout, "%8d\n", -9999);
  fclose(fout);

  /* print some info */
  printf("lib_vio> X-PLOR map written to file %s \n", vol_file);
  printf("lib_vio> X-PLOR map indexing (counting from 0): \n");
  printf("lib_vio>       NA = %8d  (# of X intervals in unit cell) \n", extx2);
  printf("lib_vio>     AMIN = %8ld  (start index X) \n", mxstart);
  printf("lib_vio>     AMAX = %8ld  (end index X) \n", mxend);
  printf("lib_vio>       NB = %8d  (# of Y intervals in unit cell) \n", exty2);
  printf("lib_vio>     BMIN = %8ld  (start index Y) \n", mystart);
  printf("lib_vio>     BMAX = %8ld  (end index Y) \n", myend);
  printf("lib_vio>       NC = %8d  (# of Z intervals in unit cell) \n", extz2);
  printf("lib_vio>     CMIN = %8ld  (start index Z) \n", mzstart);
  printf("lib_vio>     CMAX = %8ld  (end index Z) \n", mzend);
  printf("lib_vio> X-PLOR unit cell (based on the extent of the input density map): \n");
  printf("lib_vio>        A = %8.3f  (unit cell dimension) \n", extx2 * pwidth);
  printf("lib_vio>        B = %8.3f  (unit cell dimension) \n", exty2 * pwidth);
  printf("lib_vio>        C = %8.3f  (unit cell dimension) \n", extz2 * pwidth);
  printf("lib_vio>    ALPHA = %8.3f  (unit cell angle) \n", 90.0);
  printf("lib_vio>     BETA = %8.3f  (unit cell angle) \n", 90.0);
  printf("lib_vio>    GAMMA = %8.3f  (unit cell angle) \n", 90.0);
}



/* write MRC / CCP4 map to vol_file in auto or manual mode */
void write_mrc(int automode, char *vol_file, double pwidth, double porigx, 
               double porigy, double porigz, unsigned pextx, unsigned pexty, 
               unsigned pextz, double *pphi)
{
  FILE *fout;
  long nxstart, nystart, nzstart;
  float xorigin, yorigin, zorigin;
  unsigned long count, pnvox;
  long nx, ny, nz, mx, my, mz;
  long mode = 2;
  int modesel = 1;
  float xlen, ylen, zlen, alpha, beta, gamma;
  long mapc, mapr, maps;
  long ispg = 1, nlabl = 0;
  float dummy = 0.0f;
  double dmax, dmin, dav, dsig;
  float fmax, fmin, fav, fsig;
  double dshift, dscale;
  char dummychar = '\0';
  char mapstring[4];
  char machstring[4];
  int i, wmaperr;
  long lskflg = 0;
  float skwmat11 = 1.0f, skwmat12 = 0.0f, skwmat13 = 0.0f, skwmat21 = 0.0f;
  float skwmat22 = 1.0f, skwmat23 = 0.0f, skwmat31 = 0.0f, skwmat32 = 0.0f;
  float  skwmat33 = 1.0f, skwtrn1 = 0.0f, skwtrn2 = 0.0f, skwtrn3 = 0.0f;
  char endchar[4] = {1, 1, 0, 0};

  fout = fopen(vol_file, "wb");
  if (fout == NULL) {
    error_open_filename(71420, "lib_vio", vol_file);
  }

  /* set initial values of map parameters */
  pnvox = pextx * pexty * pextz;
  nx = pextx;
  ny = pexty;
  nz = pextz;
  mapstring[0]  = 'M';
  mapstring[1]  = 'A';
  mapstring[2]  = 'P';
  mapstring[3]  = ' ';
  if (*((int *)endchar) > 65536) {   /* big endian */
    machstring[0]  = 17;
    machstring[1]  = 17;
  } else { /* little endian */
    machstring[0]  = 68;
    machstring[1]  = 65;
  }
  machstring[2]  = 0;
  machstring[3]  = 0;
  calc_map_info(pphi, pnvox, &dmax, &dmin, &dav, &dsig);
  fmax = dmax;
  fmin = dmin;
  fav = dav;
  fsig = dsig;
  xorigin = porigx;
  yorigin = porigy;
  zorigin = porigz;
  mx = pextx;
  my = pexty;
  mz = pextz;
  xlen = pwidth * mx;
  ylen = pwidth * my;
  zlen = pwidth * mz;
  alpha = 90.0f;
  beta = 90.0f;
  gamma = 90.0f;
  mapc = 1;
  mapr = 2;
  maps = 3;

  /* if grid is not in register with origin, we set the CCP4 start indices to zero */
  if (test_registration(porigx, porigy, porigz, pwidth) == 0) {
    nxstart = 0;
    nystart = 0;
    nzstart = 0;
  } else {
    nxstart = (long)floor((porigx / pwidth) + 0.5);
    nystart = (long)floor((porigy / pwidth) + 0.5);
    nzstart = (long)floor((porigz / pwidth) + 0.5);
  }

  /* optional manual override of variable map parameters */
  if (automode == 0) {
    printf("lib_vio> Manual override of MRC / CCP4 header fields.\n");
    printf("lib_vio> \n");
    printf("lib_vio> The currently assigned data type is 32-bit floats (MODE 2)\n");
    printf("lib_vio> Select one of the following options: \n");
    printf("lib_vio>      1: keep MODE 2, 32-bit floats \n");
    printf("lib_vio>      2: MODE 0, signed 8-bit (range -127...128 as in CCP4), clip any out-of range\n");
    printf("lib_vio>      3: MODE 0, signed 8-bit (range -127...128 as in CCP4), scale to fit\n");
    printf("lib_vio>      4: MODE 0, signed 8-bit (range -127...128 as in CCP4), shift and scale to fit\n");
    printf("lib_vio>      5: MODE 0, unsigned 8-bit (range 0...255 as in old MRC), clip any out-of range\n");
    printf("lib_vio>      6: MODE 0, unsigned 8-bit (range 0...255 as in old MRC), scale pos., clip neg.\n");
    printf("lib_vio>      7: MODE 0, unsigned 8-bit (range 0...255 as in old MRC), shift and scale to fit\n");
    printf("lib_vio> \n");
    printf("lib_vio> Enter selection number: ");
    modesel = readln_int();
    if (modesel < 1 && modesel > 7) {
      modesel = 1;
      mode = 2;
      printf("lib_vio> Did not recognize value, assuming MODE 2, 32-bit floats. \n");
    } else if (modesel > 1) mode = 0;

    printf("lib_vio> Note: The written data is not affected by changing the following header values.\n");
    printf("lib_vio> The currently assigned CCP4-style NC field (# columns) is %ld \n", nx);
    printf("lib_vio> Enter the same or a new value: ");
    nx = readln_int();
    printf("lib_vio> The currently assigned CCP4-style NR field (# rows) is %ld \n", ny);
    printf("lib_vio> Enter the same or a new value: ");
    ny = readln_int();
    printf("lib_vio> The currently assigned CCP4-style NS field (# sections) is %ld \n", nz);
    printf("lib_vio> Enter the same or a new value: ");
    nz = readln_int();
    if (((unsigned long)nx * (unsigned long)ny * (unsigned long)nz) != pnvox) {
      printf("lib_vio> \n");
      fprintf(stderr, "lib_vio> Warning: NC * NR * NS does not match the total number of voxels in the map.\n");
      printf("lib_vio> \n");
    }
    printf("lib_vio> The currently assigned CCP4-style NCSTART field (column start index) is %ld \n", nxstart);
    if (nxstart == 0 && test_registration(porigx, porigy, porigz, pwidth) == 0) {
      printf("lib_vio> Set to zero by default because input grid not in register with origin of coordinate system.\n");
    }
    printf("lib_vio> Enter the same or a new value: ");
    nxstart = readln_int();
    printf("lib_vio> The currently assigned CCP4-style NRSTART field (row start index) is %ld \n", nystart);
    if (nxstart == 0 && test_registration(porigx, porigy, porigz, pwidth) == 0) {
      printf("lib_vio> Set to zero by default because input grid not in register with origin of coordinate system.\n");
    }
    printf("lib_vio> Enter the same or a new value: ");
    nystart = readln_int();
    printf("lib_vio> The currently assigned CCP4-style NSSTART field (section start index) is %ld \n", nzstart);
    if (nxstart == 0 && test_registration(porigx, porigy, porigz, pwidth) == 0) {
      printf("lib_vio> Set to zero by default because input grid not in register with origin of coordinate system.\n");
    }
    printf("lib_vio> Enter the same or a new value: ");
    nzstart = readln_int();
    printf("lib_vio> The currently assigned MX field (# of X intervals in the unit cell) is %ld \n", mx);
    printf("lib_vio> Enter the same or a new value: ");
    mx = readln_int();
    printf("lib_vio> The currently assigned MY field (# of Y intervals in the unit cell) is %ld \n", my);
    printf("lib_vio> Enter the same or a new value: ");
    my = readln_int();
    printf("lib_vio> The currently assigned MZ field (# of Z intervals in the unit cell) is %ld \n", mz);
    printf("lib_vio> Enter the same or a new value: ");
    mz = readln_int();
    printf("lib_vio> The currently assigned X unit cell dimension is %f Angstrom\n", xlen);
    printf("lib_vio> Enter the same or a new value: ");
    xlen = readln_double();
    printf("lib_vio> The currently assigned Y unit cell dimension is %f Angstrom\n", ylen);
    printf("lib_vio> Enter the same or a new value: ");
    ylen = readln_double();
    printf("lib_vio> The currently assigned Z unit cell dimension is %f Angstrom\n", zlen);
    printf("lib_vio> Enter the same or a new value: ");
    zlen = readln_double();
    printf("lib_vio> The currently assigned unit cell angle alpha is %f degrees \n", alpha);
    printf("lib_vio> Enter the same or a new value: ");
    alpha = readln_double();
    printf("lib_vio> The currently assigned unit cell angle beta is %f degrees \n", beta);
    printf("lib_vio> Enter the same or a new value: ");
    beta = readln_double();
    printf("lib_vio> The currently assigned unit cell angle gamma is %f degrees \n", gamma);
    printf("lib_vio> Enter the same or a new value: ");
    gamma = readln_double();
    printf("lib_vio> The currently assigned MAPC field (axis order) is %ld \n", mapc);
    printf("lib_vio> Enter the same or a new value: ");
    mapc = readln_int();
    wmaperr = 1;
    if (mapc == 1 || mapc == 2 || mapc == 3) wmaperr = 0;
    if (wmaperr) {
      mapc = 1;
      mapr = 2;
      maps = 3;
      printf("lib_vio> Inconsistent axis order values, assuming mapc = 1, mapr = 2, maps = 3. \n");
    } else {
      printf("lib_vio> The currently assigned MAPR field (axis order) is %ld \n", mapr);
      printf("lib_vio> Enter the same or a new value: ");
      mapr = readln_int();
      wmaperr = 1;
      if (mapc == 1 && mapr == 2) wmaperr = 0;
      if (mapc == 1 && mapr == 3) wmaperr = 0;
      if (mapc == 2 && mapr == 1) wmaperr = 0;
      if (mapc == 2 && mapr == 3) wmaperr = 0;
      if (mapc == 3 && mapr == 1) wmaperr = 0;
      if (mapc == 3 && mapr == 2) wmaperr = 0;
      if (wmaperr) {
        mapc = 1;
        mapr = 2;
        maps = 3;
        printf("lib_vio> Inconsistent axis order values, assuming mapc = 1, mapr = 2, maps = 3. \n");
      } else {
        printf("lib_vio> The currently assigned MAPS field (axis order) is %ld \n", maps);
        printf("lib_vio> Enter the same or a new value: ");
        maps = readln_int();
        wmaperr = 1;
        if (mapc == 1 && mapr == 2 && maps == 3) wmaperr = 0;
        if (mapc == 1 && mapr == 3 && maps == 2) wmaperr = 0;
        if (mapc == 2 && mapr == 1 && maps == 3) wmaperr = 0;
        if (mapc == 2 && mapr == 3 && maps == 1) wmaperr = 0;
        if (mapc == 3 && mapr == 1 && maps == 2) wmaperr = 0;
        if (mapc == 3 && mapr == 2 && maps == 1) wmaperr = 0;
        if (wmaperr) {
          mapc = 1;
          mapr = 2;
          maps = 3;
          printf("lib_vio> Inconsistent axis order values, assuming mapc = 1, mapr = 2, maps = 3. \n");
        }
      }
    }
    printf("lib_vio> The currently assigned MRC2000-style XORIGIN field is %f Angstrom\n", xorigin);
    printf("lib_vio> Enter the same or a new value: ");
    xorigin = readln_double();
    printf("lib_vio> The currently assigned MRC2000-style YORIGIN field is %f Angstrom\n", yorigin);
    printf("lib_vio> Enter the same or a new value: ");
    yorigin = readln_double();
    printf("lib_vio> The currently assigned MRC2000-style ZORIGIN field is %f Angstrom\n", zorigin);
    printf("lib_vio> Enter the same or a new value: ");
    zorigin = readln_double();
  } /* end manual override */

  /* adjust MODE 0 maps as necessary */
  switch (modesel) {
    case 1:
      break;
    case 2:
      if (clipped(pphi, pnvox, 127.0, -128.0)) 
        printf("lib_vio> Density values were clipped to fit signed 8-bit data format.\n");
      else 
        printf("lib_vio> Density values already fit signed 8-bit data format, no clipping necessary.\n");
      break;
    case 3:
      dscale = dmax / 127.0;
      if (dmin / -128.0 > dscale) 
        dscale = dmin / -128.0;
      printf("lib_vio> Density values will be rescaled to fit signed 8-bit data format.\n");
      printf("lib_vio> Scaling factor (by which values will be divided): %f .\n", dscale);
      normalize(pphi, pnvox, dscale);
      break;
    case 4:
      dscale = (dmax - dmin) / 255.0;
      dshift = (dmax + dmin) / 2.0 + 0.5 * dscale;
      printf("lib_vio> Density values will be centered and rescaled to fit signed 8-bit data format.\n");
      printf("lib_vio> Center offset value (will be added to map): %f .\n", -dshift);
      printf("lib_vio> Scaling factor (by which values will be divided): %f .\n", dscale);
      floatshift(pphi, pnvox, dshift);
      normalize(pphi, pnvox, dscale);
      break;
    case 5:
      if (clipped(pphi, pnvox, 255.0, 0.0)) printf("lib_vio> Density values were clipped to fit unsigned 8-bit data format.\n");
      else printf("lib_vio> Density values already fit unsigned 8-bit data format, no clipping necessary.\n");
      break;
    case 6:
      dscale = dmax / 255.0;
      printf("lib_vio> Density values will be rescaled to fit unsigned 8-bit data format.\n");
      printf("lib_vio> Scaling factor (by which values will be divided): %f .\n", dscale);
      normalize(pphi, pnvox, dscale);
      if (clipped(pphi, pnvox, 256.0, 0.0)) printf("lib_vio> Negative density values were clipped to fit unsigned 8-bit data format.\n");
      break;
    case 7:
      dscale = (dmax - dmin) / 255.0;
      dshift = (dmax + dmin) / 2.0 - 127.5 * dscale;
      printf("lib_vio> Density values will be centered and rescaled to fit unsigned 8-bit data format.\n");
      printf("lib_vio> Center offset value (will be added to map): %f .\n", -dshift);
      printf("lib_vio> Scaling factor (by which values will be divided): %f .\n", dscale);
      floatshift(pphi, pnvox, dshift);
      normalize(pphi, pnvox, dscale);
      break;
    default:
      error_option(70211, "lib_vio");
  }

  /* recompute map statistics and then adjust unsigned MODE 0 values for storage as signed */
  if (modesel > 1) {
    calc_map_info(pphi, pnvox, &dmax, &dmin, &dav, &dsig);
    fmax = dmax;
    fmin = dmin;
    fav = dav;
    fsig = dsig;
  }
  if (modesel > 4) {
    printf("lib_vio> Converting the unsigned 8-bit (0...255) values to signed (-128...127) for storage (ISO/IEC 10967).\n");
    printf("lib_vio> The conversion subtracts 256 from density values above 127.\n");
    printf("lib_vio> Check your software documentation if the `unsigned` convention is supported for MODE 0 maps.\n");
    for (count = 0; count < pnvox; count++) if (*(pphi + count) >= 127.5) *(pphi + count) -= 256.0;
  }

  /* Warning */
  /* Do not recompute map statistics after unsigned to signed conversion */

  /* write header */
  printf("lib_vio> Writing MRC / CCP4 (binary) volumetric map \n");
  wmaperr = 0;
  if (fwrite(&nx, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&ny, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&nz, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&mode, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&nxstart, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&nystart, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&nzstart, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&mx, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&my, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&mz, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&xlen, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&ylen, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&zlen, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&alpha, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&beta, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&gamma, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&mapc, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&mapr, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&maps, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&fmin, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&fmax, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&fav, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&ispg, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&dummy, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&lskflg, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&skwmat11, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&skwmat12, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&skwmat13, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&skwmat21, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&skwmat22, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&skwmat23, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&skwmat31, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&skwmat32, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&skwmat33, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&skwtrn1, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&skwtrn2, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&skwtrn3, 4, 1, fout) != 1) wmaperr = 1;
  for (i = 38; i < 50; ++i) if (fwrite(&dummy, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&xorigin, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&yorigin, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&zorigin, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(mapstring, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(machstring, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&fsig, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&nlabl, 4, 1, fout) != 1) wmaperr = 1;
  for (i = 0; i < 800; ++i) if (fwrite(&dummychar, sizeof(char), 1, fout) != 1) wmaperr = 1;

  /* write actual data */
  switch (mode) {
    case 0:
      for (count = 0; count < pnvox; count++) {
        dummy = floor(*(pphi + count) + 0.5);
        if (dummy < -128 || dummy > 127) wmaperr = 1;
        dummychar = (char)dummy;
        if (fwrite(&dummychar, 1, 1, fout) != 1) wmaperr = 1;
      }
      break;
    case 2:
      for (count = 0; count < pnvox; count++) {
        dummy = *(pphi + count);
        if (fwrite(&dummy, 4, 1, fout) != 1) wmaperr = 1;
      }
      break;
    default:
      wmaperr = 1;
      break;
  }
  if (wmaperr != 0) {
    error_write_filename(71430, "lib_vio");
  }
  fclose(fout);

  /* print some info */
  printf("lib_vio> Volumetric map written to file %s \n", vol_file);
  printf("lib_vio> Header information: \n");
  printf("lib_vio>       NC = %8ld  (# columns)\n", nx);
  printf("lib_vio>       NR = %8ld  (# rows)\n", ny);
  printf("lib_vio>       NS = %8ld  (# sections)\n", nz);
  printf("lib_vio>     MODE = %8ld  (data type: 0: 8-bit char, 2: 32-bit float)\n", mode);
  printf("lib_vio>  NCSTART = %8ld  (index of first column, counting from 0)\n", nxstart);
  printf("lib_vio>  NRSTART = %8ld  (index of first row, counting from 0)\n", nystart);
  printf("lib_vio>  NSSTART = %8ld  (index of first section, counting from 0)\n", nzstart);
  printf("lib_vio>       MX = %8ld  (# of X intervals in unit cell)\n", mx);
  printf("lib_vio>       MY = %8ld  (# of Y intervals in unit cell)\n", my);
  printf("lib_vio>       MZ = %8ld  (# of Z intervals in unit cell)\n", mz);
  printf("lib_vio> X length = %8.3f  (unit cell dimension)\n", xlen);
  printf("lib_vio> Y length = %8.3f  (unit cell dimension)\n", ylen);
  printf("lib_vio> Z length = %8.3f  (unit cell dimension)\n", zlen);
  printf("lib_vio>    Alpha = %8.3f  (unit cell angle)\n", alpha);
  printf("lib_vio>     Beta = %8.3f  (unit cell angle)\n", beta);
  printf("lib_vio>    Gamma = %8.3f  (unit cell angle)\n", gamma);
  printf("lib_vio>     MAPC = %8ld  (columns axis: 1=X,2=Y,3=Z)\n", mapc);
  printf("lib_vio>     MAPR = %8ld  (rows axis: 1=X,2=Y,3=Z)\n", mapr);
  printf("lib_vio>     MAPS = %8ld  (sections axis: 1=X,2=Y,3=Z)\n", maps);
  printf("lib_vio>     DMIN = %8.3f  (minimum density value)\n", fmin);
  printf("lib_vio>     DMAX = %8.3f  (maximum density value)\n", fmax);
  printf("lib_vio>    DMEAN = %8.3f  (mean density value)\n", fav);
  printf("lib_vio>     ISPG = %8ld  (space group number)\n", ispg);
  printf("lib_vio>   NSYMBT = %8d  (# bytes used for storing symmetry operators)\n", 0);
  printf("lib_vio>   LSKFLG = %8ld  (skew matrix flag: 0:none, 1:follows)\n", lskflg);
  if (lskflg != 0) {
    printf("lib_vio>      S11 = %8.3f  (skew matrix element 11)\n", skwmat11);
    printf("lib_vio>      S12 = %8.3f  (skew matrix element 12)\n", skwmat12);
    printf("lib_vio>      S13 = %8.3f  (skew matrix element 13)\n", skwmat13);
    printf("lib_vio>      S21 = %8.3f  (skew matrix element 21)\n", skwmat21);
    printf("lib_vio>      S22 = %8.3f  (skew matrix element 22)\n", skwmat22);
    printf("lib_vio>      S23 = %8.3f  (skew matrix element 23)\n", skwmat23);
    printf("lib_vio>      S31 = %8.3f  (skew matrix element 31)\n", skwmat31);
    printf("lib_vio>      S32 = %8.3f  (skew matrix element 32)\n", skwmat32);
    printf("lib_vio>      S33 = %8.3f  (skew matrix element 33)\n", skwmat33);
    printf("lib_vio>       T1 = %8.3f  (skew translation element 1)\n", skwtrn1);
    printf("lib_vio>       T2 = %8.3f  (skew translation element 2)\n", skwtrn2);
    printf("lib_vio>       T3 = %8.3f  (skew translation element 3)\n", skwtrn3);
  }
  printf("lib_vio>  XORIGIN = %8.3f  (X origin - MRC2000 only)\n", xorigin);
  printf("lib_vio>  YORIGIN = %8.3f  (Y origin - MRC2000 only)\n", yorigin);
  printf("lib_vio>  ZORIGIN = %8.3f  (Z origin - MRC2000 only)\n", zorigin);
  printf("lib_vio>      MAP =      %c%c%c%c (map string)\n", mapstring[0], mapstring[1], mapstring[2], mapstring[3]);
  printf("lib_vio>   MACHST = %d %d %d %d (machine stamp)\n", machstring[0], machstring[1], machstring[2], machstring[3]);
  printf("lib_vio>      RMS = %8.3f  (density rmsd)\n", fsig);

  /* if auto mode and grid is not in register with origin, issue a warning */
  if (automode == 1 && test_registration(porigx, porigy, porigz, pwidth) == 0) {
    printf("lib_vio> Input grid not in register with origin of coordinate system.\n");
    printf("lib_vio> The origin information was saved only in the MRC2000 format fields.\n");
    printf("lib_vio> The CCP4 start indexing was set to zero.\n");
    printf("lib_vio> To invoke a crystallographic CCP4 indexing, you can force an interpolation \n");
    printf("lib_vio> by converting to an intermediate X-PLOR map using the map2map tool. \n");
  }
}


/* write SPIDER map to vol_file */
void write_spider(char *vol_file, double pwidth, double porigx, double porigy,
                  double porigz, unsigned pextx, unsigned pexty, unsigned pextz,
                  double *pphi)
{
  FILE *fout;
  unsigned long count, pnvox;
  float nslice, nrow, iform, imami, nsam, headrec;
  double dmax, dmin, dav, dsig;
  float fmax, fmin, fav, fsig;
  float iangle, phi, theta, xoff, yoff, zoff, scale, labbyt, lenbyt, irec;
  float istack, inuse, maxim, kangle, phi1, theta1, psi1, phi2, theta2, psi2;
  float gamma;
  float dummy;
  int i, headlen, wmaperr;

  fout = fopen(vol_file, "wb");
  if (fout == NULL) {
    error_open_filename(71460, "lib_vio", vol_file);
  }

  /* set map parameters */
  pnvox = pextx * pexty * pextz;
  nsam = pextx;
  nrow = pexty;
  nslice = pextz;
  iform = 3.0f;
  imami = 1.0f;
  istack = 0.0f;
  inuse = -1.0f;
  maxim = 0.0f;
  kangle = 0.0f;
  phi1 = 0.0f;
  theta1 = 0.0f;
  psi1 = 0.0f;
  phi2 = 0.0f;
  theta2 = 0.0f;
  psi2 = 0.0f;
  scale = 0.0f;
  dummy = 0.0f;
  phi = 0;
  theta = 0;
  gamma = 0;
  xoff = 0;
  yoff = 0;
  zoff = 0;
  iangle = 0;

  /* compute density info */
  calc_map_info(pphi, pnvox, &dmax, &dmin, &dav, &dsig);
  fmax = dmax;
  fmin = dmin;
  fav = dav;
  fsig = dsig;

  /* write map */
  headrec = ceil(256.0f / (pextx * 1.0f));
  lenbyt = nsam * 4.0f;
  labbyt = headrec * lenbyt;
  irec = nslice * nrow + headrec;
  headlen = (int)(headrec * nsam);
  printf("lib_vio>\n");
  printf("lib_vio> Writing SPIDER (binary) volumetric map... \n");
  wmaperr = 0;
  if (fwrite(&nslice, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&nrow, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&irec, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&dummy, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&iform, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&imami, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&fmax, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&fmin, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&fav, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&fsig, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&dummy, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&nsam, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&headrec, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&iangle, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&phi, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&theta, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&gamma, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&xoff, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&yoff, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&zoff, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&scale, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&labbyt, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&lenbyt, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&istack, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&inuse, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&maxim, 4, 1, fout) != 1) wmaperr = 1;
  for (i = 0; i < 4; ++i) if (fwrite(&dummy, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&kangle, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&phi1, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&theta1, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&psi1, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&phi2, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&theta2, 4, 1, fout) != 1) wmaperr = 1;
  if (fwrite(&psi2, 4, 1, fout) != 1) wmaperr = 1;
  for (i = 0; i < (headlen - 37); ++i) if (fwrite(&dummy, 4, 1, fout) != 1) wmaperr = 1;
  for (count = 0; count < pnvox; count++) {
    dummy = *(pphi + count);
    if (fwrite(&dummy, 4, 1, fout) != 1) wmaperr = 1;
  }

  if (wmaperr != 0) {
    error_write_filename(71470, "lib_vio");
  }
  fclose(fout);

  /* print some info */
  printf("lib_vio>   SPIDER map written to file %s \n", vol_file);
  printf("lib_vio>   SPIDER map indexing: \n");
  printf("lib_vio>   NSLICE = %8.f  (# sections)\n", nslice);
  printf("lib_vio>     NROW = %8.f  (# rows)\n", nrow);
  printf("lib_vio>    IFORM = %8.f  (file type specifier)\n", iform);
  printf("lib_vio>    IMAMI = %8.f  (flag: =1 the maximum and minimum values are computed)\n", imami);
  printf("lib_vio>     FMAX = %8.3f  (maximum density value)\n", fmax);
  printf("lib_vio>     FMIN = %8.3f  (minimum density value)\n", fmin);
  printf("lib_vio>       AV = %8.3f  (average density value)\n", fav);
  printf("lib_vio>      SIG = %8.3f  (standard deviation of density distribution)\n", fsig);
  printf("lib_vio>     NSAM = %8.f  (# columns)\n", nsam);
  printf("lib_vio>  HEADREC = %8.f  (number of records in file header)\n", headrec);
  printf("lib_vio>   IANGLE = %8.f  (flag: =1 tilt angles filled)\n", iangle);
  printf("lib_vio>      PHI = %8.3f  (tilt angle)\n", phi);
  printf("lib_vio>    THETA = %8.3f  (tilt angle)\n", theta);
  printf("lib_vio>    GAMMA = %8.3f  (tilt angle)\n", gamma);
  printf("lib_vio>     XOFF = %8.3f  (X offset)\n", xoff);
  printf("lib_vio>     YOFF = %8.3f  (Y offset)\n", yoff);
  printf("lib_vio>     ZOFF = %8.3f  (Z offset)\n", zoff);
  printf("lib_vio>    SCALE = %8.3f  (scale factor)\n", scale);
  printf("lib_vio>   LABBYT = %8.f  (total number of bytes in header)\n", labbyt);
  printf("lib_vio>   LENBYT = %8.f  (record length in bytes)\n", lenbyt);
  printf("lib_vio>   ISTACK = %8.f  (flag; file contains a stack of images)\n", istack);
  printf("lib_vio>    INUSE = %8.f  (flag; this image in stack is used)\n", inuse);
  printf("lib_vio>    MAXIM = %8.f  (maximum image used in stack)\n", maxim);
  printf("lib_vio>   KANGLE = %8.f  (flag; additional angles set)\n", kangle);
  printf("lib_vio>     PHI1 = %8.3f  (additional rotation)\n", phi1);
  printf("lib_vio>   THETA1 = %8.3f  (additional rotation)\n", theta1);
  printf("lib_vio>     PSI1 = %8.3f  (additional rotation)\n", psi1);
  printf("lib_vio>     PHI2 = %8.3f  (additional rotation)\n", phi2);
  printf("lib_vio>   THETA2 = %8.3f  (additional rotation)\n", theta2);
  printf("lib_vio>     PSI2 = %8.3f  (additional rotation)\n", psi2);
  printf("lib_vio> Warning: The voxel spacing %f has not been saved to the SPIDER map.\n", pwidth);
  printf("lib_vio> Warning: The map origin %f,%f,%f (first voxel position) has not been saved to the SPIDER map.\n", porigx, porigy, porigz);
}


