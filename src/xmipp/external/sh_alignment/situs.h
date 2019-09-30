/*********************************************************************
*                           s i t u s . h                            *
**********************************************************************
* Header file for Situs C programs (c) Willy Wriggers, 1998-2015     *
* URL: situs.biomachina.org                                          *
**********************************************************************
* See legal statement for terms of distribution                      *
*********************************************************************/

/* widely used in most C programs */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <dirent.h>
#include <unistd.h>

#ifndef SITUS_H
#define SITUS_H

/* used in colores, collage, eul2pdb, pdbsymm, lib_vio, and lib_eul: */
#define PI 3.14159265358979323846

/* used in colores, collage, eul2pdb, and lib_eul: */
#define ROT_CONV (PI/180.0)

#define SWAPPING(_a,_b,_type) \
{\
  _type _tmp;\
  \
  _tmp = (_a);\
  (_a) = (_b);\
  (_b) = _tmp;\
}

/* PDB structure. Widely used in Situs C programs.
 * Note: members of this structure are NOT in the same order as in the
 * ASCII PDB files. This saves 4 bytes per structure. */
typedef struct {        /* 72-byte struct */
  int   serial;         /*  bytes 0 -  3 */
  int   seq;            /*        4 -  7 */
  float x;              /*        8 - 11 */
  float y;              /*       12 - 15 */
  float z;              /*       16 - 19 */
  float occupancy;      /*       20 - 23 */
  float beta;           /*       24 - 27 */
  int   footnote;       /*       28 - 31 */
  float weight;         /*       32 - 35 */
  char  recd[7];        /*       36 - 42 */
  char  type[3];        /*       43 - 45 */
  char  loc[3];         /*       46 - 48 */
  char  alt[2];         /*       49 - 50 */
  char  res[5];         /*       51 - 55 */
  char  chain[2];       /*       56 - 57 */
  char  icode[2];       /*       58 - 59 */
  char  segid[5];       /*       60 - 64 */
  char  element[3];     /*       65 - 67 */
  char  charge[3];      /*       68 - 70;*/
  char  padding;        /*            71 */
} PDB;

#endif
