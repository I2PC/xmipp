/*********************************************************************
*                           L I B _ V E C                            *
**********************************************************************
* Library is part of the Situs package URL: situs.biomachina.org     *
* (c) Pablo Chacon, John Heumann,  and Willy Wriggers, 2001-2015     *
**********************************************************************
*                                                                    *
* Creating, resetting, copying arrays and other data structures.     *
*                                                                    *
**********************************************************************
* See legal statement for terms of distribution                      *
*********************************************************************/

#include "situs.h"
#include "lib_vec.h"
#include "lib_err.h"

/*
 * Allocation and de-allocation of C arrays and matrices
 * All arrays and matrices are initialized to zero when created
 */

/****** Following routines are for vectors and arrays of doubles ******/

/*====================================================================*/
void zero_vect(double *vect, unsigned long len)
{
  memset(vect, 0, len * sizeof(double));
}

/*====================================================================*/
void do_vect(double **pvect, unsigned long len)
{
  *pvect = (double *) alloc_vect(len,  sizeof(double));
}

/*====================================================================*/
void zero_mat(double **mat, unsigned long len_i, unsigned long len_j)
{
  memset(mat[0], 0, len_i * len_j * sizeof(double));
}

/*====================================================================*/
void cp_vect(double **vect1, double **vect2, unsigned long len)
{
  memcpy(*vect1, *vect2, len * sizeof(double));
}

/*====================================================================*/
/* destroys memory allocated to vect2 after copying */
void cp_vect_destroy(double **pvect1, double **pvect2, unsigned long len)
{
  if (*pvect1)
    free(*pvect1);
  do_vect(pvect1, len);
  cp_vect(pvect1, pvect2, len);
  free_vect_and_zero_ptr((void**)pvect2);
}

/*====================================================================*/
void add_scaled_vect(double *to_vect, double *from_vect, double scalar, unsigned long len)
{
  unsigned long i;
  for (i = 0; i < len; ++i)
    to_vect[i] += scalar * from_vect[i];
}

/****** Following routines are for arbitrary vectors and arrays  ******/

/*====================================================================*/
void *alloc_vect(unsigned int n, size_t elem_size)
{
  void *pvect;

  pvect = calloc(n, elem_size);
  if (!pvect) {
    error_memory_allocation(99901, "lib_cvq");
  }
  return pvect;
};

/*====================================================================*/
void free_vect_and_zero_ptr(void **pvect)
{
  if (*pvect) {
    free(*pvect);
    *pvect = 0;
  }
}

/*====================================================================*/
void free_mat_and_zero_ptr(void ***pmat)
{
  if (*pmat) {
    if (**pmat)
      free(**pmat);
    free(*pmat);
    *pmat = 0;
  }
}
