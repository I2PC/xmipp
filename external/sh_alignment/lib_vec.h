#ifndef __SITUS_LIB_VEC
#define __SITUS_LIB_VEC

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Header file for lib_vec.c
 * Allocation and de-allocation of C arrays and matrices
 * All arrays and matrices are initialized to zero when created
 */

/****** Following routines are for vectors and arrays of doubles ******/
void zero_vect(double *, unsigned long);
void do_vect(double **, unsigned long);

void zero_mat(double **, unsigned long, unsigned long);

void cp_vect(double **, double **, unsigned long);
void cp_vect_destroy(double **, double **, unsigned long);
void add_scaled_vect(double *, double *, double, unsigned long);

/****** Following routines are for arbitrary vectors and arrays ******/
void *alloc_vect(unsigned int n, size_t elem_size);
void free_vect_and_zero_ptr(void **);
void free_mat_and_zero_ptr(void ***);

#ifdef __cplusplus
}
#endif

#endif
