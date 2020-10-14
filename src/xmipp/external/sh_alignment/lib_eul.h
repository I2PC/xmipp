#ifndef __SITUS_LIB_EUL
#define __SITUS_LIB_EUL

#ifdef __cplusplus
extern "C" {
#endif

/* header file for lib_eul.c */
void write_eulers(char *, unsigned long, float *, double [3][2], double);
void read_eulers(char *, unsigned long *, float **);
void remap_eulers(double *, double *, double *, double, double, double, double, double, double);
void get_rot_matrix(double [3][3], double, double, double);
void eu_spiral(double [3][2], double, unsigned long *, float **);
void eu_proportional(double [3][2], double, unsigned long *, float **);
void eu_sparsed(double [3][2], double, unsigned long *, float **);
char similar_eulers(double, double, double, double, double, double);

#ifdef __cplusplus
}
#endif

#endif
