#ifndef __SITUS_LIB_PWK
#define __SITUS_LIB_PWK

#ifdef __cplusplus
extern "C" {
#endif

/* header file for lib_pwk.c */
void copy_atoms(PDB *, PDB *, int, int, int);
void rot_axis(PDB *, PDB *, unsigned, char, double);
void rot_euler(PDB *, PDB *, unsigned, double, double, double);
void translate(PDB *, PDB *, unsigned, double, double, double);
void calc_center(PDB *, unsigned, double *, double *, double *);
double calc_mass(PDB *, unsigned);
double calc_center_mass(PDB *, unsigned, double *, double *, double *);
double calc_sphere(PDB *, unsigned, double, double, double);
void calc_box(PDB *, unsigned, double *, double *, double *, double *, double *, double *);
void project_mass(double **, unsigned long, double, double, double, unsigned, unsigned, unsigned, PDB *, unsigned, double *, unsigned *);
void project_mass_convolve_kernel_corr(double, double,  double, unsigned, unsigned, unsigned, PDB *, unsigned, double *, double *, unsigned, double, unsigned *, double *, double *);
int check_if_inside(double, double, double, double, unsigned, unsigned, unsigned, PDB *, unsigned, double *);

#ifdef __cplusplus
}
#endif

#endif
