#ifndef __SITUS_LIB_VWK
#define __SITUS_LIB_VWK

#ifdef __cplusplus
extern "C" {
#endif

/* header file for lib_vwk.c */
unsigned long gidz_cube(int, int, int, unsigned);
unsigned long gidz_general(int, int, int, unsigned, unsigned);
void create_padded_map(double **, unsigned *, unsigned *, unsigned *, double *, double *, double *,
                       unsigned long *, double *, unsigned, unsigned, unsigned, double, double,
                       double, double, double, double, unsigned *);
void interpolate_map(double **, unsigned *, unsigned *, unsigned *, double *, double *, double *,
                     double, double, double, double *, unsigned, unsigned, unsigned,
                     double, double, double, double, double, double);
void project_map_lattice(double **, unsigned, unsigned, unsigned, double, double, double,
                         double, double, double, double *, unsigned, unsigned, unsigned,
                         double, double, double, double, double, double);
void shrink_margin(double **, unsigned *, unsigned *, unsigned *, double *, double *, double *,
                   unsigned long *, double *, unsigned, unsigned, unsigned, double, double, double,
                   double, double, double);
void shrink_to_sigma_factor(double **, unsigned *, double *, unsigned, double, double);
double calc_total(double *, unsigned long);
double calc_average(double *, unsigned long);
double calc_sigma(double *, unsigned long);
double calc_norm(double *, unsigned long);
double calc_gz_average(double *, unsigned long);
double calc_gz_sigma(double *, unsigned long);
double calc_gz_norm(double *, unsigned long);
double calc_max(double *, unsigned long);
double calc_min(double *, unsigned long);
void calc_map_info(double *, unsigned long, double *, double *, double *, double *);
void print_map_info(double *, unsigned long);
void threshold(double *, unsigned long, double);
void step_threshold(double *, unsigned long, double);
void boost_factor_high(double *, unsigned long, double, double);
void boost_power_high(double *, unsigned long, double, double);
void normalize(double *, unsigned long, double);
void floatshift(double *, unsigned long, double);
int clipped(double *, unsigned long, double, double);
void create_gaussian(double **, unsigned long *, unsigned *, double, double);
void create_identity(double **, unsigned long *, unsigned *);
void create_laplacian(double **, unsigned long *, unsigned *);

void relax_laplacian(double **, unsigned, unsigned, unsigned, unsigned *, double);

void convolve_kernel_inside(double **, double *, unsigned, unsigned, unsigned, double *, unsigned);
void convolve_kernel_inside_fast(double **, double *, unsigned, unsigned,
                                 unsigned, double *, unsigned, double, unsigned *);
void convolve_kernel_inside_erode(double **, double *, unsigned, unsigned, unsigned, double *, unsigned);
void convolve_kernel_outside(double **, unsigned *, unsigned *, unsigned *, double *, double *,
                             double *, double *, unsigned, unsigned, unsigned, double, double,
                             double, double, double, double, double *, unsigned);
int print_histogram(unsigned *, unsigned *, unsigned *, double **, int);
void print_diff_histogram(unsigned *, unsigned *, unsigned *, double **, int);

#ifdef __cplusplus
}
#endif

#endif
