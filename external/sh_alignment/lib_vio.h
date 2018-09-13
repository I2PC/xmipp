#ifndef __SITUS_LIB_VIO
#define __SITUS_LIB_VIO

#ifdef __cplusplus
extern "C" {
#endif

/* header file for lib_vio.c */
void read_vol(char *, double *, double *, double *, double *, unsigned *, unsigned *, unsigned *, double **);
void write_vol(char *, double, double, double, double, unsigned, unsigned, unsigned, double *);
void read_situs(char *, double *, double *, double *, double *, unsigned *, unsigned *, unsigned *, double **);

void write_situs(char *, double, double, double, double, unsigned, unsigned, unsigned, double *);
void read_ascii(char *, unsigned long, double **);
void read_xplor(char *, int *, int *, double *, double *, double *, double *, double *, double *, int *, int *, int *, unsigned *, unsigned *, unsigned *, double **);
void xplor_skip_to_number(FILE **, char **);
void read_mrc(char *, int *, int *, int *, unsigned *, unsigned *, unsigned *, int *, int *, int *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double **);
void read_spider(char *, unsigned *, unsigned *, unsigned *, double **);
void dump_binary_and_exit(char *, char *, int);
int read_float(float *, FILE *, int);
int read_short_float(float *, FILE *, int);
int read_float_empty(FILE *);
int read_char_float(float *, FILE *);
int read_char(char *, FILE *);
int read_int(int *, FILE *, int);
unsigned long count_floats(FILE **);
int test_registration(float, float, float, float);
int test_situs(char *);
int have_situs_suffix(char *);
int test_situs_header_and_suffix(char *);
int test_mrc(char *, int);
int test_spider(char *, int);
unsigned long permuted_index(int, unsigned long, unsigned, unsigned, unsigned);
void permute_map(int, unsigned, unsigned, unsigned, unsigned *, unsigned *, unsigned *, int, int, int, int *, int *, int *, double *, double **);
void permute_dimensions(int, unsigned, unsigned, unsigned, unsigned *, unsigned *, unsigned *, int, int, int, int *, int *, int *);
void permute_print_info(int, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, int, int, int, int, int, int);
int set_origin_get_mode(int, int, int, double, double, double, double, double, double, double *, double *, double *);
void assert_cubic_map(int, int, double, double, double, double, double, double, unsigned, unsigned, unsigned, int, int, int, double, double, double, double *, double *, double *, double *, unsigned *, unsigned *, unsigned *, double **);
void interpolate_skewed_map_to_cubic(double **, unsigned *, unsigned *, unsigned *, double *, double *, double *, double *, double *, unsigned, unsigned, unsigned, int, int, int, double, double, double, double, double, double, double, double, double);
void write_xplor(char *, double, double, double, double, unsigned, unsigned, unsigned, double *);
void write_mrc(int, char *, double, double, double, double, unsigned, unsigned, unsigned, double *);
void write_spider(char *, double, double, double, double, unsigned, unsigned, unsigned, double *);

#ifdef __cplusplus
}
#endif

#endif
