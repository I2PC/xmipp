/*********************************************************************
*                           L I B _ E R R                            *
**********************************************************************
* Library is part of the Situs package URL: situs.biomachina.org     *
* (c) Paul Boyle and Mirabela Rusu, 2004-2015                        *
**********************************************************************
*                                                                    *
*   Auxiliary program for producing error messages to the user       *
*                                                                    *
**********************************************************************
* See legal statement for terms of distribution                      *
*********************************************************************/

#include "situs.h"
#include "lib_err.h"


void error_IO_files_5(char *program, char *file, char *file1, char *file2)
{
  fprintf(stderr, "%s> Usage: %s inputfile (%s format) inputfile2 (%s format) outputfile (%s format) \n", program, program, file, file1, file2);
}
void error_IO_files_4(char *program, char *file, char *file1, char *file2)
{
  fprintf(stderr, "%s> Usage: %s inputfile (%s format) [optional: inputfile (%s format)] outputfile (%s format) \n", program, program, file, file1, file2);
}
void error_IO_files_3(char *program, char *file, char *file1)
{
  fprintf(stderr, "%s> Usage: %s inputfile (%s format) outputfile (%s format)\n", program, program, file, file1);
}
void error_IO_files_2(char *program, char *file, char *file1)
{
  fprintf(stderr, "%s> Usage: %s inputfile (%s format) outputfile (%s format) \n", program, program, file, file1);
}
void error_IO_files_1(char *program, char *file, char *file1)
{
  fprintf(stderr, "%s> Usage: %s inputfile (%s format) [optional: outputfile (%s format)] \n", program, program, file, file1);
}
void error_IO_files_0(char *program, char *file)
{
  fprintf(stderr, "%s> Usage: %s inputfile (%s format) \n", program, program, file);
}
void error_fgets(const char *program)
{
  fprintf(stderr, "%s> Unspecified error while reading from stream\n", program);
}
void error_fscanf(const char *program, const char * file)
{
  fprintf(stderr, "%s> Unspecified error while reading from file %s\n", program, file);
}
void error_memory_allocation(int error_number, const char *program)
{
  fprintf(stderr, "%s> Error: Unable to satisfy memory allocation request [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_start_vectors(int error_number, char *program, char *argv2, char *argv1)
{
  fprintf(stderr, "%s> Error: Start vectors from file %s are not compatible with map from file %s. [e.c. %d]\n", program, argv2, argv1, error_number);
  exit(error_number);
}

void error_option(int error_number, const char *program)
{
  fprintf(stderr, "%s> Error: Unable to identify option [e.c. %d]\n", program, error_number);
  exit(error_number);
}

void error_open_filename(int error_number, const char *program, char *argv)
{
  fprintf(stderr, "%s> Error: Can't open file! %s  [e.c. %d]\n", program, argv, error_number);
  exit(error_number);
}

void error_read_filename(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: Can't read filename [e.c. %d]\n", program, error_number);
  exit(error_number);
}

void error_density(int error_number, const char *program)
{
  fprintf(stderr, "%s> Error: No positive density found [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_no_density(int error_number, char *program, int i)
{
  fprintf(stderr, "%s> Error: Voronoi cell %d contains no density [e.c. %d]\n", program, i, error_number);
}

void error_reading_constraints(int error_number, char *program, int numshake, char *con_file)
{
  fprintf(stderr, "%s> Error: Can't complete reading %d. constraint entry in file %s [e.c. %d]\n", program, numshake, con_file, error_number);
  exit(error_number);
}

void error_out_of_index(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: element index out of range [e.c. %d]\n", program, error_number);
  exit(error_number);
}

void error_EOF(int error_number, const char *program)
{
  fprintf(stderr, "%s> Error: EOF while reading input [e.c. %d]\n", program, error_number);
  exit(error_number);
}

void error_number_vertices(int error_number, char *program, int NUM_VERTEX)
{
  printf("%s> Error: Too many vertices; max is %d. Increase NUM_VERTEX [e.c. %d]\n", program, NUM_VERTEX, error_number);
  exit(error_number);
}

void error_no_volume(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: No volume found [e.c. %d]\n", program, error_number);
  exit(error_number);
}

void error_in_allocation(char *program)
{
  printf("%s> Bye bye!\n", program);
}
void error_number_columns(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: number of columns must be larger than 0 [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_number_rows(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: number of rows must be larger than 0 [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_number_sections(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: number of sections must be larger than 0 [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_number_spacing(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: grid spacing must be larger than 0 [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_unreadable_file_short(int error_number, const char *program, const char *filename)
{
  fprintf(stderr, "%s> Error: file %s is too short or data is unreadable, incorrect format? [e.c. %d]\n", program, filename, error_number);
  exit(error_number);
}
void error_unreadable_file_long(int error_number, const char *program, const char *filename)
{
  fprintf(stderr, "%s> Error: file %s is too long or data is unreadable, incorrect format? [e.c. %d]\n", program, filename, error_number);
  exit(error_number);
}
void error_xplor_file_indexing(int error_number, const char *program)
{
  fprintf(stderr, "%s> Error: Can't read X-PLOR indexing [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_xplor_file_unit_cell(int error_number, const char *program)
{
  fprintf(stderr, "%s> Error: Can't read X-PLOR unit cell info [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_xplor_file_map_section(int error_number, const char *program)
{
  fprintf(stderr, "%s> Error: Can't read X-PLOR map section number [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_xplor_file_map_section_number(int error_number, const char *program)
{
  fprintf(stderr, "%s> Error: X-PLOR map section number and index don't match [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_EOF_ZYX_mode(int error_number, const char *program)
{
  fprintf(stderr, "%s> Error: EOF or error occurred before \"ZYX\" mode specifier was found [e.c. %d]\n", program, error_number);
  fprintf(stderr, "%s> Check if X-PLOR map is in ZYX mode\n", program);
  exit(error_number);
}
void error_xplor_maker(const char *program)
{
  fprintf(stderr, "%s> Warning: Can't find '-9999' X-PLOR map end marker.\n", program);
}
void error_file_convert(int error_number, const char *program, const char *filename)
{
  fprintf(stderr, "%s> Error: Unable to convert all data from file %s [e.c. %d]\n", program, filename, error_number);
  exit(error_number);
}
void error_file_header(int error_number, const char *program, const char *filename)
{
  fprintf(stderr, "%s> Error: Unable to read header of file %s [e.c. %d]\n", program, filename, error_number);
  exit(error_number);
}
void error_file_float_mode(int error_number, const char *program, const char *filename)
{
  fprintf(stderr, "%s> Error: Float mode of file %s is not supported. Mode must be 0 (1-byte char), 1 (2-byte float), or 2 (4-byte float). Sorry. \n", program, filename);
  exit(error_number);
}
void error_axis_assignment(int error_number, const char *program)
{
  fprintf(stderr, "%s> Error: Unable to assign axes (variables MAPC,MAPR,MAPS) [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_skew_transform(char *program)
{
  fprintf(stderr, "%s> Warning: Skew transformations are not supported (variable LSKFLG)\n", program);
}
void error_spider_header(int error_number, const char *program)
{
  fprintf(stderr, "%s> Error: SPIDER header length is not compatible with map size [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_index_conversion(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: Unable to identify index conversion mode [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_divide_zero(int error_number, const char *program)
{
  fprintf(stderr, "%s> Error: dividing by zero [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_sqrt_negative(int error_number, const char *program)
{
  fprintf(stderr, "%s> Error: sqrt argument negative [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_write_filename(int error_number, const char *program)
{
  fprintf(stderr, "%s> Error: Can't write to file [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_map_not_square(int error_number, char *program, int extx, int exty)
{
  fprintf(stderr, "%s> Error: Map z-sections are not square (%d x %d), map is apparently not helical. [e.c. 34010]\n", program, extx, exty);
  fprintf(stderr, "%s> Check map symmetry or create square sections with voledit. \n", program);
  exit(error_number);
}
void error_voxel_size(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: voxel size must be > 0 [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_negative_euler(int error_number, const char *program)
{
  fprintf(stderr, "%s> Error: negative number of Euler angle steps [e.c. 15080]\n", program);
  exit(error_number);
}
void error_eigenvec_not_converged(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: Eigenvector algorithm did not converge [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_atom_count(int error_number, const char *program, int i, int atom_count)
{
  fprintf(stderr, "%s> Error: Inconsistent atom count %d %d [e.c. %d]\n", program, i, atom_count, error_number);
  exit(error_number);
}
void error_no_bounding(int error_number, const char *program, const char *shape)
{
  fprintf(stderr, "%s> Error: no bounding %s found [e.c. %d]\n", program, shape, error_number);
  exit(error_number);
}
void error_underflow(int error_number, const char *program)
{
  fprintf(stderr, "%s> Error: interpolation output map size underflow [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_threshold(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: Threshold value negative [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_normalize(int error_number, const char *program)
{
  fprintf(stderr, "%s> Error: Normalization by zero [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_kernels(int error_number, const char *program)
{
  fprintf(stderr, "%s> Error: Input and output kernels not compatible [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_kernel_size(int error_number, const char *program, unsigned kernal_size)
{
  fprintf(stderr, "%s> Error: Kernel size %d must be a positive odd number [e.c. %d]\n", program, kernal_size, error_number);
  exit(error_number);
}
void error_lattice_smoothing(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: lattice smoothing exceeds kernel size [e.c. %d]\n", program, error_number);
  exit(error_number);
}


void error_codebook_vectors(int error_number, char *program, char *file1, char *file3)
{
  fprintf(stderr, "%s> Error: Number of codebook vectors in files %s and %s are not compatible [e.c. %d]\n", program, file1, file3, error_number);
  exit(error_number);
}
void error_vector_pairs(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: At least three pairs of vectors are required [e.c. %d] \n", program, error_number);
  exit(error_number);
}
void error_protein_data(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: Protein data out of bounds [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_alpha_carbons(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: No alpha carbons found [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_number_fits(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: g_numkeep must be larger than 12 [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_kabsch(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: Kabsch algorithm returned negative mean-square deviation [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_codebook_range(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: number of codebook vectors out of range [e.c. %d]\n", program, error_number);
  exit(error_number);
}

void error_files_incompatible(int error_number, char *program, char *file1, char *file2)
{
  fprintf(stderr, "%s> Error: Files %s and %s are incompatible [e.c. %d]\n", program, file1, file2, error_number);
  exit(error_number);
}
void error_symmetry_option(char *program)
{
  fprintf(stderr, "%s> Error: Unknown symmetry type\n", program);
  exit(1);
}

void error_resolution(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: High resolution map is empty [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_extends_beyond(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: Initially placed structure extends beyond map [e.c. %d]\n", program, error_number);
  fprintf(stderr, "%s> Suggestion: Try larger -sizef option.\n", program);
  exit(error_number);
}
void error_map_dimensions(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: Map intervals must be odd for all dimensions \n", program);
  exit(error_number);
}
void error_resolution_range(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: Resolution out of range [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_anisotropy_range(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: Anisotropy out of range [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void  error_euler_sampling(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: Euler angle sampling step too small [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_euler_below_start(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: Euler angle range end value below start value [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_euler_below_neg_360(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: Euler angle range start value below -360 [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_euler_above_pos_360(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: Euler angle range start value above +360 [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_psi_euler_range_above_360(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: First (psi) Euler range exceeds 360 [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_theta_euler_range_above_180(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: Second (theta) Euler range exceeds 180 [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_phi_euler_range_above_360(int error_number, char *program)
{
  fprintf(stderr, "%s> Error: Third (phi) Euler range exceeds 360 [e.c. %d]\n", program, error_number);
  exit(error_number);
}
void error_sba(int error_number, char *err_string)
{
  fprintf(stderr, "lib_sba> %s [e.c. %d]\n", err_string, error_number);
  exit(error_number);
}
