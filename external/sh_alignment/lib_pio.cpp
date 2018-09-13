/*********************************************************************
*                          L I B _ P I O                             *
**********************************************************************
* Library is part of the Situs package  URL: situs.biomachina.org    *
* (c) Willy Wriggers and Valerio Mariani, 1998-2012                  *
**********************************************************************
*                                                                    *
* Auxiliary program to read and write files in PDB format.           *
*                                                                    *
**********************************************************************
* Loosely based on some components of the DOWSER program -           *
* a product of UNC Computational Structural Biology group      *
* Authors: Li Zhang, Xinfu Xia, Jan Hermans, Univ. of North Carolina *
**********************************************************************
* See legal statement for terms of distribution                      *
*********************************************************************/

#include "situs.h"
#include "lib_pio.h"
#include "lib_err.h"
#include "lib_vec.h"

/* reads atomic entries from PDB file and assigns atom mass, no stdout */
void read_pdb_silent(char *file_name, unsigned *num_atoms, PDB **in_pdb)
{
  PDB *new_pdb;
  FILE *fin;
  char line[101];
  char field00[7], field01[6], field02[2], field03[3], field04[3],
       field05[2], field06[5], field08[2], field09[5],
       field10[2], field11[4], field12[9], field13[9], field14[9],
       field15[7], field16[7], field17[2], field18[4], field19[3],
       field20[5], field21[3], field22[3];
  char  recd_name[7]; /* 1 - 6 */
  const char *element_name[] = {
    "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE",
    "NA", "MG", "AL", "SI", "P", "S", "CL", "AR", "K", "CA",
    "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN"
  };
  float element_mass[30] = {
    1.008f , 4.003f, 6.940f, 9.012f, 10.810f, 12.011f, 14.007f, 16.000f, 18.998f, 20.179f,
    22.990f, 24.305f, 26.982f, 28.086f, 30.974f, 32.060f, 35.453f, 39.948f, 39.098f, 40.080f,
    44.956f, 47.880f, 50.942f, 51.996f, 54.938f, 55.847f, 58.933f, 58.710f, 63.546f, 65.380f
  };
  int frequent_element[5] = {5, 6, 7, 14, 15};
  unsigned i, j, k, line_width, atom_count;
  const char *program = "lib_pio";

  fin = fopen(file_name, "r");
  if (fin == NULL) {
    error_open_filename(12110, program, file_name);
  }

  /* count number of atom records */
  atom_count = 0;
  for (;;) {
    if (line != fgets(line, 100, fin)) error_fgets(program);
    if (feof(fin)) break;
    if (sscanf(line, "%6s", recd_name) != 1) continue;
    if (strcmp(recd_name, "ATOM") == 0 || strcmp(recd_name, "HETATM") == 0 || strcmp(recd_name, "LABEL") == 0) atom_count++;
  }
  rewind(fin);

  /* allocate memory */
  new_pdb = (PDB *) alloc_vect(atom_count, sizeof(PDB));
  /* now read fields */
  i = 0;
  for (;;) {
    if (line != fgets(line, 100, fin)) error_fgets(program);
    if (feof(fin)) break;
    if (sscanf(line, "%6s", recd_name) != 1) continue;
    if (strcmp(recd_name, "ATOM") == 0 || strcmp(recd_name, "HETATM") == 0 || strcmp(recd_name, "LABEL") == 0) {
      line_width = 80;
      for (j = 0; j < 80; j++) {
        if (line[j] == '\n') {
          line_width = j;
          break;
        }
      }
      for (j = line_width; j < 80; j++) {
        line[j] = ' ';
      }
      get_fld(line, 1, 6, field00);
      fld2s(field00, new_pdb[i].recd);
      get_fld(line, 7, 11, field01);
      fld2i(field01, 5, &(new_pdb[i].serial));
      get_fld(line, 12, 12, field02);
      get_fld(line, 13, 14, field03);
      fld2s(field03, new_pdb[i].type);
      get_fld(line, 15, 16, field04);
      fld2s(field04, new_pdb[i].loc);
      get_fld(line, 17, 17, field05);
      fld2s(field05, new_pdb[i].alt);
      get_fld(line, 18, 21, field06);
      fld2s(field06, new_pdb[i].res);
      get_fld(line, 22, 22, field08);
      fld2s(field08, new_pdb[i].chain);
      get_fld(line, 23, 26, field09);
      fld2i(field09, 4, &(new_pdb[i].seq));
      get_fld(line, 27, 27, field10);
      fld2s(field10, new_pdb[i].icode);
      get_fld(line, 28, 30, field11);
      get_fld(line, 31, 38, field12);
      new_pdb[i].x = (float) atof(field12);
      get_fld(line, 39, 46, field13);
      new_pdb[i].y = (float) atof(field13);
      get_fld(line, 47, 54, field14);
      new_pdb[i].z = (float) atof(field14);
      get_fld(line, 55, 60, field15);
      new_pdb[i].occupancy = (float) atof(field15);
      get_fld(line, 61, 66, field16);
      new_pdb[i].beta = (float) atof(field16);
      get_fld(line, 67, 67, field17);
      get_fld(line, 68, 70, field18);
      fld2i(field18, 3, &(new_pdb[i].footnote));
      get_fld(line, 71, 72, field19);
      get_fld(line, 73, 76, field20);
      fld2s(field20, new_pdb[i].segid);
      get_fld(line, 77, 78, field21);
      fld2s(field21, new_pdb[i].element);
      get_fld(line, 79, 80, field22);
      fld2s(field22, new_pdb[i].charge);

      /* check if codebook vector, assign zero mass */
      if ((strcmp(new_pdb[i].type, "QV") == 0 && strcmp(new_pdb[i].loc, "OL") == 0) || (strcmp(new_pdb[i].type, "QP") == 0 && strcmp(new_pdb[i].loc, "DB") == 0)) {
        new_pdb[i].weight = 0;
      }
      /* check if map density, assign mass from occupancy field */
      else if (strcmp(new_pdb[i].type, "DE") == 0 && strcmp(new_pdb[i].loc, "NS") == 0) {
        new_pdb[i].weight = new_pdb[i].occupancy;
      }
      /* looks like we have a standard atom */
      else { /* check if found in element table */
        for (j = 1; j < 30; ++j) {
          /* check if hydrogen */
          if (new_pdb[i].type[0] == 'H' || (new_pdb[i].type[0] == ' ' && new_pdb[i].type[1] == 'H')) {
            new_pdb[i].weight = element_mass[0];
            break;
          } /* note: above false hydrogen positives possible for Hg, Hf, Ho (these are very rare in proteins) */
          else if (strcmp(new_pdb[i].type, element_name[j]) == 0) {
            new_pdb[i].weight = element_mass[j];
            break;
          }
        }
        /* if this fails, use first letter only of most frequent elements */
        k = 0;
        if (j == 30) for (k = 0; k < 5; ++k) {
            if (strncmp(new_pdb[i].type, element_name[frequent_element[k]], 1) == 0) {
              new_pdb[i].weight = element_mass[frequent_element[k]];
              break;
            }
          }
        /* if this fails again, assign carbon mass and print warning */
        if (k == 6) {
          printf("lib_pio> Warning: Unable to identify atom type %s, assigning carbon mass.\n", new_pdb[i].type);
          new_pdb[i].weight = element_mass[5];
        }
      }
      ++i;
    }
  }
  fclose(fin);

  if (i != atom_count) {
    error_atom_count(12210, program, i, atom_count);
  }

  *num_atoms = atom_count;
  *in_pdb = new_pdb;
  return;
}


/* reads atomic entries from PDB file and provides some user feedback */
void read_pdb(char *file_name, unsigned *num_atoms, PDB **in_pdb)
{

  read_pdb_silent(file_name, num_atoms, in_pdb);
  printf("lib_pio> %d atoms read from file %s.\n", *num_atoms, file_name);
  return;
}

/* output PDB coordinate precision */
int coord_precision(double fcoord)
{
  if (fcoord < 9999.9995 && fcoord >= -999.9995) return 3;
  else if (fcoord < 99999.995 && fcoord >= -9999.995) return 2;
  else if (fcoord < 999999.95 && fcoord >= -99999.95) return 1;
  else if (fcoord < 99999999.5 && fcoord >= -9999999.5) return 0;
  else {
    fprintf(stderr, "lib_pio> Error: PDB coordinate out of range, PDB output aborted.\n");
    exit(0);
  }
}

/* writes atom entries to PDB file */
void write_pdb(char *filename, unsigned num_atoms, PDB *out_pdb)
{
  unsigned i, j, k, space_count;
  FILE *fout;
  const char *program = "lib_pio";

  fout = fopen(filename, "w");
  if (fout == NULL) {
    error_open_filename(12320, program, filename);
  }

  for (i = 0; i < num_atoms; i++) {
    space_count = 6 - strlen(out_pdb[i].recd);
    fprintf(fout, "%s", out_pdb[i].recd);
    print_space(fout, space_count);

    /* renumber atoms using 5 digits, ignore out_pdb[i].serial */
    if ((i + 1) / 100000 == 0) {
      k = i + 1;
      fprintf(fout, "%5d", k);
    } else {
      k = i + 1 - ((i + 1) / 100000) * 100000;
      fprintf(fout, "%05d", k);
    }
    print_space(fout, 1);

    space_count = 2 - strlen(out_pdb[i].type);
    print_space(fout, space_count);
    fprintf(fout, "%s", out_pdb[i].type);

    space_count = 2 - strlen(out_pdb[i].loc);
    fprintf(fout, "%s", out_pdb[i].loc);
    print_space(fout, space_count);

    print_space(fout, 1);

    j = strlen(out_pdb[i].res);
    space_count = 3 - j;
    if (j < 4) {
      print_space(fout, space_count);
      fprintf(fout, "%s", out_pdb[i].res);
      print_space(fout, 1);
    } else fprintf(fout, "%s", out_pdb[i].res);

    space_count = 1 - strlen(out_pdb[i].chain);
    print_space(fout, space_count);
    fprintf(fout, "%s", out_pdb[i].chain);

    fprintf(fout, "%4d", out_pdb[i].seq);

    space_count = 1 - strlen(out_pdb[i].icode);
    print_space(fout, space_count);
    fprintf(fout, "%s", out_pdb[i].icode);

    print_space(fout, 3);

    fprintf(fout, "%8.*f", coord_precision(out_pdb[i].x), out_pdb[i].x);
    fprintf(fout, "%8.*f", coord_precision(out_pdb[i].y), out_pdb[i].y);
    fprintf(fout, "%8.*f", coord_precision(out_pdb[i].z), out_pdb[i].z);
    fprintf(fout, "%6.2f", out_pdb[i].occupancy);
    fprintf(fout, "%6.2f", out_pdb[i].beta);

    print_space(fout, 1);

    fprintf(fout, "%3d", out_pdb[i].footnote);

    print_space(fout, 2);

    space_count = 4 - strlen(out_pdb[i].segid);
    fprintf(fout, "%s", out_pdb[i].segid);
    print_space(fout, space_count);

    space_count = 2 - strlen(out_pdb[i].element);
    print_space(fout, space_count);
    fprintf(fout, "%s", out_pdb[i].element);

    space_count = 2 - strlen(out_pdb[i].charge);
    fprintf(fout, "%s", out_pdb[i].charge);
    print_space(fout, space_count);

    fprintf(fout, "\n");
  }
  fclose(fout);
  return;
}

/* appends atom entries to PDB file */
void append_pdb(char *filename, unsigned num_atoms, PDB *out_pdb)
{
  unsigned i, j, k, space_count;
  FILE *fout;
  const char *program = "lib_pio";


  fout = fopen(filename, "a");
  if (fout == NULL) {
    error_open_filename(12420, program, filename);
  }

  for (i = 0; i < num_atoms; i++) {
    space_count = 6 - strlen(out_pdb[i].recd);
    fprintf(fout, "%s", out_pdb[i].recd);
    print_space(fout, space_count);

    /* renumber atoms using 5 digits */
    if (out_pdb[i].serial / 100000 == 0) {
      fprintf(fout, "%5d", out_pdb[i].serial);
    } else {
      k = out_pdb[i].serial - (out_pdb[i].serial / 100000) * 100000;
      fprintf(fout, "%05d", k);
    }
    print_space(fout, 1);

    space_count = 2 - strlen(out_pdb[i].type);
    print_space(fout, space_count);
    fprintf(fout, "%s", out_pdb[i].type);
    space_count = 2 - strlen(out_pdb[i].loc);
    fprintf(fout, "%s", out_pdb[i].loc);
    print_space(fout, space_count);
    print_space(fout, 1);

    j = strlen(out_pdb[i].res);
    space_count = 3 - j;
    if (j < 4) {
      print_space(fout, space_count);
      fprintf(fout, "%s", out_pdb[i].res);
      print_space(fout, 1);
    } else fprintf(fout, "%s", out_pdb[i].res);

    space_count = 1 - strlen(out_pdb[i].chain);
    print_space(fout, space_count);
    fprintf(fout, "%s", out_pdb[i].chain);
    fprintf(fout, "%4d", out_pdb[i].seq);
    space_count = 1 - strlen(out_pdb[i].icode);
    print_space(fout, space_count);
    fprintf(fout, "%s", out_pdb[i].icode);

    print_space(fout, 3);

    fprintf(fout, "%8.*f", coord_precision(out_pdb[i].x), out_pdb[i].x);
    fprintf(fout, "%8.*f", coord_precision(out_pdb[i].y), out_pdb[i].y);
    fprintf(fout, "%8.*f", coord_precision(out_pdb[i].z), out_pdb[i].z);
    fprintf(fout, "%6.2f", out_pdb[i].occupancy);
    fprintf(fout, "%6.2f", out_pdb[i].beta);

    print_space(fout, 1);

    fprintf(fout, "%3d", out_pdb[i].footnote);

    print_space(fout, 2);

    space_count = 4 - strlen(out_pdb[i].segid);
    fprintf(fout, "%s", out_pdb[i].segid);
    print_space(fout, space_count);

    space_count = 2 - strlen(out_pdb[i].element);
    print_space(fout, space_count);
    fprintf(fout, "%s", out_pdb[i].element);

    space_count = 2 - strlen(out_pdb[i].charge);
    fprintf(fout, "%s", out_pdb[i].charge);
    print_space(fout, space_count);

    fprintf(fout, "\n");
  }
  fclose(fout);
  return;
}

/* gets designated string between two markers i1 and i2 */
void get_fld(char *line, unsigned i1, unsigned i2, char *field)
{
  unsigned i;
  for (i = i1; i <= i2; i++) {
    field[i - i1] = line[i - 1];
  }
  field[i - i1] = '\0';
}

/* copies string and terminates empty field if necessary */
void fld2s(char *field, char *str)
{
  if (sscanf(field, "%s", str) < 1) {
    str[0] = '\0';
  }
}

/* attempts to extract integer from field or returns zero if empty */
void fld2i(char *field, unsigned n, int *num)
{
  unsigned i;
  for (i = 0; i < n; i++) {
    if (field[i] != ' ') {
      sscanf(field, "%d", num);
      return;
    }
  }
  if (i == n) {
    *num = 0;
    return;
  }
}

/* print whitespace to fout */
void print_space(FILE *fout, unsigned space_count)
{
  unsigned j;
  for (j = 0; j < space_count; j++)
    fprintf(fout, " ");
}


