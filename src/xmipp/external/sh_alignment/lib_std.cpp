/*********************************************************************
*                          L I B _ S T D                             *
**********************************************************************
* Library is part of the Situs package (c) Willy Wriggers, 1998-2003 *
* URL: situs.biomachina.org                                          *
**********************************************************************
*                                                                    *
* Auxiliary routines to read from stdin.                             *
*                                                                    *
**********************************************************************
* See legal statement for terms of distribution                      *
*********************************************************************/

#include "situs.h"
#include "lib_std.h"
#include "lib_err.h"

/* returns first integer from input stream and checks for EOF or errors */
int readln_int()
{
  int i, ch;
  const char *program = "lib_std";

  if (scanf("%d", &i) == EOF) {
    error_EOF(14010, program);
  }
  for (; ;) {
    ch = getchar();
    if (ch == EOF) {
      error_EOF(14020, program);
    }
    if (ch == '\n') break;
  }
  return i;
}

/* returns first double from input stream and checks for EOF or errors */
double readln_double()
{
  double ddx;
  int ch;
  const char *program = "lib_std";

  if (scanf("%le", &ddx) == EOF) {
    error_EOF(14030, program);
  }
  for (; ;) {
    ch = getchar();
    if (ch == EOF) {
      error_EOF(14040, program);
    }
    if (ch == '\n') break;
  }
  return ((double)ddx);
}

/* removes erroneously pasted spaces from file name string */
void removespaces(char *file_name, unsigned fl)
{
  int i;

  file_name[fl - 1] = '\0';

  /* remove leading space */
  for (;;) {
    if (file_name[0] == ' ') {
      for (i = 1; i < (int)fl; ++i) file_name[i - 1] = file_name[i];
      --fl;
    } else break;
  }

  /* remove trailing white space */
  for (i = 0; i < (int)fl; ++i) if (file_name[i] == '\n' || file_name[i] == ' ') {
      file_name[i] = '\0';
      break;
    }
}

/* returns first char from input stream and checks for EOF or errors */
char readln_char()
{
  char cx;
  int ch;
  const char *program = "lib_std";

  if (scanf("%c", &cx) == EOF) {
    error_EOF(14050, program);
  }
  for (; ;) {
    ch = getchar();
    if (ch == EOF) {
      error_EOF(14060, program);
    }
    if (ch == '\n') break;
  }
  return ((char)cx);
}


