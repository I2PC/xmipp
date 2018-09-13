/*********************************************************************
*                           L I B _ T I M                            *
**********************************************************************
* Library is part of the Situs package URL: situs.biomachina.org     *
**********************************************************************
*                                                                    *
* Various timing routines, adapted from FFTW library.                *
*                                                                    *
**********************************************************************
* See legal statement for terms of distribution                      *
*********************************************************************/

#include "situs.h"
#include "lib_tim.h"


/*====================================================================*/
/* timing function modified from FFTW */
the_time get_the_time(void)
{
  the_time tv;
  gettimeofday(&tv, 0);
  return tv;
}

/*====================================================================*/
/* timing function modified from FFTW */
the_time time_diff(the_time t1, the_time t2)
{
  the_time diff;
  diff.tv_sec = t1.tv_sec - t2.tv_sec;
  diff.tv_usec = t1.tv_usec - t2.tv_usec;

  /* normalize */
  while (diff.tv_usec < 0) {
    diff.tv_usec += 1000000L;
    diff.tv_sec -= 1;
  }
  return diff;
}

/*====================================================================*/
/* timing function modified from FFTW */
char *smart_sprint_time(double x)
{
  static char buf[128];
  if (x < 1.0E-6)
    sprintf(buf, "%f ns", x * 1.0E9);
  else if (x < 1.0E-3)
    sprintf(buf, "%f us", x * 1.0E6);
  else if (x < 1.0)
    sprintf(buf, "%f ms", x * 1.0E3);
  else if (x < 60.0)
    sprintf(buf, "%f s", x);
  else sprintf(buf, "%d h %d m %d s", (int) floor(x / 3600.0), ((int) x / 60 % 60), ((int)x % 60));
  return buf;
}

/*====================================================================*/
/* timing function modified from FFTW */
double time_to_sec(the_time t)
{
  return (t.tv_sec + t.tv_usec * 1.0E-6);
}
