#ifndef __SITUS_LIB_TIM
#define __SITUS_LIB_TIM

#ifdef __cplusplus
extern "C" {
#endif

/* header file for lib_tim.c */
typedef struct timeval the_time;

the_time get_the_time(void);
the_time time_diff(the_time, the_time);
char *smart_sprint_time(double);
double time_to_sec(the_time);

#ifdef __cplusplus
}
#endif

#endif

