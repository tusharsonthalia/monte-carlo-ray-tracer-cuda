#ifndef TIMER_H
#define TIMER_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#ifdef __cplusplus
    extern "C" {
#endif

typedef struct {
    double startTime;
    int64_t timeElapsed;
    int running;
} Timer;

/* Function Prototypes */
void error(char *message, int code);
int64_t diff_microseconds(double now, double start);
int64_t get_time(Timer *t);
Timer* timer_init(void);
void timer_start(Timer *t);
int64_t timer_pause(Timer *t);
void timer_resume(Timer *t);
int64_t timer_end(Timer *t);

#ifdef __cplusplus
}
#endif

#endif
