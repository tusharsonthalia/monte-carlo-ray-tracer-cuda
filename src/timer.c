#include "timer.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void error(char *message, int code) {
  printf("%s", message);
  exit(code);
}

int64_t diff_microseconds(double now, double start) {
  return (int64_t)((now - start) * 1000000.0);
}

inline int64_t get_time(Timer *t) { return t->timeElapsed; }

Timer *timer_init(void) {
  Timer *t = malloc(sizeof(Timer));

  if (!t) {
    error("Error Occurred while creating the timer.", 1);
  }

  t->running = 0;
  t->timeElapsed = 0.0;
  return t;
}

/* Start starts or restarts the timer (resets elapsed time) */
void timer_start(Timer *t) {
  t->timeElapsed = 0;
  t->startTime = omp_get_wtime();
  t->running = 1;
}

/* Pause pauses the timer, accumulating elapsed time so far */
int64_t timer_pause(Timer *t) {
  if (t->running == 1) {
    t->timeElapsed += diff_microseconds(omp_get_wtime(), t->startTime);
    t->running = 0;
  }

  return get_time(t);
}

/* Resume resumes the timer from the paused state */
void timer_resume(Timer *t) {
  if (t->running == 0) {
    t->startTime = omp_get_wtime();
    t->running = 1;
  }
}

/* End stops the timer and returns the total elapsed time in microseconds */
int64_t timer_end(Timer *t) {
  if (t->running == 1) {
    t->timeElapsed += diff_microseconds(omp_get_wtime(), t->startTime);
    t->running = 0;
  }

  return get_time(t);
}
