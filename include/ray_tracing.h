#ifndef RAY_TRACING_H
#define RAY_TRACING_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Precision toggle ────────────────────────────────────────────────── */

#ifdef USE_FLOAT
    typedef float real;
    #define PI 3.141592653589793f
    #define FLOAT_FLAG 1
#else
    typedef double real;
    #define PI 3.141592653589793
    #define FLOAT_FLAG 0
#endif

/* ── Data Structures ─────────────────────────────────────────────────── */

typedef struct Window_ {
    real *data;
    int rows;
    int cols;
    int size;
    int global_size;
} Window;

typedef struct Vector_ {
    real x;
    real y;
    real z;
} Vector;

/* Scratch vectors used throughout the ray-tracing pipeline */
typedef struct Vectors_ {
    Vector V; /* ray direction          */
    Vector W; /* window intersection    */
    Vector C; /* sphere centre          */
    Vector I; /* intersection point     */
    Vector N; /* surface normal         */
    Vector S; /* direction toward light */
    Vector L; /* light source position  */
} Vectors;

typedef struct Input_ {
    Vector light_source;
    real window_position;
    real window_size;
    Vector sphere_position;
    real sphere_radius;
    int window_resolution;
    int rows;
    int cols;
    long n_rays;
} Input;

/* ── Memory helpers ──────────────────────────────────────────────────── */

static inline Window *allocate_window(const int rows, const int cols,
                                      const int N) {
    Window *grid = (Window *)malloc(sizeof(Window));

    grid->rows = rows;
    grid->cols = cols;
    grid->size = rows * cols;
    grid->global_size = N;

    grid->data = (real *)calloc(grid->size, sizeof(real));

    return grid;
}

static inline void free_window(Window *grid) {
    free(grid->data);
    free(grid);
}

static inline Vectors *allocate_vectors(Input *args) {
    Vectors *vectors = (Vectors *)malloc(sizeof(Vectors));

    vectors->C = args->sphere_position;
    vectors->L = args->light_source;

    return vectors;
}

static inline void free_vectors(Vectors *vectors) {
    free(vectors);
}

/* ── Output ──────────────────────────────────────────────────────────── */

static inline void write_grid(Window *grid) {
    char filename[256];
    snprintf(filename, sizeof(filename), "./static/data/grid.dat");

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error: Unable to save grid data.\n");
        exit(-1);
    }

    /* float flag — tells the visualiser whether to read float32 or float64 */
    int float_flag = FLOAT_FLAG;
    fwrite(&float_flag, sizeof(int), 1, fp);

    fwrite(&(grid->global_size), sizeof(int), 1, fp);
    fwrite(&(grid->rows), sizeof(int), 1, fp);
    fwrite(&(grid->cols), sizeof(int), 1, fp);
    fwrite(grid->data, sizeof(real), grid->size, fp);

    fclose(fp);
}

/* ── Vector arithmetic (CPU, inlined) ────────────────────────────────── */

#ifndef __CUDA_ARCH__

static inline void add_vectors(const Vector *a, const Vector *b, Vector *out) {
    out->x = a->x + b->x;
    out->y = a->y + b->y;
    out->z = a->z + b->z;
}

static inline void subtract_vectors(const Vector *a, const Vector *b,
                                    Vector *out) {
    out->x = a->x - b->x;
    out->y = a->y - b->y;
    out->z = a->z - b->z;
}

static inline real dot_vectors(const Vector *a, const Vector *b) {
    return (a->x * b->x) + (a->y * b->y) + (a->z * b->z);
}

static inline real norm_vectors(const Vector *a) {
    return sqrt(dot_vectors(a, a));
}

static inline void multiply_vectors(const real t, const Vector *a,
                                    Vector *out) {
    out->x = a->x * t;
    out->y = a->y * t;
    out->z = a->z * t;
}

static inline void divide_vectors(const real t, const Vector *a, Vector *out) {
    const real val = 1.0 / t;
    out->x = a->x * val;
    out->y = a->y * val;
    out->z = a->z * val;
}

static inline real max_(const real a, const real b) {
    if (a < b) {
        return b;
    }

    return a;
}

#endif /* __CUDA_ARCH__ */

#ifdef __cplusplus
}
#endif

#endif /* RAY_TRACING_H */
