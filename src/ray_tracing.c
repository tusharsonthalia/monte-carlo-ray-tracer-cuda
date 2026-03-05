/*
 * ray_tracing.c — OpenMP shared-memory ray tracer
 *
 * Each OpenMP thread gets its own RNG instance and a private copy of the
 * output grid.  After the parallel region the per-thread grids are reduced
 * into the global grid.  This avoids any atomic contention on grid writes.
 */

#include "ray_tracing.h"
#include "rng.h"
#include "timer.h"
#include <omp.h>

/* ── Input ───────────────────────────────────────────────────────────── */

Input *read_input_args(const int nrays, const int ngrid) {
    Input *args = malloc(sizeof(Input));

    args->light_source.x = 4;
    args->light_source.y = 4;
    args->light_source.z = -1;

    args->window_position = 2;
    args->window_size = 2;

    args->sphere_position.x = 0;
    args->sphere_position.y = 12;
    args->sphere_position.z = 0;

    args->sphere_radius = 6;
    args->window_resolution = ngrid;
    args->n_rays = nrays;

    args->rows = args->window_resolution;
    args->cols = args->window_resolution;

    return args;
}

/* ── Ray sampling (rejection method) ─────────────────────────────────
 *
 * Generates a random unit direction V, projects it to the window, and
 * checks (1) the window bounds and (2) sphere intersection.  Loops until
 * both pass.  Returns the parametric distance t along V to the hit point.
 */
static inline real sample_ray_vector(Vector *V, Vector *W, const Vector *C,
                                     const real cc, const real W_y,
                                     const real W_max_squared,
                                     const real R_squared, long *counter,
                                     RNG *rng) {
    real vc, t_inner;
    while (1) {
        const real phi = next_double(rng) * PI;
        const real cos_theta = (next_double(rng) * 2.0) - 1.0;
        *counter += 2;

        const real sin_theta = sqrt(1.0 - (cos_theta * cos_theta));
        V->x = sin_theta * cos(phi);
        V->y = sin_theta * sin(phi);
        V->z = cos_theta;

        /* project ray onto the window plane at y = W_y */
        multiply_vectors(W_y / V->y, V, W);

        if (((W->x * W->x) >= W_max_squared) || ((W->z * W->z) >= W_max_squared)) {
            continue;
        }

        vc = dot_vectors(V, C);

        t_inner = (vc * vc) + R_squared - cc;

        if (t_inner >= 0) {
            return vc - sqrt(t_inner);
        }
    }
}

/* ── Main ────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {

    char *usage_string = "./ray_tracing nrays ngrid threads\n";
    if (argc != 4) {
        printf(usage_string);
        return 0;
    }

    int nrays = atoi(argv[1]);
    int ngrid = atoi(argv[2]);
    int threads = atoi(argv[3]);

    omp_set_num_threads(threads);

    Input *args = read_input_args(nrays, ngrid);
    Window *grid = allocate_window(args->rows, args->cols, args->window_resolution);

    /* each thread accumulates into its own slab to avoid false sharing */
    real *local_grids = calloc(grid->size * threads, sizeof(real));
    long counter = 0;

    Timer *t = timer_init();
    timer_start(t);

    #pragma omp parallel default(none) shared(args, local_grids, grid) reduction(+:counter)
    {
        const int tid = omp_get_thread_num();

        RNG *rng = new_rng(4238811 * tid);

        Vectors *vectors = allocate_vectors(args);
        real *local_grid = &local_grids[grid->size * tid];

        const int window_resolution = args->window_resolution;
        const real window_position = args->window_position;
        const real window_size = args->window_size;
        const real window_size_squared = window_size * window_size;
        const real sphere_radius_squared = args->sphere_radius * args->sphere_radius;
        const real window_position_factor = window_resolution / (2.0 * window_size);
        const real cc = dot_vectors(&vectors->C, &vectors->C);

        #pragma omp for
        for (int i = 0; i < args->n_rays; i++) {
            const real t = sample_ray_vector(&vectors->V, &vectors->W, &vectors->C, cc,
                                             window_position, window_size_squared,
                                             sphere_radius_squared, &counter, rng);

            /* I = t * V (intersection point on the sphere) */
            multiply_vectors(t, &vectors->V, &vectors->I);

            /* N = normalise(I - C) (outward surface normal) */
            subtract_vectors(&vectors->I, &vectors->C, &vectors->N);
            divide_vectors(norm_vectors(&vectors->N), &vectors->N, &vectors->N);

            /* S = normalise(L - I) (direction toward the light) */
            subtract_vectors(&vectors->L, &vectors->I, &vectors->S);
            divide_vectors(norm_vectors(&vectors->S), &vectors->S, &vectors->S);

            /* Lambertian brightness: b = max(0, S · N) */
            const real b = max_(0, dot_vectors(&vectors->S, &vectors->N));
            const int ix = (vectors->W.x + window_size) * window_position_factor;
            const int iz = (vectors->W.z + window_size) * window_position_factor;

            local_grid[ix * grid->cols + iz] += b;
        }

        free_rng(rng);
        free_vectors(vectors);
    }

    /* reduce per-thread grids into the global grid */
    for (int t = 0; t < threads; t++) {
        for (int i = 0; i < grid->size; i++) {
            grid->data[i] += local_grids[(t * grid->size + i)];
        }
    }

    const int64_t time = timer_end(t);
    printf("Time taken for %ld rays: %.2f secs with Number of rejected rays: %ld\n",
           args->n_rays, time / 1000000.0, counter);

    write_grid(grid);

    free(t);
    free(args);
    free(local_grids);
    free_window(grid);

    return 0;
}
