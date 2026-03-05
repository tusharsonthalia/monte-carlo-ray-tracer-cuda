/*
 * ray_tracing_mpi.cu — Distributed multi-GPU ray tracer (MPI + CUDA)
 *
 * Each MPI rank owns one GPU.  Rays are evenly split across ranks; the
 * last rank absorbs any remainder.  After all local kernels finish, the
 * per-rank grids are summed via MPI_Reduce and rank 0 writes the result.
 */

#include "ray_tracing.h"
#include "timer.h"
#include <assert.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

/* ── MPI + CUDA precision macros ─────────────────────────────────────── */

#ifdef USE_FLOAT
    #define MPI_REAL_CUSTOM_DTYPE MPI_FLOAT
    #define R_SQRT rsqrtf
    #define SQRT sqrtf
    #define COS cosf
    #define SIN sinf
    #define CURAND_UNIFORM curand_uniform
#else
    #define MPI_REAL_CUSTOM_DTYPE MPI_DOUBLE
    #define R_SQRT rsqrt
    #define SQRT sqrt
    #define COS cos
    #define SIN sin
    #define CURAND_UNIFORM curand_uniform_double
#endif

/* ── Input (each rank gets its share of total rays) ──────────────────── */

Input *read_input_args(const long nrays, const int ngrid, const int rank, const int nprocs) {
    Input *args = (Input *)malloc(sizeof(Input));

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
    args->n_rays = nrays / nprocs;

    /* last rank picks up the remainder */
    if (rank == nprocs - 1) {
        args->n_rays += nrays % nprocs;
    }

    args->rows = args->window_resolution;
    args->cols = args->window_resolution;

    return args;
}

void cudaCheckError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

/* ── Device vector math ──────────────────────────────────────────────── */

__device__ __forceinline__ void add_vectors(const Vector *a, const Vector *b, Vector *out) {
    out->x = a->x + b->x;
    out->y = a->y + b->y;
    out->z = a->z + b->z;
}

__device__ __forceinline__ void subtract_vectors(const Vector *a, const Vector *b, Vector *out) {
    out->x = a->x - b->x;
    out->y = a->y - b->y;
    out->z = a->z - b->z;
}

__device__ __forceinline__ real dot_vectors(const Vector *a, const Vector *b) {
    return (a->x * b->x) + (a->y * b->y) + (a->z * b->z);
}

__device__ __forceinline__ void multiply_vectors(const real t, const Vector *a, Vector *out) {
    out->x = a->x * t;
    out->y = a->y * t;
    out->z = a->z * t;
}

__device__ __forceinline__ void norm_inplace(const Vector *a, Vector *out) {
    const real val = R_SQRT(dot_vectors(a, a));
    out->x = a->x * val;
    out->y = a->y * val;
    out->z = a->z * val;
}

__device__ __forceinline__ real max_(const real a, const real b) {
    if (a < b) {
        return b;
    }

    return a;
}

/* ── Ray sampling — same rejection logic as the single-GPU version ─── */

__device__ __forceinline__ real sample_ray_vector(
    Vector *V, Vector *W, const Vector *C,
    const real W_y, const real W_max_squared, const real R_2_cc,
    unsigned long long *counter, curandState *rng
) {
    real vc, t_inner;

    while (1) {
        const real phi = CURAND_UNIFORM(rng) * PI;
        const real cos_theta = (CURAND_UNIFORM(rng) * 2.0) - 1.0;
        const real sin_theta = SQRT(1.0 - (cos_theta * cos_theta));
        *counter += 2;

        V->x = sin_theta * COS(phi);
        V->y = sin_theta * SIN(phi);
        V->z = cos_theta;

        multiply_vectors(W_y / V->y, V, W);

        vc = dot_vectors(V, C);

        t_inner = (vc * vc) + R_2_cc;

        if (((W->x * W->x) >= W_max_squared) || ((W->z * W->z) >= W_max_squared) || (t_inner < 0)) {
            continue;
        }

        return vc - SQRT(t_inner);
    }
}

/* ── CUDA kernel ─────────────────────────────────────────────────────
 *
 * The `rank` parameter offsets the curand subsequence so that different
 * MPI ranks produce independent random streams even with the same seed.
 */
__global__ void ray_tracer(
    real *grid, int rows, int cols, long n_rays, Vector C,
    Vector L, real window_position, real window_size_squared,
    real window_position_factor, real sphere_radius_squared_cc,
    unsigned long long *d_counter, real window_size,
    const int stride, const int rank
) {
    long tid = (blockIdx.x * blockDim.x + threadIdx.x) * stride;

    extern __shared__ unsigned long long sample_counter[];
    real t;

    sample_counter[threadIdx.x] = 0;

    curandState rng;
    curand_init(4238811ULL, tid, (unsigned long long)rank, &rng);

    for (int i = 0; i < stride; i++) {
        if ((tid + i) >= n_rays) {
            break;
        }
        Vector V, W, I, N, S;

        t = sample_ray_vector(&V, &W, &C, window_position,
                              window_size_squared, sphere_radius_squared_cc,
                              &sample_counter[threadIdx.x], &rng);

        multiply_vectors(t, &V, &I);

        subtract_vectors(&I, &C, &N);
        norm_inplace(&N, &N);

        subtract_vectors(&L, &I, &S);
        norm_inplace(&S, &S);

        const real b = max_(0, dot_vectors(&S, &N));
        const int ix = (W.x + window_size) * window_position_factor;
        const int iz = (W.z + window_size) * window_position_factor;

        atomicAdd(&grid[ix * cols + iz], b);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned long long local_samples = 0;
        for (int i = 0; i < blockDim.x; i++) {
            local_samples += sample_counter[i];
        }
        atomicAdd(d_counter, local_samples);
    }

    __syncthreads();
}

/* ── Main ────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    int nprocs;
    int rank;
    int stat;

    MPI_Init(&argc, &argv);

    stat = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    assert(stat == MPI_SUCCESS);

    stat = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    assert(stat == MPI_SUCCESS);

    /* rank 0 parses CLI args; everyone else receives via broadcast */
    long input[4];

    if (rank == 0) {
        char *usage_string = (char *)"./ray_tracing nrays ngrid nblocks threads\n";
        if (argc != 5) {
            printf("%s", usage_string);
            return 1;
        }

        input[0] = atol(argv[1]);
        input[1] = atol(argv[2]);
        input[2] = atol(argv[3]);
        input[3] = atol(argv[4]);
    }

    MPI_Bcast(&input, 4, MPI_LONG, 0, MPI_COMM_WORLD);

    long nrays = input[0];
    int ngrid = input[1];
    int blocks = input[2];
    int threads = input[3];

    Input *args = read_input_args(nrays, ngrid, rank, nprocs);
    Window *h_grid = allocate_window(args->rows, args->cols, args->window_resolution);
    Vectors *vectors = allocate_vectors(args);
    unsigned long long h_counter;
    int stride = (args->n_rays + (blocks * threads) - 1) / (blocks * threads);

    const int window_resolution = args->window_resolution;
    const real window_position = args->window_position;
    const real window_size = args->window_size;
    const real window_size_squared = window_size * window_size;
    const real sphere_radius_squared = args->sphere_radius * args->sphere_radius;
    const real window_position_factor = window_resolution / (2.0 * window_size);
    const real cc = (vectors->C.x * vectors->C.x) +
                    (vectors->C.y * vectors->C.y) +
                    (vectors->C.z * vectors->C.z);

    unsigned long long *d_counter;
    real *d_grid;
    size_t grid_bytes = h_grid->rows * h_grid->cols * sizeof(real);

    Timer *total_timer = timer_init();
    Timer *kernel_timer = timer_init();
    timer_start(total_timer);

    /* bind each rank to a different GPU (round-robin if ranks > GPUs) */
    int device_count;
    cudaGetDeviceCount(&device_count);
    cudaSetDevice(rank % device_count);

    cudaMalloc(&d_grid, grid_bytes);
    cudaCheckError();
    cudaMemset(d_grid, 0, grid_bytes);
    cudaCheckError();

    cudaMalloc(&d_counter, sizeof(unsigned long long));
    cudaCheckError();
    cudaMemset(d_counter, 0, sizeof(unsigned long long));
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();

    timer_start(kernel_timer);

    size_t shared_bytes = sizeof(unsigned long long) * threads;

    ray_tracer<<<blocks, threads, shared_bytes>>>(
        d_grid, args->rows, args->cols, args->n_rays, vectors->C,
        vectors->L, window_position, window_size_squared,
        window_position_factor, sphere_radius_squared - cc,
        d_counter, window_size, stride, rank
    );

    cudaDeviceSynchronize();
    cudaCheckError();

    int64_t kernel_time_elapsed = timer_end(kernel_timer);

    cudaMemcpy(h_grid->data, d_grid, grid_bytes, cudaMemcpyDeviceToHost);
    cudaCheckError();
    cudaMemcpy(&h_counter, d_counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();

    /* sum all per-rank grids into rank 0 */
    if (rank == 0) {
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : (void *)h_grid->data,
                   (void *)h_grid->data, h_grid->global_size,
                   MPI_REAL_CUSTOM_DTYPE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(h_grid->data, NULL, h_grid->global_size,
                   MPI_REAL_CUSTOM_DTYPE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    int64_t total_time_elapsed = timer_end(total_timer);

    /* gather timing and sample-count stats on rank 0 */
    int64_t global_kernel_time_elapsed = 0;
    int64_t global_total_time_elapsed = 0;
    unsigned long long global_counter = 0;

    MPI_Reduce(&total_time_elapsed, &global_total_time_elapsed, 1, MPI_INT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&kernel_time_elapsed, &global_kernel_time_elapsed, 1, MPI_INT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&h_counter, &global_counter, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total Time taken: %.5f secs.\nKernel Time Taken: %.5f secs.\nNumber of random samples: %llu\n",
               global_total_time_elapsed / 1000000.0, global_kernel_time_elapsed / 1000000.0, global_counter);
        write_grid(h_grid);
    }

    cudaFree(d_grid);
    cudaCheckError();
    cudaFree(d_counter);
    cudaCheckError();
    free_window(h_grid);
    free(args);
    free(total_timer);
    free(kernel_timer);
    free_vectors(vectors);

    MPI_Finalize();

    return 0;
}
