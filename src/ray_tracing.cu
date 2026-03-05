/*
 * ray_tracing.cu — Single-GPU CUDA ray tracer
 *
 * Uses grid-stride loops so each thread processes multiple rays, amortising
 * the cost of curand_init.  Per-thread rejection counts are stored in shared
 * memory and reduced once per block before a single atomicAdd to global
 * memory.
 *
 * Compile with -DUSE_FLOAT for single-precision (~2x faster on consumer GPUs).
 */

#include "ray_tracing.h"
#include "timer.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

/* ── CUDA-specific precision macros ──────────────────────────────────── */

#ifdef USE_FLOAT
    #define R_SQRT rsqrtf
    #define SQRT sqrtf
    #define COS cosf
    #define SIN sinf
    #define CURAND_UNIFORM curand_uniform
#else
    #define R_SQRT rsqrt
    #define SQRT sqrt
    #define COS cos
    #define SIN sin
    #define CURAND_UNIFORM curand_uniform_double
#endif

/* ── Input ───────────────────────────────────────────────────────────── */

Input *read_input_args(const long nrays, const int ngrid) {
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
    args->n_rays = nrays;

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

/* ── Device vector math (force-inlined for ILP) ──────────────────────── */

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

/* normalise in-place using reciprocal square root (rsqrt) */
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

/* ── Ray sampling (rejection method, device side) ────────────────────
 *
 * Both rejection conditions (window bounds + sphere miss) are evaluated
 * in a single branch to minimise warp divergence.
 */
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

        /* unified rejection: window miss OR imaginary intersection */
        if (((W->x * W->x) >= W_max_squared) || ((W->z * W->z) >= W_max_squared) || (t_inner < 0)) {
            continue;
        }

        return vc - SQRT(t_inner);
    }
}

/* ── CUDA kernel ─────────────────────────────────────────────────────
 *
 * Each thread processes `stride` rays via a grid-stride loop.
 * A shared-memory buffer collects per-thread sample counts; thread 0
 * in each block commits the block total with a single atomicAdd.
 */
__global__ void ray_tracer(
    real *grid, int rows, int cols, long n_rays, Vector C,
    Vector L, real window_position, real window_size_squared,
    real window_position_factor, real sphere_radius_squared_cc,
    unsigned long long *d_counter, real window_size, const int stride
) {
    long tid = (blockIdx.x * blockDim.x + threadIdx.x) * stride;

    extern __shared__ unsigned long long sample_counter[];
    real t;

    sample_counter[threadIdx.x] = 0;

    curandState rng;
    curand_init(4238811ULL, tid, 0, &rng);

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

    /* block-level reduction of sample counts */
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
    char *usage_string = (char *)"./ray_tracing nrays ngrid nblocks threads\n";
    if (argc != 5) {
        printf("%s", usage_string);
        return 1;
    }

    long nrays = atol(argv[1]);
    int ngrid = atoi(argv[2]);
    int blocks = atoi(argv[3]);
    int threads = atoi(argv[4]);
    int stride = (nrays + (blocks * threads) - 1) / (blocks * threads);

    Input *args = read_input_args(nrays, ngrid);
    Window *h_grid = allocate_window(args->rows, args->cols, args->window_resolution);
    Vectors *vectors = allocate_vectors(args);
    unsigned long long h_counter;

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
        d_counter, window_size, stride
    );

    cudaDeviceSynchronize();
    cudaCheckError();

    const int64_t kernel_time_elapsed = timer_end(kernel_timer);

    cudaMemcpy(h_grid->data, d_grid, grid_bytes, cudaMemcpyDeviceToHost);
    cudaCheckError();
    cudaMemcpy(&h_counter, d_counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();

    const int64_t total_time_elapsed = timer_end(total_timer);

    printf("Total Time taken: %.5f secs.\nKernel Time Taken: %.5f secs.\nNumber of random samples: %llu\n",
           total_time_elapsed / 1000000.0, kernel_time_elapsed / 1000000.0, h_counter);

    /* query the runtime for the best launch configuration */
    int minGridSize = 0;
    int blockSize = 0;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ray_tracer, shared_bytes, 0);

    printf("Recommended config:\n");
    printf("  Block size: %d threads\n", blockSize);
    printf("  Min grid size: %d blocks\n", minGridSize * 4);
    printf("  Total threads: %d\n", minGridSize * blockSize);

    write_grid(h_grid);

    cudaFree(d_grid);
    cudaCheckError();
    cudaFree(d_counter);
    cudaCheckError();
    free_window(h_grid);
    free(args);
    free(total_timer);
    free(kernel_timer);
    free_vectors(vectors);

    return 0;
}
