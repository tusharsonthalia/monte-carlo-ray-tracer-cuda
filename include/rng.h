/*
 * xoshiro256** pseudo-random number generator.
 *
 * Each RNG instance holds its own 256-bit state so multiple threads can
 * draw random numbers without contention.  The state is seeded via
 * splitmix64 to decorrelate a single user-supplied seed into the four
 * 64-bit words the generator needs.
 *
 * Reference: Blackman & Vigna, "Scrambled Linear Pseudorandom Number
 * Generators"
 */

#include <stdint.h>
#include <stdlib.h>

const double INV_DIV = (1.0 / 9007199254740992.0);

typedef struct rng_ {
  uint64_t s[4]; /* 256-bit internal state */
} RNG;

/* splitmix64 — used only during seeding to expand one 64-bit seed */
uint64_t splitmix64(uint64_t *x) {
  uint64_t z = (*x += 0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

/* Seed all four state words from a single 64-bit value */
void seed_rng(uint64_t seed, RNG *rng) {
  for (int i = 0; i < 4; i++)
    rng->s[i] = splitmix64(&seed);
}

/* Allocate and seed a new RNG on the heap */
RNG *new_rng(uint64_t seed) {
  RNG *rng = malloc(sizeof(RNG));
  seed_rng(seed, rng);

  return rng;
}

void free_rng(RNG *rng) {
  free(rng);
  return;
}

inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

/* xoshiro256** core — returns 64 uniformly-distributed bits */
inline uint64_t next_u64(RNG *rng) {
  const uint64_t result = rotl(rng->s[1] * 5, 7) * 9;

  const uint64_t t = rng->s[1] << 17;

  rng->s[2] ^= rng->s[0];
  rng->s[3] ^= rng->s[1];
  rng->s[1] ^= rng->s[2];
  rng->s[0] ^= rng->s[3];

  rng->s[2] ^= t;
  rng->s[3] = rotl(rng->s[3], 45);

  return result;
}

/* Return a double in [0, 1) with 53 bits of mantissa precision */
inline double next_double(RNG *rng) {
  const double val = (next_u64(rng) >> 11) * INV_DIV;
  return val;
}
