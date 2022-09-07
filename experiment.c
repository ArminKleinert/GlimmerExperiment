
#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define upto(v, max) for (ssize_t v = 0; v < max; v++)

#define NUM_GROUPS 3
#define NUM_ATTR 4
#define NUM_ENTITIES 48
#define GENERATIONS 10000

#define MIN_AFF -64
#define MAX_AFF 64
#define MIN_ATTR_PRIO -100
#define MAX_ATTR_PRIO 100
#define MIN_ATTR_VAL -128
#define MAX_ATTR_VAL 255

#define MIN_AFF_FOR_REPRO -32
#define MIN_G_FOR_REPRO 1

#define CH_MUTATION (1 << 30)
#define MUTATION_SHIFT 31
#define CH_PREFER_NEW 0
#define CH_OTHER_GROUP 1000
#define OWN_GROUP_AFF_MOD 2

#define DO_SHUFFLE 1

uint32_t xorshift(uint32_t state) {
  uint32_t x = state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

typedef struct {
  int32_t group;
  int32_t affinities[NUM_GROUPS]; // How much the entity likes each group
  int32_t attr_prio[NUM_ATTR];    // How much the entity values each attribute
  int32_t attr_vals[NUM_ATTR];    // The attribute values of the entity
} ENTITY;

ENTITY entities[NUM_ENTITIES];
ENTITY buffer_entities[NUM_ENTITIES];

void shuffle_entities(uint32_t *rand_seed) {
  uint32_t rseed = *rand_seed;
  if (NUM_ENTITIES > 1) {
    upto(i, NUM_ENTITIES - 1) {
      size_t j = i + (rseed = xorshift(rseed)) % (NUM_ENTITIES - i);
      ENTITY t = entities[j];
      entities[j] = entities[i];
      entities[i] = t;
    }
  }
}

int64_t reproductional_value(const ENTITY *entity0, const ENTITY *other) {
  int64_t rv = 0;
  const int32_t *prio = entity0->attr_prio;
  const int32_t *vals = other->attr_vals;

  // Calculate reproductional_value
  for (int i = 0; i < NUM_ATTR; i++) {
    rv += prio[i] * vals[i];
  }

  return rv;
}

// Choose one of 2 inputs (ca. 50% chance each)
int32_t choose32(int32_t c0, int32_t c1, uint32_t *rand_seed) {
  *rand_seed = xorshift(*rand_seed);
  if (*rand_seed & 1)
    return c0;
  else
    return c1;
}

int32_t maybe_flip_bit(int32_t n, uint32_t *rand_seed) {
  if (CH_MUTATION && (*rand_seed = xorshift(*rand_seed)) % CH_MUTATION == 0) {
    *rand_seed = xorshift(*rand_seed);
    n ^= 1 << (*rand_seed % (MUTATION_SHIFT + 1));
  }
  return n;
}

void make_child(ENTITY *par0, ENTITY *par1, ENTITY *child,
                uint32_t *rand_seed) {
  // Group of child depends on who cared more abou the other.
  // There is a 1/4 chance that the other group is used instead.
  int64_t rv01 = reproductional_value(par0, par1);
  int64_t rv10 = reproductional_value(par1, par0);
  int32_t rand_factor =
      (*rand_seed = xorshift(*rand_seed)) % (CH_OTHER_GROUP + 1);
  int32_t group = (rv01 < rv10 && rand_factor) ? par1->group : par0->group;

  child->group = group;
  upto(i, NUM_GROUPS) {
    child->affinities[i] =
        choose32(par0->affinities[i], par1->affinities[i], rand_seed);

    if (i == group && OWN_GROUP_AFF_MOD != 0) {
      child->affinities[i] +=
          (*rand_seed = xorshift(*rand_seed)) % OWN_GROUP_AFF_MOD;
    }

    child->affinities[i] = maybe_flip_bit(child->affinities[i], rand_seed);
  }

  upto(i, NUM_ATTR) {
    child->attr_prio[i] =
        choose32(par0->attr_prio[i], par1->attr_prio[i], rand_seed);
    child->attr_prio[i] = maybe_flip_bit(child->attr_prio[i], rand_seed);
  }

  upto(i, NUM_ATTR) {
    child->attr_vals[i] =
        choose32(par0->attr_vals[i], par1->attr_vals[i], rand_seed);
    child->attr_vals[i] = maybe_flip_bit(child->attr_vals[i], rand_seed);
  }
}

void reproduce(ENTITY *entity0, ENTITY *entity1, uint32_t *rand_seed) {
  ptrdiff_t e0_idx = entity0 - entities;
  make_child(entity0, entity1, buffer_entities + e0_idx, rand_seed);

  /*
  if (((*rand_seed = xorshift(*rand_seed)) % 16) != 0) {
  ptrdiff_t e1_idx = entity1 - entities;
  make_child(entity0, entity1, buffer_entities+e1_idx, rand_seed);
}
*/
}

void finalize_generation(void) {
  memcpy(entities, buffer_entities, sizeof(entities));
}

int32_t affinity_multiplier(const int32_t *affinities, int32_t group) {
  int32_t affinity = affinities[group];
  /*
  if (affinity < -64) {
    return -2;
  } else if (affinity < 0) {
    return 1;
  } else if (affinity < 64) {
    return 4;
  } else {
    return 8;
  }
  */
  return affinity;
}

int64_t gravity(const ENTITY *entity0, const ENTITY *other) {
  int64_t g = reproductional_value(entity0, other);
  int64_t aff = affinity_multiplier(entity0->affinities, other->group);

  if (g < MIN_G_FOR_REPRO || aff < MIN_AFF_FOR_REPRO) {
    return 0;
  }

  return g * aff;
}

#define PART_FORMAT "%04d"
void print_arr(FILE *stream, const int len, const int32_t *arr) {
  fprintf(stream, " (");
  for (int i = 0; i < len; i++) {
    fprintf(stream, PART_FORMAT, arr[i]);
    if (i < len - 1)
      fprintf(stream, " ");
  }
  fprintf(stream, ")");
}

void print_entity(FILE *stream, const ENTITY *entity0) {
  if (entity0->group != 0xFFFF) {
    fprintf(stream, PART_FORMAT, entity0->group);
    print_arr(stream, NUM_GROUPS, entity0->affinities);
    print_arr(stream, NUM_ATTR, entity0->attr_prio);
    print_arr(stream, NUM_ATTR, entity0->attr_vals);
  }
}

void randomize_arr(uint32_t *seed, const int len, const int32_t min_bound,
                   const int32_t max_bound, int32_t *arr) {
  int32_t lower_b = min_bound < 0 ? -min_bound : min_bound;
  int32_t higher_b = min_bound < 0 ? max_bound + lower_b : max_bound - lower_b;
  for (int i = 0; i < len; i++) {
    *seed = xorshift(*seed);
    arr[i] = (*seed & 0x7FFFFFFF) % higher_b - lower_b;
  }
}

void print_all_entities(void) {
  upto(j, NUM_ENTITIES) {
    print_entity(stdout, &(entities[j]));
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  uint32_t rand_seed = 0xDEAD10CC;
  finalize_generation();
  upto(i, NUM_ENTITIES) {
    ENTITY *e = entities + i;
    e->group = i / (NUM_ENTITIES / NUM_GROUPS);
    // e->group = i % NUM_GROUPS;

    randomize_arr(&rand_seed, NUM_GROUPS, MIN_AFF, MAX_AFF, e->affinities);
    e->affinities[e->group] = 255;

    randomize_arr(&rand_seed, NUM_ATTR, MIN_ATTR_PRIO, MAX_ATTR_PRIO,
                  e->attr_prio);
    randomize_arr(&rand_seed, NUM_ATTR, MIN_ATTR_VAL, MAX_ATTR_VAL,
                  e->attr_vals);
  }

  print_all_entities();

  upto(gen, GENERATIONS) {
    upto(curr_i, NUM_ENTITIES) {
      ENTITY *curr = entities + curr_i;
      ENTITY *best = entities;
      int64_t best_score = -255;
      upto(inner_i, NUM_ENTITIES) {
        if (curr_i != inner_i) {
          int64_t g = gravity(curr, entities + inner_i);
          rand_seed = xorshift(rand_seed);
          if (g > best_score ||
              (CH_PREFER_NEW && (rand_seed % CH_PREFER_NEW == 0))) {
            best_score = g;
            best = entities + inner_i;
          }
        }
      }
      reproduce(curr, best, &rand_seed);
    }
    finalize_generation();
    // print_all_entities();
    if (DO_SHUFFLE)
      shuffle_entities(&rand_seed);
  }

  print_all_entities();

  clock_gettime(CLOCK_MONOTONIC_RAW, &end);

  uint64_t delta_us = (end.tv_sec - start.tv_sec) * 1000000 +
                      (end.tv_nsec - start.tv_nsec) / 1000;
  printf("\nTime taken: %lu.%lums\n", delta_us / 1000, delta_us % 1000);

  return 0;
}
