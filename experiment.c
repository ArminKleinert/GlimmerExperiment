#include <limits.h>
//#include <math.h>
//#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

typedef int INDEX;

#define upto(v, max) for (INDEX v = 0; v < max; v++)
#define downto(v, max, min) for (INDEX v = max; v >= min; v--)
#define LOG_ERROR(msg)                                                         \
  fprintf(stderr, "Error in file %s at line %d: %s.\n", __FILE__, __LINE__, msg)

static uint32_t WHERE_AM_I_COUNTER = 0;
#define WHERE_AM_I()                                                           \
  do {                                                                         \
    fprintf(stderr, "I am in file %s in line %d. (counter: %d)\n", __FILE__,   \
            __LINE__, WHERE_AM_I_COUNTER++);                                   \
    fflush(stderr);                                                            \
  } while (0)

#define NUM_ATTR 6 // Number of attributes.
#define NUM_ENTITIES 5
#define MAX_ENTITIES 1024
#define MAX_CANDIDATES 4
#define CURR_MAX_CANDIDATES(current_bound)                                     \
  (MAX_CANDIDATES < current_bound ? MAX_CANDIDATES : current_bound)
#define MIN_APPEAL 0
#define GENERATIONS 79

#define MIN_ATTR_PRIO -256
#define MAX_ATTR_PRIO 255
#define MIN_ATTR_VAL -128
#define MAX_ATTR_VAL 127

#define CHANCE_RAND_SHIFT                                                      \
  8 // A 1 in CHANCE_RAND_SHIFT chance that a random mutation will occur
#define MAX_SHIFT                                                              \
  ((1 << 6) - 1) // If a random mutation occurs, a value between MAX_SHIFT and
                 // -MAX_SHIFT will be added to the attribute.

#define INIT_SEED 0xDEAD10CC

/*
 * The algorithm:
 * GENERATION(entities, BS, CandidateSets):
 * - for every entity E do
 * - - Order all other entities in order of desirability.
 * - - Keep only the top MAX_CANDIDATES entities of and collect then into a set
 * CS
 * - - Add CS to CandidateSets (of E)
 * - - Make BS the union of CS and BS
 * - OUT_ENTITIES = empty_set
 * - for every entity E do
 * - - if E is not desired by any of its desired candidates, E does not get to
 * reproduce
 * - - if E is desired by any of its top candidates, they reproduce and the new
 * entity is put into the OUT_ENTITIES set
 * - - if OUT_ENTITIES is full, break out of the loop
 * - Update the set of entities to be OUT_ENTITIES
 *
 * desirability(E1, E2):
 * - rep_value = 0
 * - for each attribute A of E2:
 * - - rep_value += A * priority(E1, attribute)
 * - return rep_value
 *
 * reproduce(E1, E2):
 * - E3 = empty_entity
 * - for each attribute type A
 * - - Choose one of attr_val(A, E1) and attr_val(A, E2)
 * - - attr_val(A, E3) := that value
 * - return E3
 */

static uint32_t xorshift(uint32_t state) {
  uint32_t x = state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

typedef struct {
  int32_t attr_prio[NUM_ATTR]; // How much the entity values each attribute
  int32_t attr_vals[NUM_ATTR]; // The attribute values of the entity
  int32_t entity_id;
} ENTITY;

static int64_t desirability(const ENTITY *admirer, const ENTITY *other) {
  int64_t rv = 0;
  const int32_t *prio = admirer->attr_prio;
  const int32_t *vals = other->attr_vals;

  // Calculate desirability
  for (INDEX i = 0; i < NUM_ATTR; i++) {
    rv += prio[i] * vals[i];
  }

  if (rv < MIN_APPEAL) {
    return INT_MIN;
  } else {
    return rv;
  }
}

// Choose one of 2 inputs (ca. 50% chance each)
static int32_t choose32(int32_t c0, int32_t c1, uint32_t *rand_seed) {
  *rand_seed = xorshift(*rand_seed);
  int32_t c_out;
  if (*rand_seed & 1)
    c_out = c0;
  else
    c_out = c1;

  *rand_seed = xorshift(*rand_seed);
  int do_shift = (*rand_seed) % 1 == 0;
  if (do_shift) {
    *rand_seed = xorshift(*rand_seed) % MAX_SHIFT;
    int32_t shift_value = (int32_t)(*rand_seed);

    *rand_seed = xorshift(*rand_seed);
    int direction = (*rand_seed) & 1;

    if (direction) {
      c_out += shift_value;
    } else {
      c_out -= shift_value;
    }
  }
  return c_out;
}

#define PART_FORMAT "%4d"
static void print_arr(FILE *stream, const INDEX len, const int32_t *arr) {
  fprintf(stream, " (");
  for (INDEX i = 0; i < len; i++) {
    fprintf(stream, PART_FORMAT, arr[i]);
    if (i < len - 1) {
      fprintf(stream, " ");
    }
  }
  fprintf(stream, ")");
}

static void print_entity(FILE *stream, const ENTITY *entity0) {
  printf(PART_FORMAT, entity0->entity_id);
  print_arr(stream, NUM_ATTR, entity0->attr_vals);
  printf(" ");
  print_arr(stream, NUM_ATTR, entity0->attr_prio);
}

static void randomize_arr(uint32_t *seed, const INDEX len,
                          const int32_t min_bound, const int32_t max_bound,
                          int32_t *arr) {
  const int32_t low_b = min_bound < 0 ? -min_bound : min_bound;
  const int32_t high_b = min_bound < 0 ? max_bound + low_b : max_bound - low_b;
  for (INDEX i = 0; i < len; i++) {
    *seed = xorshift(*seed);
    arr[i] = (*seed & 0x7FFFFFFF) % high_b - low_b;
  }
}

static void make_random_entities(uint32_t *rand_seed, ENTITY *entities,
                                 const INDEX used_entities) {
  upto(i, used_entities) {
    ENTITY *e = entities + i;
    e->entity_id = i;
    randomize_arr(rand_seed, NUM_ATTR, MIN_ATTR_PRIO, MAX_ATTR_PRIO,
                  e->attr_prio);
    randomize_arr(rand_seed, NUM_ATTR, MIN_ATTR_VAL, MAX_ATTR_VAL,
                  e->attr_vals);
  }
}

static void print_all_entities(const INDEX used_entities, ENTITY *entities) {
  upto(entity_index, used_entities) {
    print_entity(stdout, &(entities[entity_index]));
    printf("\n");
  }
  printf("\n");
}

// Sorts 2 arrays in reverse order, using the second array as the point of
// comparison.
static void insertion_sort(ENTITY *entities, int32_t *scores, INDEX used_len) {
  int32_t key;
  for (INDEX index = 1; index < used_len; index++) {
    key = scores[index];
    INDEX comparison_index = index - 1;

    while (comparison_index >= 0 && scores[comparison_index] < key) {
      scores[comparison_index + 1] = scores[comparison_index];
      entities[comparison_index + 1] = entities[comparison_index];
      comparison_index = comparison_index - 1;
    }
    scores[comparison_index + 1] = key;
  }
}

static void do_reproduce(ENTITY *entity, ENTITY *other, ENTITY *new_entity,
                         INDEX id, uint32_t *rand_seed) {
  new_entity->entity_id = id;
  upto(attr_index, NUM_ATTR) {
    int32_t score = entity->attr_vals[attr_index];
    int32_t other_score = other->attr_vals[attr_index];
    uint32_t rand_score = choose32(score, other_score, rand_seed);
    *rand_seed = rand_score;
    new_entity->attr_vals[attr_index] = (int32_t)(*rand_seed);
  }
  upto(attr_index, NUM_ATTR) {
    int32_t prio = entity->attr_prio[attr_index];
    int32_t other_prio = other->attr_prio[attr_index];
    uint32_t rand_prio = choose32(prio, other_prio, rand_seed);
    *rand_seed = rand_prio;
    new_entity->attr_prio[attr_index] = (int32_t)(*rand_seed);
  }
}

// Return next available index in entity array
static INDEX reproduce_if_possible(ENTITY *entity, ENTITY *new_entity_array,
                                   ENTITY *old_entity_array,
                                   const INDEX used_entities,
                                   const INDEX next_available_index,
                                   int32_t **candidate_sets,
                                   uint32_t *rand_seed) {
  INDEX new_max_index = next_available_index;
  upto(current_entity, used_entities) { // IDs of all other entities
    if (new_max_index >= MAX_ENTITIES - 1) {
      break;
    }
    if (current_entity == entity->entity_id) {
      continue; // Do not reproduce with self.
    }
    if (current_entity != entity->entity_id) {
      // IDs in candidate set of other entity
      upto(index_in_candidate_set, CURR_MAX_CANDIDATES(used_entities)) {
        int32_t desired_id_of_candidate =
            candidate_sets[current_entity][index_in_candidate_set];
        if (desired_id_of_candidate == INT_MIN ||
            new_max_index >= MAX_ENTITIES) {
          break;
        }
        if (desired_id_of_candidate == entity->entity_id) {
          ENTITY *old_entity = old_entity_array + current_entity;
          ENTITY *new_entity = new_entity_array + new_max_index;
          do_reproduce(entity, old_entity, new_entity, new_max_index,
                       rand_seed);
          new_max_index++;
        }
      }
    }
  }
  return new_max_index;
}

// Returns number of used entities
static INDEX generation(ENTITY *entities, ENTITY *buffer_entities,
                        int32_t *entity_scores, int32_t **candidate_sets,
                        int32_t **candidate_scores, INDEX used_entities,
                        uint32_t *rand_seed) {
  memcpy(buffer_entities, entities, used_entities * sizeof(ENTITY));

  // memset(entity_scores, 0, MAX_ENTITIES * sizeof(int32_t));

  // Calculate and save candidates for each entity.
  upto(i, used_entities) {
    entity_scores[i] = INT_MIN;
    const ENTITY current = entities[i];
    upto(inner, used_entities) {
      if (i != inner) { // An entity does not have to rate itself.
        const ENTITY e = entities[inner];
        const int32_t score = desirability(&current, &e);
        entity_scores[inner] = score;
      }
    }

    // Sort entities by score.
    insertion_sort(buffer_entities, entity_scores, used_entities);

    // Copy the first MAX_CANDIDATES candidates from the buffer together with
    // their scores.
    upto(j, CURR_MAX_CANDIDATES(used_entities)) {
      if (entity_scores[j] < MIN_APPEAL) {
        candidate_scores[i][j] = INT_MIN;
        candidate_sets[i][j] = INT_MIN;
      } else {
        candidate_scores[i][j] = entity_scores[j];
        candidate_sets[i][j] = buffer_entities[j].entity_id;
      }
    }
  }

  INDEX next_entity_index = 0;
  upto(entity_index, used_entities) {
    if (entity_index < MAX_ENTITIES) {
      next_entity_index = reproduce_if_possible(
          entities + entity_index, buffer_entities, entities, used_entities,
          next_entity_index, candidate_sets, rand_seed);
      if (next_entity_index >= MAX_ENTITIES - 1) {
        break;
      }
    }
    if (next_entity_index >= MAX_ENTITIES - 1) {
      break;
    }
  }

  memcpy(entities, buffer_entities, next_entity_index * sizeof(ENTITY));

  return next_entity_index;
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  (void)WHERE_AM_I_COUNTER;

  INDEX used_entities = NUM_ENTITIES;
  ENTITY *entities = malloc(MAX_ENTITIES * sizeof(ENTITY));
  ENTITY *buffer_entities = malloc(MAX_ENTITIES * sizeof(ENTITY));
  int32_t **candidate_sets = malloc(MAX_ENTITIES * sizeof(int32_t *));
  int32_t *entity_scores = malloc(MAX_ENTITIES * sizeof(int32_t));
  int32_t **candidate_scores = malloc(MAX_ENTITIES * sizeof(int32_t *));

  if (!(entities && buffer_entities && entity_scores && candidate_sets &&
        candidate_scores)) {
    LOG_ERROR("Could not allocate memory.");
    free(entities);
    free(buffer_entities);
    free(candidate_sets);
    free(entity_scores);
    free(candidate_scores);
    return 1;
  }

  upto(i, MAX_ENTITIES) {
    candidate_sets[i] = malloc(MAX_CANDIDATES * sizeof(int32_t));
    if (!candidate_sets[i]) {
      LOG_ERROR("Could not allocate memory.");
      downto(j, i, 0) {
        free(candidate_sets[i]); // Go backwards and free
      }
      return 1;
    }
  }
  upto(i, MAX_ENTITIES) {
    candidate_scores[i] = malloc(MAX_CANDIDATES * sizeof(int32_t));
    if (!candidate_scores[i]) {
      LOG_ERROR("Could not allocate memory.");
      downto(j, i, 0) {
        free(candidate_scores[i]); // Go backwards and free
      }
      return 1;
    }
  }

  uint32_t rand_seed = INIT_SEED;
  make_random_entities(&rand_seed, entities, used_entities);

  print_all_entities(used_entities, entities);

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  upto(gen, GENERATIONS) {
    used_entities =
        generation(entities, buffer_entities, entity_scores, candidate_sets,
                   candidate_scores, used_entities, &rand_seed);
    if (used_entities == 0) {
      printf("The entities died out. (Generation %d)\n", gen);
      break;
    }
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &end);

  print_all_entities(used_entities, entities);

  uint64_t delta_us = (end.tv_sec - start.tv_sec) * 1000000 +
                      (end.tv_nsec - start.tv_nsec) / 1000;
  printf("\nTime taken: %lu.%lums\n", delta_us / 1000, delta_us % 1000);

  // Clean the house.
  upto(i, MAX_ENTITIES) { free(candidate_sets[i]); }
  upto(i, MAX_ENTITIES) { free(candidate_scores[i]); }
  free(entities);
  free(buffer_entities);
  free(candidate_sets);
  free(entity_scores);
  free(candidate_scores);

  return 0;
}
