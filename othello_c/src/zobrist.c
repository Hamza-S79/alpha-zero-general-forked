#include "zobrist.h"

#include <stdint.h>

uint64_t ZOBRIST_P0[OTHELLO_NN];
uint64_t ZOBRIST_P1[OTHELLO_NN];

/* SplitMix64 — deterministic, fast, good distribution. Seeded with a fixed
 * constant so hashes are stable across runs. */
static uint64_t splitmix64(uint64_t* s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void zobrist_init(void) {
    static int initialised = 0;
    if (initialised) return;
    uint64_t s = 0xA5A5A5A5DEADBEEFULL;
    for (int i = 0; i < OTHELLO_NN; i++) ZOBRIST_P0[i] = splitmix64(&s);
    for (int i = 0; i < OTHELLO_NN; i++) ZOBRIST_P1[i] = splitmix64(&s);
    initialised = 1;
}

uint64_t zobrist_hash(const Board* b) {
    uint64_t h = 0;
    uint64_t bb = b->p0;
    while (bb) {
        int bit = __builtin_ctzll(bb);
        h ^= ZOBRIST_P0[bit];
        bb &= bb - 1;
    }
    bb = b->p1;
    while (bb) {
        int bit = __builtin_ctzll(bb);
        h ^= ZOBRIST_P1[bit];
        bb &= bb - 1;
    }
    return h;
}
