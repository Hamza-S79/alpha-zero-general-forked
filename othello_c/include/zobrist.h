#ifndef OTHELLO_C_ZOBRIST_H
#define OTHELLO_C_ZOBRIST_H

#include <stdint.h>

#include "othello.h"

/*
 * Zobrist hash for canonical-form Othello boards.
 *
 * Two precomputed tables of NN random 64-bit numbers, one for p0 stones and
 * one for p1. Hash(board) = XOR of ZOBRIST_P0[bit] for each set bit in p0,
 * XORed with ZOBRIST_P1[bit] for each set bit in p1.
 *
 * The tables are seeded once at program start (or first-use) from a fixed
 * RNG so hashes are stable across runs.
 */

extern uint64_t ZOBRIST_P0[OTHELLO_NN];
extern uint64_t ZOBRIST_P1[OTHELLO_NN];

/* Seed the tables. Idempotent; safe to call multiple times. */
void zobrist_init(void);

/* Hash a canonical board. */
uint64_t zobrist_hash(const Board* b);

#endif /* OTHELLO_C_ZOBRIST_H */
