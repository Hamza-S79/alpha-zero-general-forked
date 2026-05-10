#ifndef OTHELLO_C_COACH_H
#define OTHELLO_C_COACH_H

#include <stdio.h>

#include "mcts.h"
#include "nn.h"

/*
 * One self-play episode: starts from the initial board, plays both sides
 * with the same NN via MCTS, records (canonical_board, pi, player) per move,
 * and writes the labeled examples (with final outcome z) to `out`.
 *
 * Format per record (matches python/dump_*_fixture conventions):
 *   int8[N*N]              canonical_board (row-major (x,y))
 *   float[ACTION_SIZE]     pi (post-temperature)
 *   int8                   z   (outcome from this record's player's POV: +1, -1)
 *
 * The file MUST already have its header written by coach_write_header.
 * `out_examples_written` is incremented by the number of moves played.
 *
 * The MCTS tree is reset between episodes (matches Coach.learn's
 * "self.mcts = _make_mcts(...)" reset per episode).
 */

typedef struct {
    MCTSArgs mcts_args;
    int      tempThreshold;     /* moves before switching to temp=0 (Python: 15) */
    unsigned rng_seed;
} CoachArgs;

void coach_write_header(FILE* out, int n);

int  coach_run_episode(NN* nn, CoachArgs args, FILE* out, int* out_examples_written);

/*
 * Patch the header's num_examples field at the start of `out`. Call once at
 * the end of self-play after writing all episodes.
 */
void coach_finalize_header(FILE* out, int total_examples);

#endif /* OTHELLO_C_COACH_H */
