#ifndef OTHELLO_C_MCTS_H
#define OTHELLO_C_MCTS_H

#include <stdint.h>

#include "nn.h"
#include "othello.h"
#include "tree.h"

/*
 * Leaf-parallel MCTS with virtual loss + a batched NN evaluator.
 *
 * Mirrors the algorithm in BatchedMCTS.py:
 *   - Wsa stores SUM of values (not running mean Qsa); Q = Wsa / Nsa.
 *   - At select: Nsa += 1, Wsa -= virtualLoss, Ns += 1.
 *   - At backup: Wsa += virtualLoss + v.
 *
 * With numParallelSims == 1 and virtualLoss == 0, this collapses to a serial
 * MCTS that produces the same visit counts as MCTS.py at fixed seed (matching
 * the sanity_check in benchmark_mcts.py).
 */

typedef struct {
    int   numMCTSSims;       /* total simulations per mcts_action_prob call  */
    float cpuct;
    int   numParallelSims;   /* worker threads; 1 = serial                  */
    float virtualLoss;
    int   evalBatchSize;     /* used when numParallelSims > 1               */
    float evalTimeoutMs;
    int   treeCapacityPow2;  /* tree hash table size (power of 2)           */
} MCTSArgs;

typedef struct MCTS MCTS;

/* nn must outlive the returned MCTS. */
MCTS* mcts_new(NN* nn, MCTSArgs args);
void  mcts_free(MCTS* m);

/*
 * Run numMCTSSims simulations from `root`, then write out_pi (length
 * OTHELLO_ACTION_SIZE):
 *   temp == 0  → one-hot on the most-visited action (smallest index on tie)
 *   temp != 0  → counts^(1/temp) normalized
 *
 * The MCTS tree persists across calls; reset by calling mcts_reset.
 */
void  mcts_action_prob(MCTS* m, const Board* root, float temp, float* out_pi);

/* Drop all tree state. */
void  mcts_reset(MCTS* m);

/* Profiling counters (for benchmark output). */
typedef struct {
    int64_t n_simulations;
    int64_t n_gpu_calls;
    int64_t total_batch_size;   /* sum of sizes; mean = total / n_gpu_calls */
    int64_t n_virtual_loss_collisions;
    double  total_seconds;
} MCTSStats;

void mcts_get_stats(const MCTS* m, MCTSStats* out);

#endif /* OTHELLO_C_MCTS_H */
