#ifndef OTHELLO_C_ARENA_H
#define OTHELLO_C_ARENA_H

#include "mcts.h"
#include "nn.h"

/*
 * Arena: head-to-head between two NN models. Each game alternates which side
 * starts; total games are split half/half. Action selection is greedy (temp=0,
 * argmax over visit counts).
 */

typedef struct { int wins_a, wins_b, draws; } ArenaResult;

ArenaResult arena_play(NN* nn_a, NN* nn_b, MCTSArgs mcts_args, int num_games);

#endif /* OTHELLO_C_ARENA_H */
