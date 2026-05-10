#include "arena.h"

#include <stdio.h>

#include "othello.h"

/*
 * Pick the action with the highest visit count from the given MCTS instance.
 * Ties broken to lowest action index — matches Coach.learn's
 * `np.argmax(getActionProb(temp=0))` (np.argmax returns the first max).
 */
static int greedy_action(MCTS* m, const Board* canonical) {
    float pi[OTHELLO_ACTION_SIZE];
    mcts_action_prob(m, canonical, 0.0f, pi);
    int best = 0;
    for (int a = 1; a < OTHELLO_ACTION_SIZE; a++) {
        if (pi[a] > pi[best]) best = a;
    }
    return best;
}

/*
 * Play one game between two MCTS players. Returns +1 if `mcts_first` won
 * (the side that played the first move), -1 if `mcts_first` lost, 0 on draw.
 *
 * The board flips canonical-form each move (othello_apply swaps p0/p1), so
 * the side currently to move is always represented by p0. We just have to
 * track which MCTS instance is up.
 */
static int play_one_game(MCTS* mcts_first, MCTS* mcts_second) {
    Board b; othello_init(&b);
    MCTS* turn = mcts_first;
    int first_player_just_moved = 0;  /* 0 → first hasn't moved yet */

    while (1) {
        int term = othello_terminal(&b);
        if (term != 0) {
            /* `term` is from the POV of the side-to-move (the one who now
             * lacks moves). We want it from `mcts_first`'s POV:
             *   if it's mcts_first's turn now: result = term
             *   else: result = -term
             */
            return (turn == mcts_first) ? term : -term;
        }
        int a = greedy_action(turn, &b);
        othello_apply(&b, a);
        first_player_just_moved = (turn == mcts_first);
        turn = (turn == mcts_first) ? mcts_second : mcts_first;
    }
    (void)first_player_just_moved;
}

ArenaResult arena_play(NN* nn_a, NN* nn_b, MCTSArgs mcts_args, int num_games) {
    /* Match Python Arena: one MCTS per side, persisted across games. */
    MCTS* m_a = mcts_new(nn_a, mcts_args);
    MCTS* m_b = mcts_new(nn_b, mcts_args);

    ArenaResult r = { 0, 0, 0 };

    int half = num_games / 2;
    for (int g = 0; g < half; g++) {
        /* A starts. */
        int outcome = play_one_game(m_a, m_b);  /* +1 = A wins, -1 = B wins (draws map to -1 in our terminal()). */
        if (outcome == +1)      r.wins_a++;
        else if (outcome == -1) r.wins_b++;
        else                    r.draws++;
    }
    for (int g = 0; g < num_games - half; g++) {
        /* B starts. */
        int outcome = play_one_game(m_b, m_a);  /* +1 = B wins, -1 = A wins */
        if (outcome == +1)      r.wins_b++;
        else if (outcome == -1) r.wins_a++;
        else                    r.draws++;
    }

    mcts_free(m_a);
    mcts_free(m_b);
    return r;
}
