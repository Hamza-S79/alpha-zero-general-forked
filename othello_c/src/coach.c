#include "coach.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "othello.h"

/*
 * RNG: small xorshift64. Local to coach so caller controls determinism via
 * CoachArgs.rng_seed; no other component touches it.
 */
static uint64_t xs_state;
static void xs_seed(unsigned s) { xs_state = s ? s : 0xDEADBEEFu; }
static uint64_t xs_next(void) {
    uint64_t x = xs_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    xs_state = x;
    return x;
}

/* Sample an action index a from probability vector pi (length OTHELLO_ACTION_SIZE).
 * Mirrors np.random.choice(len(pi), p=pi). */
static int sample_action(const float* pi) {
    float r = (float)xs_next() / (float)UINT64_MAX;
    float acc = 0.0f;
    for (int a = 0; a < OTHELLO_ACTION_SIZE; a++) {
        acc += pi[a];
        if (r < acc) return a;
    }
    /* numerical fallback: return last non-zero action */
    for (int a = OTHELLO_ACTION_SIZE - 1; a >= 0; a--) {
        if (pi[a] > 0.0f) return a;
    }
    return OTHELLO_PASS_ACTION;
}

void coach_write_header(FILE* out, int n) {
    int32_t header[3] = { (int32_t)n, OTHELLO_ACTION_SIZE, 0 /* num_examples placeholder */ };
    fwrite(header, sizeof(int32_t), 3, out);
}

void coach_finalize_header(FILE* out, int total_examples) {
    long pos = ftell(out);
    fseek(out, 2 * sizeof(int32_t), SEEK_SET);   /* skip N, action_size */
    int32_t v = (int32_t)total_examples;
    fwrite(&v, sizeof(int32_t), 1, out);
    fseek(out, pos, SEEK_SET);
}

/*
 * Per-record on-disk layout:
 *   int8 [N*N]               canonical_board (row-major (x,y))
 *   float[ACTION_SIZE]       pi
 *   int8                     z  (filled in once the game ends)
 *   int8                     player (+1/-1) — used to derive z, then dropped
 *
 * We buffer the (board, pi, player) tuples in memory, then walk back through
 * them at game-end to assign z = r * (+1 if player == winner_pov else -1).
 * The `player` field is NOT written to disk (z carries the meaning).
 */
typedef struct {
    int8_t board[OTHELLO_NN];
    float  pi[OTHELLO_ACTION_SIZE];
    int8_t player;     /* +1 or -1 */
} EpisodeMove;

int coach_run_episode(NN* nn, CoachArgs args, FILE* out, int* out_examples_written) {
    xs_seed(args.rng_seed);

    /* Episodes max ~OTHELLO_NN moves but pad generously */
    EpisodeMove moves[2 * OTHELLO_NN + 16];
    int n_moves = 0;

    MCTS* mcts = mcts_new(nn, args.mcts_args);

    Board absolute;
    othello_init(&absolute);
    int player = +1;     /* +1 == "p0 in canonical view"; we track it explicitly */

    int episode_step = 0;
    while (1) {
        episode_step++;
        /* canonical = absolute viewed from current player's POV.
         * Our absolute board is itself in canonical-of-current form because
         * othello_apply swaps p0/p1 each step. So `absolute` IS the canonical
         * for the side whose turn it is. */
        Board canonical = absolute;

        float pi[OTHELLO_ACTION_SIZE];
        float temp = (episode_step < args.tempThreshold) ? 1.0f : 0.0f;
        mcts_action_prob(mcts, &canonical, temp, pi);

        /* Persist (canonical, pi, player) for later z-labeling. */
        if (n_moves >= (int)(sizeof(moves)/sizeof(moves[0]))) {
            fprintf(stderr, "coach: episode exceeded max moves\n");
            mcts_free(mcts);
            return 1;
        }
        othello_to_int8(&canonical, moves[n_moves].board);
        memcpy(moves[n_moves].pi, pi, sizeof(pi));
        moves[n_moves].player = (int8_t)player;
        n_moves++;

        /* Sample action and step. */
        int action = sample_action(pi);
        if (action == OTHELLO_PASS_ACTION && othello_legal_mask(&absolute) != 0) {
            /* Numerical edge: pi gave non-zero weight to pass when other moves exist.
             * Fall back to first legal move. */
            action = __builtin_ctzll(othello_legal_mask(&absolute));
        }
        othello_apply(&absolute, action);
        player = -player;

        int term = othello_terminal(&absolute);
        if (term != 0) {
            /* `term` is from the perspective of whoever's turn it is now (= -player_who_just_moved).
             * To match Coach.executeEpisode's z formula:
             *   r = getGameEnded(board, curPlayer)   (curPlayer = side to move now = `player`)
             *   z[i] = r * ((-1) ** (move.player != curPlayer))
             * which is: z = r if move.player == curPlayer else -r.
             */
            int r = term;
            int curPlayer = player;
            for (int i = 0; i < n_moves; i++) {
                int8_t z = (int8_t)((moves[i].player == curPlayer) ? r : -r);
                fwrite(moves[i].board, sizeof(int8_t), OTHELLO_NN, out);
                fwrite(moves[i].pi,    sizeof(float),  OTHELLO_ACTION_SIZE, out);
                fwrite(&z,             sizeof(int8_t), 1, out);
            }
            if (out_examples_written) *out_examples_written += n_moves;
            mcts_free(mcts);
            return 0;
        }
    }
}
