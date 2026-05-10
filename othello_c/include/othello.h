#ifndef OTHELLO_C_OTHELLO_H
#define OTHELLO_C_OTHELLO_H

#include <stdbool.h>
#include <stdint.h>

/*
 * Bitboard Othello rules. Mirrors the public surface of Python's OthelloGame.
 *
 * Encoding:
 *   - 6x6 board, 36 squares packed into uint64_t.
 *   - bit_index = x * N + y, matching the Python action encoding
 *     action = x * N + y from OthelloGame.getNextState().
 *   - Board is stored in CANONICAL form: p0 is the side-to-move's stones,
 *     p1 is the opponent's stones. othello_apply() flips sides automatically.
 *   - Action n*n is the pass action (matches Python).
 */

#define OTHELLO_N            6
#define OTHELLO_NN           (OTHELLO_N * OTHELLO_N)
#define OTHELLO_ACTION_SIZE  (OTHELLO_NN + 1)
#define OTHELLO_PASS_ACTION  OTHELLO_NN
#define OTHELLO_BOARD_MASK   ((1ULL << OTHELLO_NN) - 1ULL)

typedef struct {
    uint64_t p0;   /* current side-to-move's stones */
    uint64_t p1;   /* opponent's stones             */
} Board;

/* Index helpers */
static inline int othello_xy_to_action(int x, int y) { return x * OTHELLO_N + y; }
static inline int othello_action_to_x(int a)         { return a / OTHELLO_N; }
static inline int othello_action_to_y(int a)         { return a % OTHELLO_N; }

/* Initial board with side-to-move == +1 (matches Python canonical form). */
void othello_init(Board* b);

/* Bitmask of legal non-pass moves for side-to-move. Pass is legal iff this is 0. */
uint64_t othello_legal_mask(const Board* b);

/* Returns true iff side-to-move has no legal non-pass move. */
bool othello_must_pass(const Board* b);

/* Fill out a (n*n + 1) {0,1} valid-move vector. Pass slot set iff no other moves. */
void othello_valid_moves_vec(const Board* b, uint8_t out[OTHELLO_ACTION_SIZE]);

/*
 * Apply action (must be legal, or PASS). Flips opponent stones along the move's rays
 * and SWAPS p0/p1 so the board stays in canonical form for the next side-to-move.
 */
void othello_apply(Board* b, int action);

/*
 * Game-end check. Mirrors OthelloGame.getGameEnded(board, 1):
 *   0   game ongoing (either side has a non-pass move)
 *   +1  side-to-move won by piece count
 *   -1  side-to-move lost OR drew (Python returns -1 for draws)
 */
int othello_terminal(const Board* b);

/* Piece-count differential from side-to-move's POV (own - opp). */
int othello_score_diff(const Board* b);

/* Conversion to/from a Python-style int8 (n*n) array indexed [x*N + y]. */
void othello_from_int8(Board* out, const int8_t* arr_xy);
void othello_to_int8(const Board* b, int8_t* arr_xy);

/* Pretty-print a canonical board to stderr. Useful for tests. */
void othello_debug_print(const Board* b);

#endif /* OTHELLO_C_OTHELLO_H */
