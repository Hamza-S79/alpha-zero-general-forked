#include "othello.h"

#include <stdio.h>
#include <string.h>

/*
 * Bitboard layout (6x6, N=6):
 *
 *   bit index = x * N + y, with x as column and y as row.
 *
 *   Column X mask:  0x3F << (X * N)            (6 bits, the column's rows 0..5)
 *   Row    Y mask:  0x041041041 << Y           (every N-th bit set)
 *
 *   Direction shifts:
 *     +y  (south): bb << 1            (pre-mask out row N-1)
 *     -y  (north): bb >> 1            (pre-mask out row 0)
 *     +x  (east):  bb << N            (pre-mask out col N-1)
 *     -x  (west):  bb >> N            (pre-mask out col 0)
 *     +x+y (SE):   bb << (N+1)        (pre-mask out col N-1 and row N-1)
 *     +x-y (NE):   bb << (N-1)        (pre-mask out col N-1 and row 0)
 *     -x+y (SW):   bb >> (N-1)        (pre-mask out col 0 and row N-1)
 *     -x-y (NW):   bb >> (N+1)        (pre-mask out col 0 and row 0)
 *
 *   Pre-masking before shifting keeps bits from wrapping into a neighbouring
 *   column or row. After pre-mask, no extra post-mask is needed because all
 *   surviving bits stay inside [0, N*N).
 */

#define N  OTHELLO_N

/* Column masks: column X is bits X*N .. X*N + N-1. */
#define COL_MASK(X) (((uint64_t)0x3F) << ((X) * N))

/* Row masks: row Y is bits Y, Y+N, Y+2N, ..., Y+(N-1)*N. */
#define ROW_BASE   ((uint64_t)0x041041041ULL)   /* bits 0,6,12,18,24,30 */
#define ROW_MASK(Y) (ROW_BASE << (Y))

#define COL_FIRST  COL_MASK(0)
#define COL_LAST   COL_MASK(N - 1)
#define ROW_FIRST  ROW_MASK(0)
#define ROW_LAST   ROW_MASK(N - 1)

#define NOT_COL_FIRST (OTHELLO_BOARD_MASK & ~COL_FIRST)
#define NOT_COL_LAST  (OTHELLO_BOARD_MASK & ~COL_LAST)
#define NOT_ROW_FIRST (OTHELLO_BOARD_MASK & ~ROW_FIRST)
#define NOT_ROW_LAST  (OTHELLO_BOARD_MASK & ~ROW_LAST)

/* Direction shifters. Each clears the wrap-source bits, then shifts. */
static inline uint64_t sh_pos_y(uint64_t b)       { return (b & NOT_ROW_LAST)  << 1;       }
static inline uint64_t sh_neg_y(uint64_t b)       { return (b & NOT_ROW_FIRST) >> 1;       }
static inline uint64_t sh_pos_x(uint64_t b)       { return (b & NOT_COL_LAST)  << N;       }
static inline uint64_t sh_neg_x(uint64_t b)       { return (b & NOT_COL_FIRST) >> N;       }
static inline uint64_t sh_pos_x_pos_y(uint64_t b) { return (b & NOT_COL_LAST  & NOT_ROW_LAST)  << (N + 1); }
static inline uint64_t sh_pos_x_neg_y(uint64_t b) { return (b & NOT_COL_LAST  & NOT_ROW_FIRST) << (N - 1); }
static inline uint64_t sh_neg_x_pos_y(uint64_t b) { return (b & NOT_COL_FIRST & NOT_ROW_LAST)  >> (N - 1); }
static inline uint64_t sh_neg_x_neg_y(uint64_t b) { return (b & NOT_COL_FIRST & NOT_ROW_FIRST) >> (N + 1); }

typedef uint64_t (*shift_fn)(uint64_t);

static const shift_fn DIR_SHIFTS[8] = {
    sh_pos_y, sh_neg_y, sh_pos_x, sh_neg_x,
    sh_pos_x_pos_y, sh_pos_x_neg_y, sh_neg_x_pos_y, sh_neg_x_neg_y
};

void othello_init(Board* b) {
    /*
     * Python sets:
     *   pieces[N/2 - 1][N/2]     = +1
     *   pieces[N/2]    [N/2 - 1] = +1
     *   pieces[N/2 - 1][N/2 - 1] = -1
     *   pieces[N/2]    [N/2]     = -1
     * For N=6: (2,3)=+1, (3,2)=+1, (2,2)=-1, (3,3)=-1.
     * Side-to-move starts as +1, so canonical-form p0 = +1 stones.
     */
    int c = N / 2;
    uint64_t pos = (1ULL << ((c - 1) * N + c)) | (1ULL << (c * N + (c - 1)));
    uint64_t neg = (1ULL << ((c - 1) * N + (c - 1))) | (1ULL << (c * N + c));
    b->p0 = pos;
    b->p1 = neg;
}

/*
 * For each direction d, compute the set of empty squares that "cap" a contiguous
 * run of opponent stones starting from one of own's stones. The full move set
 * is the union over all 8 directions.
 *
 *   cand_0 = shift(own, d) & opp                       (one opp adjacent to own)
 *   cand_k = cand_{k-1} | (shift(cand_{k-1}, d) & opp) (extend the opp run)
 *   moves |= shift(cand_{N-2}, d) & empty              (cap the run with empty)
 *
 * N-2 extension steps are enough: the longest legal opp run in 6x6 is 4 stones
 * (since at least one own stone and one capping empty must fit on the line),
 * which we can build with 1 init + 3 extensions = N - 2 = 4 ANDs total. We do
 * N - 1 to keep margin without measurable cost.
 */
uint64_t othello_legal_mask(const Board* b) {
    uint64_t own = b->p0;
    uint64_t opp = b->p1;
    uint64_t empty = ~(own | opp) & OTHELLO_BOARD_MASK;
    uint64_t moves = 0;

    for (int d = 0; d < 8; d++) {
        shift_fn sh = DIR_SHIFTS[d];
        uint64_t cand = sh(own) & opp;
        for (int k = 0; k < N - 1; k++) {
            cand |= sh(cand) & opp;
        }
        moves |= sh(cand) & empty;
    }
    return moves;
}

bool othello_must_pass(const Board* b) {
    return othello_legal_mask(b) == 0;
}

void othello_valid_moves_vec(const Board* b, uint8_t out[OTHELLO_ACTION_SIZE]) {
    uint64_t legal = othello_legal_mask(b);
    memset(out, 0, OTHELLO_ACTION_SIZE);
    if (legal == 0) {
        out[OTHELLO_PASS_ACTION] = 1;
        return;
    }
    while (legal) {
        int bit = __builtin_ctzll(legal);
        out[bit] = 1;
        legal &= legal - 1;
    }
}

/*
 * Compute the bits to flip when own plays at move_bb in direction d:
 *   cand collects opp bits along the ray adjacent to move_bb. If the next
 *   step lands on an own stone, those opp bits flip.
 */
static inline uint64_t flips_in_direction(uint64_t move_bb, uint64_t own,
                                          uint64_t opp, shift_fn sh) {
    uint64_t cand = sh(move_bb) & opp;
    for (int k = 0; k < N - 1; k++) {
        cand |= sh(cand) & opp;
    }
    return (sh(cand) & own) ? cand : 0;
}

void othello_apply(Board* b, int action) {
    if (action == OTHELLO_PASS_ACTION) {
        /* Pass: no stones change, side-to-move flips. */
        uint64_t tmp = b->p0;
        b->p0 = b->p1;
        b->p1 = tmp;
        return;
    }

    uint64_t move_bb = 1ULL << action;
    uint64_t own = b->p0;
    uint64_t opp = b->p1;
    uint64_t flips = 0;

    for (int d = 0; d < 8; d++) {
        flips |= flips_in_direction(move_bb, own, opp, DIR_SHIFTS[d]);
    }

    own |= move_bb | flips;
    opp &= ~flips;

    /* Swap so canonical form follows side-to-move. */
    b->p0 = opp;
    b->p1 = own;
}

int othello_score_diff(const Board* b) {
    return __builtin_popcountll(b->p0) - __builtin_popcountll(b->p1);
}

int othello_terminal(const Board* b) {
    /* Game continues if current side has any non-pass move. */
    if (othello_legal_mask(b) != 0) return 0;
    /* Otherwise check the opponent. We flip the board temporarily. */
    Board flipped = { b->p1, b->p0 };
    if (othello_legal_mask(&flipped) != 0) return 0;
    /*
     * Both sides must pass: terminal. Python's getGameEnded returns -1 on draw.
     * We match that: countDiff > 0 → +1, else -1 (covers loss AND draw).
     */
    return (othello_score_diff(b) > 0) ? +1 : -1;
}

void othello_from_int8(Board* out, const int8_t* arr_xy) {
    /* arr_xy is row-major in (x, y): arr_xy[x * N + y] holds piece at (x, y).
     * Values: +1 = side-to-move's stone (p0), -1 = opponent (p1), 0 = empty.
     * Python's canonical-form board has the side-to-move's stones at +1.
     */
    uint64_t p0 = 0, p1 = 0;
    for (int i = 0; i < OTHELLO_NN; i++) {
        if (arr_xy[i] == +1) p0 |= (1ULL << i);
        else if (arr_xy[i] == -1) p1 |= (1ULL << i);
    }
    out->p0 = p0;
    out->p1 = p1;
}

void othello_to_int8(const Board* b, int8_t* arr_xy) {
    for (int i = 0; i < OTHELLO_NN; i++) {
        uint64_t bit = 1ULL << i;
        if (b->p0 & bit)      arr_xy[i] = +1;
        else if (b->p1 & bit) arr_xy[i] = -1;
        else                   arr_xy[i] = 0;
    }
}

void othello_debug_print(const Board* b) {
    fprintf(stderr, "    ");
    for (int y = 0; y < N; y++) fprintf(stderr, "%d ", y);
    fprintf(stderr, "\n   ");
    for (int y = 0; y < N; y++) fprintf(stderr, "--");
    fprintf(stderr, "-\n");
    for (int y = 0; y < N; y++) {
        fprintf(stderr, "%d | ", y);
        for (int x = 0; x < N; x++) {
            uint64_t bit = 1ULL << (x * N + y);
            char c = '.';
            if (b->p0 & bit) c = 'O';
            else if (b->p1 & bit) c = 'X';
            fprintf(stderr, "%c ", c);
        }
        fprintf(stderr, "|\n");
    }
}
