#ifndef OTHELLO_C_TREE_H
#define OTHELLO_C_TREE_H

#include <stdbool.h>
#include <stdint.h>

#include "othello.h"

/*
 * Open-addressing hash table keyed by 64-bit Zobrist hash, payloaded with the
 * MCTS bookkeeping for one canonical-board state.
 *
 * Empty slot sentinel: key == 0. The probability that a real position hashes
 * to exactly 0 is 1/2^64; we ignore that.
 *
 * Invariant: edges[k].action is the action index in [0, OTHELLO_ACTION_SIZE).
 *   - For an expanded non-terminal node, edges covers the legal-move set
 *     densely (no slot per illegal action).
 *   - For a terminal or unexpanded node, edges == NULL and n_edges == 0.
 */

typedef struct {
    int32_t   N;          /* visit count            (Nsa)   */
    int32_t   in_flight;  /* virtual-loss in-flight (Insa)  */
    float     W;          /* sum of values          (Wsa)   */
    float     P;          /* prior from NN          (Ps)    */
    uint16_t  action;     /* action index in [0, ACTION_SIZE) */
} Edge;

typedef struct {
    uint64_t  key;          /* zobrist hash; 0 == empty slot                */
    Board     board;        /* canonical board                              */
    Edge*     edges;        /* legal moves, dense; NULL if not expanded     */
    uint16_t  n_edges;
    uint8_t   expanded;     /* priors installed?                            */
    uint8_t   is_terminal;
    int8_t    terminal_v;   /* {-1, 0, +1} from side-to-move's POV          */
    int32_t   Ns;           /* sum of edge.N (Ns)                           */
} TreeNode;

typedef struct {
    TreeNode* slots;
    uint64_t  capacity;     /* power of 2          */
    uint64_t  mask;         /* capacity - 1        */
    uint64_t  count;        /* used slots          */
} Tree;

void tree_init(Tree* t, uint64_t init_capacity_pow2);
void tree_free(Tree* t);

/*
 * Find the slot for `key`. If empty, returns the slot to fill in (caller sets
 * key/board/etc.) and *out_inserted = true. If occupied with the same key,
 * returns it and *out_inserted = false. Aborts if the table is too full to
 * insert (caller responsibility to size correctly).
 */
TreeNode* tree_get_or_insert(Tree* t, uint64_t key, bool* out_inserted);

/* Read-only lookup. Returns NULL if not present. */
TreeNode* tree_lookup(const Tree* t, uint64_t key);

#endif /* OTHELLO_C_TREE_H */
