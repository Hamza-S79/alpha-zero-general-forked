/*
 * Smoke test for zobrist + tree. Generates many distinct boards by random
 * play and verifies:
 *   - same board hashes to same value;
 *   - distinct boards hash to distinct values (no false collisions);
 *   - tree get/insert/lookup behave consistently.
 *
 * Build:
 *   cc -std=c11 -O2 -Wall -Wextra -I include \
 *      src/othello.c src/zobrist.c src/tree.c tests/test_tree.c -o build/test_tree
 */

#include "othello.h"
#include "tree.h"
#include "zobrist.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Tiny xorshift for deterministic random play */
static uint64_t xs_state;
static uint64_t xs_next(void) {
    uint64_t x = xs_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    xs_state = x;
    return x;
}

static int random_legal_action(const Board* b) {
    uint64_t legal = othello_legal_mask(b);
    if (legal == 0) return OTHELLO_PASS_ACTION;
    int n = __builtin_popcountll(legal);
    int pick = (int)(xs_next() % (uint64_t)n);
    while (pick--) legal &= legal - 1;
    return __builtin_ctzll(legal);
}

int main(void) {
    zobrist_init();

    /* Sanity: same input → same hash; distinct inputs → distinct hashes. */
    Board b1; othello_init(&b1);
    Board b2; othello_init(&b2);
    if (zobrist_hash(&b1) != zobrist_hash(&b2)) {
        fprintf(stderr, "zobrist not deterministic\n"); return 1;
    }
    Board b3 = b1;
    othello_apply(&b3, __builtin_ctzll(othello_legal_mask(&b3)));
    if (zobrist_hash(&b1) == zobrist_hash(&b3)) {
        fprintf(stderr, "zobrist collision on adjacent positions\n"); return 1;
    }

    /* Larger sweep: collect 50k unique boards from random self-play, store
     * (key, p0, p1) in a tree-like dedup. Verify no two distinct (p0, p1)
     * pairs share a key. */
    Tree t;
    tree_init(&t, 1ULL << 17); /* 131k slots */

    xs_state = 0xC0FFEEULL;
    int episodes = 5000;
    int total_boards = 0, dup_states = 0, hash_collisions = 0;
    for (int e = 0; e < episodes; e++) {
        Board b; othello_init(&b);
        while (1) {
            uint64_t key = zobrist_hash(&b);
            bool inserted;
            TreeNode* n = tree_get_or_insert(&t, key, &inserted);
            if (inserted) {
                n->board = b;
            } else if (n->board.p0 != b.p0 || n->board.p1 != b.p1) {
                hash_collisions++;
                fprintf(stderr, "HASH COLLISION on key=%016llx:\n",
                        (unsigned long long)key);
                othello_debug_print(&n->board);
                fprintf(stderr, "vs.\n");
                othello_debug_print(&b);
                return 1;
            } else {
                dup_states++;
            }
            total_boards++;

            if (othello_terminal(&b) != 0) break;
            int a = random_legal_action(&b);
            othello_apply(&b, a);
            if (total_boards >= 50000) goto done;
        }
    }
done:
    printf("OK: %d boards visited, %llu unique, %d dup-state hits, %d hash collisions\n",
           total_boards, (unsigned long long)t.count, dup_states, hash_collisions);

    /* Tree sanity: every inserted key should be findable. */
    int verified = 0;
    for (uint64_t i = 0; i < t.capacity; i++) {
        if (t.slots[i].key != 0) {
            TreeNode* n = tree_lookup(&t, t.slots[i].key);
            if (n != &t.slots[i]) {
                fprintf(stderr, "tree_lookup returned wrong slot\n"); return 1;
            }
            verified++;
        }
    }
    printf("Tree lookup: %d slots verified\n", verified);

    tree_free(&t);
    return 0;
}
