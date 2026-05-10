#include "tree.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int is_pow2(uint64_t x) { return x && (x & (x - 1)) == 0; }

void tree_init(Tree* t, uint64_t init_capacity_pow2) {
    if (!is_pow2(init_capacity_pow2)) {
        fprintf(stderr, "tree_init: capacity must be a power of 2 (got %llu)\n",
                (unsigned long long)init_capacity_pow2);
        abort();
    }
    t->slots = calloc((size_t)init_capacity_pow2, sizeof(TreeNode));
    if (!t->slots) {
        fprintf(stderr, "tree_init: calloc failed\n");
        abort();
    }
    t->capacity = init_capacity_pow2;
    t->mask     = init_capacity_pow2 - 1;
    t->count    = 0;
}

void tree_free(Tree* t) {
    if (!t->slots) return;
    for (uint64_t i = 0; i < t->capacity; i++) {
        free(t->slots[i].edges);
    }
    free(t->slots);
    t->slots = NULL;
    t->capacity = 0;
    t->mask = 0;
    t->count = 0;
}

TreeNode* tree_get_or_insert(Tree* t, uint64_t key, bool* out_inserted) {
    if (key == 0) key = 1; /* reserve 0 as empty sentinel */
    /* Cap at 75% load factor; refuse to insert past that. Caller sizes the
     * table large enough that this never trips during a real run. */
    if (t->count * 4 >= t->capacity * 3) {
        fprintf(stderr, "tree: full (count=%llu cap=%llu) — increase init_capacity\n",
                (unsigned long long)t->count, (unsigned long long)t->capacity);
        abort();
    }
    uint64_t i = key & t->mask;
    while (1) {
        if (t->slots[i].key == 0) {
            t->slots[i].key = key;
            t->count++;
            if (out_inserted) *out_inserted = true;
            return &t->slots[i];
        }
        if (t->slots[i].key == key) {
            if (out_inserted) *out_inserted = false;
            return &t->slots[i];
        }
        i = (i + 1) & t->mask;
    }
}

TreeNode* tree_lookup(const Tree* t, uint64_t key) {
    if (key == 0) key = 1;
    uint64_t i = key & t->mask;
    while (1) {
        if (t->slots[i].key == 0) return NULL;
        if (t->slots[i].key == key) return &t->slots[i];
        i = (i + 1) & t->mask;
    }
}
