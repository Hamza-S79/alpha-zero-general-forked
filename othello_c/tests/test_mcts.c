/*
 * Serial-MCTS parity test. Loads the fixture from python/dump_mcts_fixture.py
 * (Python BatchedMCTS in serial mode), runs the C MCTS with the same args
 * and the same .onnx model, and compares visit counts and pi vector.
 *
 * Build:
 *   cc -std=c11 -O2 -Wall -Wextra -I include -I/opt/homebrew/include/onnxruntime \
 *      src/othello.c src/zobrist.c src/tree.c src/nn.c src/mcts.c \
 *      tests/test_mcts.c \
 *      -L/opt/homebrew/lib -lonnxruntime -lm -o build/test_mcts
 *
 * Run:
 *   build/test_mcts model.onnx fixture.bin
 */

#include "mcts.h"
#include "nn.h"
#include "othello.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_exact(FILE* f, void* buf, size_t n) {
    return fread(buf, 1, n, f) == n ? 0 : -1;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s model.onnx fixture.bin\n", argv[0]);
        return 2;
    }

    FILE* f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }

    int32_t n, asz, sims;
    float cpuct;
    if (read_exact(f, &n, 4) || read_exact(f, &asz, 4) || read_exact(f, &sims, 4)
     || read_exact(f, &cpuct, 4)) {
        fprintf(stderr, "short read on header\n"); return 2;
    }
    if (n != OTHELLO_N || asz != OTHELLO_ACTION_SIZE) {
        fprintf(stderr, "fixture shape mismatch\n"); return 2;
    }

    int8_t  canonical[OTHELLO_NN];
    int32_t py_visits[OTHELLO_ACTION_SIZE];
    float   py_pi[OTHELLO_ACTION_SIZE];
    if (read_exact(f, canonical, OTHELLO_NN)
     || read_exact(f, py_visits, OTHELLO_ACTION_SIZE * sizeof(int32_t))
     || read_exact(f, py_pi, OTHELLO_ACTION_SIZE * sizeof(float))) {
        fprintf(stderr, "short read on body\n"); return 2;
    }
    fclose(f);

    NN* nn;
    if (nn_load(&nn, argv[1], 0) != 0) return 2;

    Board root;
    othello_from_int8(&root, canonical);

    MCTSArgs args = {
        .numMCTSSims     = sims,
        .cpuct           = cpuct,
        .numParallelSims = 1,
        .virtualLoss     = 0.0f,
        .evalBatchSize   = 1,
        .evalTimeoutMs   = 1.0f,
        .treeCapacityPow2 = 14,   /* 16k slots; far more than serial sims */
    };
    MCTS* m = mcts_new(nn, args);

    float c_pi[OTHELLO_ACTION_SIZE];
    mcts_action_prob(m, &root, /*temp=*/1.0f, c_pi);

    /* Compare pi distributions and report. */
    float max_diff = 0.0f;
    for (int a = 0; a < OTHELLO_ACTION_SIZE; a++) {
        float d = fabsf(py_pi[a] - c_pi[a]);
        if (d > max_diff) max_diff = d;
    }

    /* Argmax (deterministic tie-break to smallest index) */
    int py_argmax = 0, c_argmax = 0;
    for (int a = 1; a < OTHELLO_ACTION_SIZE; a++) {
        if (py_pi[a] > py_pi[py_argmax]) py_argmax = a;
        if (c_pi[a]  > c_pi[c_argmax])   c_argmax  = a;
    }

    printf("sims=%d cpuct=%.2f\n", sims, cpuct);
    printf("max |pi_py - pi_c| = %.4e\n", max_diff);
    printf("argmax: py=%d c=%d\n", py_argmax, c_argmax);

    /* Visit-count totals (sanity). */
    int py_total = 0;
    for (int a = 0; a < OTHELLO_ACTION_SIZE; a++) py_total += py_visits[a];
    printf("python visit total: %d  (== sims expected: %d)\n", py_total, sims);

    /* Side-by-side visit/pi diff for non-zero rows */
    printf("non-zero pi rows:\n");
    printf("  %-5s %-10s %-10s %-10s\n", "a", "py_visits", "py_pi", "c_pi");
    for (int a = 0; a < OTHELLO_ACTION_SIZE; a++) {
        if (py_pi[a] > 1e-6f || c_pi[a] > 1e-6f) {
            printf("  %-5d %-10d %-10.4f %-10.4f\n", a, py_visits[a], py_pi[a], c_pi[a]);
        }
    }

    int rc = 0;
    /* Tolerance: argmax must agree, and max |pi_py - pi_c| should be small. */
    if (py_argmax != c_argmax) {
        printf("FAIL: argmax disagreement\n");
        rc = 1;
    } else if (max_diff > 0.05f) {
        printf("FAIL: pi divergence > 5%%\n");
        rc = 1;
    } else {
        printf("OK\n");
    }

    mcts_free(m);
    nn_free(nn);
    return rc;
}
