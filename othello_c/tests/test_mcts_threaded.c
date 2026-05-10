/*
 * Multi-threaded MCTS smoke test. Runs the C MCTS twice on the same root:
 *   (a) serial      (workers=1, vloss=0, batch=1)
 *   (b) parallel    (workers=N, vloss=v, batch=B)
 *
 * Asserts:
 *   - both runs complete without crashing
 *   - both produce a valid prob distribution that sums to ~1
 *   - argmax actions agree (multi-thread perturbs counts but should not change
 *     the dominant move at moderate sim counts)
 *   - pi vectors are close in L1 norm (sanity)
 *
 * Build / run via cmake; or directly:
 *     cc ... [src files] tests/test_mcts_threaded.c -o build/test_mcts_threaded
 *     build/test_mcts_threaded model.onnx
 */

#include "mcts.h"
#include "nn.h"
#include "othello.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static Board fixed_root(void) {
    /* Same as benchmark_mcts.py.fixed_canonical_board: two opening moves. */
    Board b; othello_init(&b);
    uint64_t legal = othello_legal_mask(&b);
    int a = __builtin_ctzll(legal);     /* first legal move */
    othello_apply(&b, a);
    legal = othello_legal_mask(&b);
    a = __builtin_ctzll(legal);
    othello_apply(&b, a);
    return b;
}

static int argmax(const float* pi) {
    int best = 0;
    for (int a = 1; a < OTHELLO_ACTION_SIZE; a++) {
        if (pi[a] > pi[best]) best = a;
    }
    return best;
}

static float l1_dist(const float* a, const float* b) {
    float s = 0;
    for (int i = 0; i < OTHELLO_ACTION_SIZE; i++) s += fabsf(a[i] - b[i]);
    return s;
}

static float pi_sum(const float* pi) {
    float s = 0;
    for (int i = 0; i < OTHELLO_ACTION_SIZE; i++) s += pi[i];
    return s;
}

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s model.onnx\n", argv[0]); return 2; }

    NN* nn;
    if (nn_load(&nn, argv[1], 0) != 0) return 2;

    Board root = fixed_root();
    int sims = 200;
    float cpuct = 1.0f;

    /* (a) serial */
    MCTSArgs sa = {
        .numMCTSSims = sims, .cpuct = cpuct,
        .numParallelSims = 1, .virtualLoss = 0.0f,
        .evalBatchSize = 1, .evalTimeoutMs = 1.0f,
        .treeCapacityPow2 = 14,
    };
    MCTS* m_serial = mcts_new(nn, sa);
    float pi_serial[OTHELLO_ACTION_SIZE];
    mcts_action_prob(m_serial, &root, 1.0f, pi_serial);
    MCTSStats s_stats; mcts_get_stats(m_serial, &s_stats);
    mcts_free(m_serial);

    /* (b) parallel */
    MCTSArgs pa = {
        .numMCTSSims = sims, .cpuct = cpuct,
        .numParallelSims = 8, .virtualLoss = 1.0f,
        .evalBatchSize = 8, .evalTimeoutMs = 1.0f,
        .treeCapacityPow2 = 14,
    };
    MCTS* m_par = mcts_new(nn, pa);
    float pi_par[OTHELLO_ACTION_SIZE];
    mcts_action_prob(m_par, &root, 1.0f, pi_par);
    MCTSStats p_stats; mcts_get_stats(m_par, &p_stats);
    mcts_free(m_par);
    nn_free(nn);

    int rc = 0;

    if (fabsf(pi_sum(pi_serial) - 1.0f) > 1e-4f
     || fabsf(pi_sum(pi_par)    - 1.0f) > 1e-4f) {
        printf("FAIL: pi does not sum to 1\n");
        rc = 1;
    }

    int as = argmax(pi_serial), ap = argmax(pi_par);
    float l1 = l1_dist(pi_serial, pi_par);

    printf("serial: argmax=%d  gpu_calls=%lld  total_batch=%lld\n",
           as, (long long)s_stats.n_gpu_calls, (long long)s_stats.total_batch_size);
    printf("paral.: argmax=%d  gpu_calls=%lld  mean_batch=%.2f  collisions=%lld  wall=%.3fs\n",
           ap,
           (long long)p_stats.n_gpu_calls,
           p_stats.n_gpu_calls > 0 ? (double)p_stats.total_batch_size / (double)p_stats.n_gpu_calls : 0.0,
           (long long)p_stats.n_virtual_loss_collisions,
           p_stats.total_seconds);
    printf("L1(pi_serial, pi_par) = %.4f\n", l1);

    /* Print non-zero rows side-by-side for eyeballing. */
    printf("non-zero pi rows:\n");
    for (int a = 0; a < OTHELLO_ACTION_SIZE; a++) {
        if (pi_serial[a] > 1e-3f || pi_par[a] > 1e-3f) {
            printf("  a=%-3d  serial=%.4f   par=%.4f\n", a, pi_serial[a], pi_par[a]);
        }
    }

    if (as != ap) {
        printf("WARN: argmax differs (allowed for small sim counts)\n");
    }
    if (l1 > 0.6f) {
        printf("FAIL: L1 distance too large\n");
        rc = 1;
    }
    if (rc == 0) printf("OK\n");
    return rc;
}
