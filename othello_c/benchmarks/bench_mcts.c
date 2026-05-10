/*
 * C-side analogue of /benchmark_mcts.py — sweeps (workers, batch_size) and
 * prints a table directly comparable to the Python output.
 *
 * Build:
 *   cc -std=c11 -O3 -Wall -Wextra -I include -I/opt/homebrew/include/onnxruntime \
 *      src/othello.c src/zobrist.c src/tree.c src/nn.c src/mcts.c \
 *      benchmarks/bench_mcts.c \
 *      -L/opt/homebrew/lib -lonnxruntime -lpthread -lm -o build/bench_mcts
 *
 * Run (defaults match benchmark_mcts.py):
 *   build/bench_mcts --model M.onnx --sims 50 --workers 1,2,4,8,16 --batches 1,4,8,16
 */

#include "mcts.h"
#include "nn.h"
#include "othello.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int parse_int_list(const char* s, int* out, int max) {
    int n = 0;
    const char* p = s;
    while (*p && n < max) {
        char* end;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        out[n++] = (int)v;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    return n;
}

static Board fixed_root(void) {
    Board b; othello_init(&b);
    int a = __builtin_ctzll(othello_legal_mask(&b)); othello_apply(&b, a);
    a = __builtin_ctzll(othello_legal_mask(&b));     othello_apply(&b, a);
    return b;
}

static double bench_config(NN* nn, int sims, int workers, int batch,
                           float vloss, int repeats, MCTSStats* out_stats) {
    Board root = fixed_root();
    double best = 1e9;
    MCTSStats best_stats = {0};
    for (int r = 0; r < repeats; r++) {
        MCTSArgs args = {
            .numMCTSSims = sims, .cpuct = 1.0f,
            .numParallelSims = workers, .virtualLoss = vloss,
            .evalBatchSize = batch, .evalTimeoutMs = 1.0f,
            .treeCapacityPow2 = 16,
        };
        MCTS* m = mcts_new(nn, args);
        double t0 = now_seconds();
        float pi[OTHELLO_ACTION_SIZE];
        mcts_action_prob(m, &root, 1.0f, pi);
        double t1 = now_seconds();
        if (t1 - t0 < best) {
            best = t1 - t0;
            mcts_get_stats(m, &best_stats);
        }
        mcts_free(m);
    }
    if (out_stats) *out_stats = best_stats;
    return best;
}

int main(int argc, char** argv) {
    const char* model_path = NULL;
    int sims = 50, repeats = 3;
    float vloss = 1.0f;
    int workers[8] = {1, 2, 4, 8, 16, 0};
    int batches[8] = {1, 4, 8, 16, 0};
    int n_workers = 5, n_batches = 4;
    int use_cuda = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i+1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--sims") && i+1 < argc) sims = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--repeat") && i+1 < argc) repeats = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vloss") && i+1 < argc) vloss = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--workers") && i+1 < argc)
            n_workers = parse_int_list(argv[++i], workers, 8);
        else if (!strcmp(argv[i], "--batches") && i+1 < argc)
            n_batches = parse_int_list(argv[++i], batches, 8);
        else if (!strcmp(argv[i], "--cuda")) use_cuda = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!model_path) { fprintf(stderr, "--model required\n"); return 2; }

    NN* nn;
    if (nn_load(&nn, model_path, use_cuda) != 0) return 2;

    /* Warm up to amortize first-call overhead (CUDA init etc.). */
    bench_config(nn, 4, 1, 1, 0.0f, 1, NULL);

    /* Baseline: serial single-thread. */
    MCTSStats base_stats;
    double t_base = bench_config(nn, sims, 1, 1, 0.0f, repeats, &base_stats);
    printf("Baseline (workers=1, batch=1, vloss=0): wall=%.1fms  gpu_calls=%lld\n",
           t_base * 1000.0, (long long)base_stats.n_gpu_calls);
    printf("\n");

    printf("BatchedMCTS sweep (sims=%d, vloss=%.1f, repeats=%d)\n", sims, vloss, repeats);
    printf("%7s %6s %9s %8s %9s %9s %10s\n",
           "Workers", "Batch", "Wall(ms)", "Speedup", "GPU calls", "MeanBatch", "Collisions");
    printf("------- ------ --------- -------- --------- --------- ----------\n");

    for (int wi = 0; wi < n_workers; wi++) {
        for (int bi = 0; bi < n_batches; bi++) {
            int W = workers[wi], B = batches[bi];
            MCTSStats st;
            double t = bench_config(nn, sims, W, B, vloss, repeats, &st);
            double mean_batch = st.n_gpu_calls > 0
                              ? (double)st.total_batch_size / (double)st.n_gpu_calls
                              : 0.0;
            double speedup = t > 0 ? t_base / t : 0.0;
            printf("%7d %6d %9.1f %7.2fx %9lld %9.2f %10lld\n",
                   W, B, t * 1000.0, speedup,
                   (long long)st.n_gpu_calls, mean_batch,
                   (long long)st.n_virtual_loss_collisions);
        }
    }
    printf("\n");
    printf("Notes:\n");
    printf("  - Single-thread C vs single-thread Python is the constant-factor\n");
    printf("    speedup; threading on top is the parallel-descent speedup.\n");
    printf("  - Run /benchmark_mcts.py with the same --sims for direct comparison.\n");

    nn_free(nn);
    return 0;
}
