#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "arena.h"
#include "coach.h"
#include "mcts.h"
#include "nn.h"
#include "othello.h"

static void usage(const char* prog) {
    fprintf(stderr,
        "usage:\n"
        "  %s selfplay --model M.onnx --episodes N --sims S --workers W --batch B [--vloss V]\n"
        "                   [--temp-threshold T] [--seed S] [--out FILE] [--cuda]\n"
        "  %s arena    --a A.onnx --b B.onnx --games G --sims S --workers W --batch B [--vloss V]\n"
        "                   [--cuda]\n"
        "  %s bench-mcts --model M.onnx --sims S --workers W [--batch B] [--vloss V] [--cuda]\n",
        prog, prog, prog);
}

static int parse_int(const char* s, const char* name) {
    char* end;
    long v = strtol(s, &end, 10);
    if (*end != '\0' || end == s) {
        fprintf(stderr, "bad integer for --%s: %s\n", name, s);
        exit(2);
    }
    return (int)v;
}

static float parse_float(const char* s, const char* name) {
    char* end;
    float v = strtof(s, &end);
    if (*end != '\0' || end == s) {
        fprintf(stderr, "bad float for --%s: %s\n", name, s);
        exit(2);
    }
    return v;
}

#define ARG_STR(out, name) \
    do { if (i+1 >= argc) { fprintf(stderr,"missing --%s\n",name); return 2; } \
         out = argv[++i]; } while(0)
#define ARG_INT(out, name) \
    do { if (i+1 >= argc) { fprintf(stderr,"missing --%s\n",name); return 2; } \
         out = parse_int(argv[++i], name); } while(0)
#define ARG_FLOAT(out, name) \
    do { if (i+1 >= argc) { fprintf(stderr,"missing --%s\n",name); return 2; } \
         out = parse_float(argv[++i], name); } while(0)

static int cmd_selfplay(int argc, char** argv) {
    const char* model_path = NULL;
    const char* out_path = NULL;
    int episodes = 1, sims = 25, workers = 8, batch = 8, temp_thresh = 15, seed = 0;
    float vloss = 1.0f;
    int use_cuda = 0;
    for (int i = 0; i < argc; i++) {
        if      (!strcmp(argv[i], "--model"))    ARG_STR(model_path, "model");
        else if (!strcmp(argv[i], "--out"))      ARG_STR(out_path, "out");
        else if (!strcmp(argv[i], "--episodes")) ARG_INT(episodes, "episodes");
        else if (!strcmp(argv[i], "--sims"))     ARG_INT(sims, "sims");
        else if (!strcmp(argv[i], "--workers"))  ARG_INT(workers, "workers");
        else if (!strcmp(argv[i], "--batch"))    ARG_INT(batch, "batch");
        else if (!strcmp(argv[i], "--vloss"))    ARG_FLOAT(vloss, "vloss");
        else if (!strcmp(argv[i], "--temp-threshold")) ARG_INT(temp_thresh, "temp-threshold");
        else if (!strcmp(argv[i], "--seed"))     ARG_INT(seed, "seed");
        else if (!strcmp(argv[i], "--cuda"))     use_cuda = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!model_path || !out_path) { fprintf(stderr, "--model and --out required\n"); return 2; }

    NN* nn;
    if (nn_load(&nn, model_path, use_cuda) != 0) return 2;

    FILE* out = fopen(out_path, "wb");
    if (!out) { perror(out_path); return 2; }
    coach_write_header(out, OTHELLO_N);

    CoachArgs ca = {
        .mcts_args = {
            .numMCTSSims = sims, .cpuct = 1.0f,
            .numParallelSims = workers, .virtualLoss = vloss,
            .evalBatchSize = batch, .evalTimeoutMs = 1.0f,
            .treeCapacityPow2 = 17,
        },
        .tempThreshold = temp_thresh,
        .rng_seed = (unsigned)seed,
    };

    int total_examples = 0;
    double t0 = (double)clock() / CLOCKS_PER_SEC;
    for (int e = 0; e < episodes; e++) {
        ca.rng_seed = (unsigned)seed + (unsigned)e;
        int n_before = total_examples;
        if (coach_run_episode(nn, ca, out, &total_examples) != 0) {
            fprintf(stderr, "episode %d failed\n", e);
            fclose(out);
            return 1;
        }
        printf("episode %d/%d: %d moves, total %d examples\n",
               e + 1, episodes, total_examples - n_before, total_examples);
    }
    double t1 = (double)clock() / CLOCKS_PER_SEC;
    coach_finalize_header(out, total_examples);
    fclose(out);
    nn_free(nn);
    printf("wrote %d examples in %.1fs (%.2fs/episode)\n",
           total_examples, t1 - t0, (t1 - t0) / (double)episodes);
    return 0;
}

static int cmd_arena(int argc, char** argv) {
    const char* a_path = NULL, *b_path = NULL;
    int games = 20, sims = 25, workers = 8, batch = 8;
    float vloss = 1.0f;
    int use_cuda = 0;
    for (int i = 0; i < argc; i++) {
        if      (!strcmp(argv[i], "--a"))       ARG_STR(a_path, "a");
        else if (!strcmp(argv[i], "--b"))       ARG_STR(b_path, "b");
        else if (!strcmp(argv[i], "--games"))   ARG_INT(games, "games");
        else if (!strcmp(argv[i], "--sims"))    ARG_INT(sims, "sims");
        else if (!strcmp(argv[i], "--workers")) ARG_INT(workers, "workers");
        else if (!strcmp(argv[i], "--batch"))   ARG_INT(batch, "batch");
        else if (!strcmp(argv[i], "--vloss"))   ARG_FLOAT(vloss, "vloss");
        else if (!strcmp(argv[i], "--cuda"))    use_cuda = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!a_path || !b_path) { fprintf(stderr, "--a and --b required\n"); return 2; }

    NN *na, *nb;
    if (nn_load(&na, a_path, use_cuda) != 0) return 2;
    if (nn_load(&nb, b_path, use_cuda) != 0) return 2;

    MCTSArgs args = {
        .numMCTSSims = sims, .cpuct = 1.0f,
        .numParallelSims = workers, .virtualLoss = vloss,
        .evalBatchSize = batch, .evalTimeoutMs = 1.0f,
        .treeCapacityPow2 = 18,
    };

    double t0 = (double)clock() / CLOCKS_PER_SEC;
    ArenaResult r = arena_play(na, nb, args, games);
    double t1 = (double)clock() / CLOCKS_PER_SEC;
    printf("Arena: A wins %d, B wins %d, draws %d (%d games, %.1fs)\n",
           r.wins_a, r.wins_b, r.draws, games, t1 - t0);

    nn_free(na);
    nn_free(nb);
    return 0;
}

static int cmd_bench(int argc, char** argv) {
    const char* model_path = NULL;
    int sims = 50, workers = 8, batch = 8;
    float vloss = 1.0f;
    int use_cuda = 0;
    for (int i = 0; i < argc; i++) {
        if      (!strcmp(argv[i], "--model"))   ARG_STR(model_path, "model");
        else if (!strcmp(argv[i], "--sims"))    ARG_INT(sims, "sims");
        else if (!strcmp(argv[i], "--workers")) ARG_INT(workers, "workers");
        else if (!strcmp(argv[i], "--batch"))   ARG_INT(batch, "batch");
        else if (!strcmp(argv[i], "--vloss"))   ARG_FLOAT(vloss, "vloss");
        else if (!strcmp(argv[i], "--cuda"))    use_cuda = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!model_path) { fprintf(stderr, "--model required\n"); return 2; }

    NN* nn;
    if (nn_load(&nn, model_path, use_cuda) != 0) return 2;

    /* Same fixed root as benchmark_mcts.py / test_mcts_threaded.c: two
     * opening moves applied to the initial board. */
    Board root; othello_init(&root);
    int a = __builtin_ctzll(othello_legal_mask(&root)); othello_apply(&root, a);
    a = __builtin_ctzll(othello_legal_mask(&root));     othello_apply(&root, a);

    MCTSArgs args = {
        .numMCTSSims = sims, .cpuct = 1.0f,
        .numParallelSims = workers, .virtualLoss = vloss,
        .evalBatchSize = batch, .evalTimeoutMs = 1.0f,
        .treeCapacityPow2 = 16,
    };
    MCTS* m = mcts_new(nn, args);

    float pi[OTHELLO_ACTION_SIZE];
    mcts_action_prob(m, &root, 1.0f, pi);
    MCTSStats st; mcts_get_stats(m, &st);

    printf("workers=%d batch=%d vloss=%.1f sims=%d:\n",
           workers, batch, vloss, sims);
    printf("  wall=%.3fs  gpu_calls=%lld  mean_batch=%.2f  collisions=%lld\n",
           st.total_seconds,
           (long long)st.n_gpu_calls,
           st.n_gpu_calls > 0 ? (double)st.total_batch_size / (double)st.n_gpu_calls : 0.0,
           (long long)st.n_virtual_loss_collisions);
    int argmax = 0;
    for (int i = 1; i < OTHELLO_ACTION_SIZE; i++) if (pi[i] > pi[argmax]) argmax = i;
    printf("  argmax=%d  pi[argmax]=%.4f\n", argmax, pi[argmax]);

    mcts_free(m);
    nn_free(nn);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 2; }
    if      (!strcmp(argv[1], "selfplay"))   return cmd_selfplay(argc - 2, argv + 2);
    else if (!strcmp(argv[1], "arena"))      return cmd_arena(argc - 2, argv + 2);
    else if (!strcmp(argv[1], "bench-mcts")) return cmd_bench(argc - 2, argv + 2);
    else if (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
        usage(argv[0]); return 0;
    }
    fprintf(stderr, "unknown subcommand: %s\n", argv[1]);
    usage(argv[0]);
    return 2;
}
