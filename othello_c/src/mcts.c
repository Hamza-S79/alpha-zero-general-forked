#include "mcts.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "zobrist.h"

#define MAX_DEPTH       256
#define EVAL_QUEUE_CAP  4096      /* must be > workers, comfortably             */

/*
 * Leaf-parallel MCTS with virtual loss + a single batcher thread.
 *
 * Concurrency model (mirrors BatchedMCTS.py):
 *   - One global tree_lock guards every read/write to TreeNode/Edge fields.
 *     Acquired at each "step" of descent; released around the NN evaluation
 *     so workers don't block each other on GPU calls.
 *   - One request queue feeds the batcher. Workers push EvalRequest pointers,
 *     batcher pops up to evalBatchSize within evalTimeoutMs, runs one
 *     nn_predict_batch, signals each request's cond.
 *   - Per-request lock+cond for completion handshake.
 *
 * With numParallelSims == 1 and virtualLoss == 0, this still spawns one
 * worker + one batcher thread, but the bookkeeping is identical to the
 * single-threaded path and reproduces MCTS.py's visit counts.
 *
 * For pure single-threaded use we skip the threads entirely (search_serial)
 * — that path is what test_mcts.c covers and stays bit-exact with Python.
 */

typedef struct {
    Board  board;
    float  pi[OTHELLO_ACTION_SIZE];
    float  v;
    int    done;
    pthread_mutex_t lock;
    pthread_cond_t  cond;
} EvalRequest;

typedef struct {
    EvalRequest* items[EVAL_QUEUE_CAP];
    int head, tail, count;
    pthread_mutex_t lock;
    pthread_cond_t  cond_nonempty;
    int stop;                  /* set after all workers join: batcher drains then exits */
} ReqQueue;

struct MCTS {
    NN*        nn;
    MCTSArgs   args;
    Tree       tree;

    /* Mutated only in mcts_action_prob path; non-atomic OK because
     * mcts_action_prob is single-entry per MCTS instance. */
    MCTSStats  stats;

    /* Tree lock + queue are persistent so we don't pay setup cost per call. */
    pthread_mutex_t tree_lock;
    ReqQueue   queue;

    /* Atomic counters mutated under tree_lock OR by batcher. */
    atomic_long stat_gpu_calls;
    atomic_long stat_total_batch;
    atomic_long stat_collisions;
};

typedef struct {
    TreeNode* node;
    int       edge_idx;
} PathEntry;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------------- */
/* Queue                                                                      */
/* ------------------------------------------------------------------------- */

static void queue_init(ReqQueue* q) {
    q->head = q->tail = q->count = 0;
    q->stop = 0;
    pthread_mutex_init(&q->lock, NULL);
    pthread_cond_init(&q->cond_nonempty, NULL);
}

static void queue_destroy(ReqQueue* q) {
    pthread_mutex_destroy(&q->lock);
    pthread_cond_destroy(&q->cond_nonempty);
}

static void queue_push(ReqQueue* q, EvalRequest* r) {
    pthread_mutex_lock(&q->lock);
    if (q->count >= EVAL_QUEUE_CAP) {
        fprintf(stderr, "mcts: eval queue full (cap=%d). Increase.\n", EVAL_QUEUE_CAP);
        abort();
    }
    q->items[q->tail] = r;
    q->tail = (q->tail + 1) % EVAL_QUEUE_CAP;
    q->count++;
    pthread_cond_signal(&q->cond_nonempty);
    pthread_mutex_unlock(&q->lock);
}

/* Pop one item, blocking until available OR stop is set with empty queue.
 * Returns NULL when shutting down. */
static EvalRequest* queue_pop_blocking(ReqQueue* q) {
    pthread_mutex_lock(&q->lock);
    while (q->count == 0 && !q->stop) {
        pthread_cond_wait(&q->cond_nonempty, &q->lock);
    }
    if (q->count == 0 && q->stop) {
        pthread_mutex_unlock(&q->lock);
        return NULL;
    }
    EvalRequest* r = q->items[q->head];
    q->head = (q->head + 1) % EVAL_QUEUE_CAP;
    q->count--;
    pthread_mutex_unlock(&q->lock);
    return r;
}

/* Pop with deadline (wall-clock seconds). Returns NULL on timeout (queue empty)
 * or when stop is set with empty queue. */
static EvalRequest* queue_pop_until(ReqQueue* q, double deadline_s) {
    pthread_mutex_lock(&q->lock);
    while (q->count == 0 && !q->stop) {
        struct timespec ts;
        ts.tv_sec  = (time_t)deadline_s;
        ts.tv_nsec = (long)((deadline_s - (double)ts.tv_sec) * 1e9);
        int rc = pthread_cond_timedwait(&q->cond_nonempty, &q->lock, &ts);
        if (rc != 0) break;  /* timeout or error → bail to caller, may retry */
    }
    if (q->count == 0) {
        pthread_mutex_unlock(&q->lock);
        return NULL;
    }
    EvalRequest* r = q->items[q->head];
    q->head = (q->head + 1) % EVAL_QUEUE_CAP;
    q->count--;
    pthread_mutex_unlock(&q->lock);
    return r;
}

static void queue_signal_stop(ReqQueue* q) {
    pthread_mutex_lock(&q->lock);
    q->stop = 1;
    pthread_cond_broadcast(&q->cond_nonempty);
    pthread_mutex_unlock(&q->lock);
}

static void queue_clear_stop(ReqQueue* q) {
    pthread_mutex_lock(&q->lock);
    q->stop = 0;
    pthread_mutex_unlock(&q->lock);
}

/* ------------------------------------------------------------------------- */
/* MCTS shared helpers (used by both serial and parallel paths)              */
/* ------------------------------------------------------------------------- */

static void install_priors(TreeNode* node, const Board* b, const float* pi) {
    uint64_t legal = othello_legal_mask(b);
    int n_edges;
    Edge* edges;

    if (legal == 0) {
        n_edges = 1;
        edges = malloc(sizeof(Edge));
        edges[0].action = OTHELLO_PASS_ACTION;
        edges[0].P = pi[OTHELLO_PASS_ACTION];
        if (edges[0].P <= 0.0f) edges[0].P = 1.0f;
        edges[0].N = 0;
        edges[0].W = 0.0f;
        edges[0].in_flight = 0;
    } else {
        n_edges = __builtin_popcountll(legal);
        edges = malloc(sizeof(Edge) * (size_t)n_edges);

        float total = 0.0f;
        uint64_t bits = legal;
        while (bits) { total += pi[__builtin_ctzll(bits)]; bits &= bits - 1; }

        bits = legal;
        int idx = 0;
        if (total > 0.0f) {
            float inv = 1.0f / total;
            while (bits) {
                int bit = __builtin_ctzll(bits);
                edges[idx].action = (uint16_t)bit;
                edges[idx].P = pi[bit] * inv;
                edges[idx].N = 0;
                edges[idx].W = 0.0f;
                edges[idx].in_flight = 0;
                idx++;
                bits &= bits - 1;
            }
        } else {
            float uniform = 1.0f / (float)n_edges;
            while (bits) {
                int bit = __builtin_ctzll(bits);
                edges[idx].action = (uint16_t)bit;
                edges[idx].P = uniform;
                edges[idx].N = 0;
                edges[idx].W = 0.0f;
                edges[idx].in_flight = 0;
                idx++;
                bits &= bits - 1;
            }
        }
    }

    node->edges = edges;
    node->n_edges = (uint16_t)n_edges;
    node->Ns = 0;
    node->expanded = 1;
}

static int select_action(const TreeNode* node, float cpuct) {
    int Ns = node->Ns;
    float sqrt_Ns = sqrtf((float)Ns + 1e-8f);
    float best_u = -INFINITY;
    int   best_idx = -1;
    for (int e = 0; e < node->n_edges; e++) {
        const Edge* edge = &node->edges[e];
        float u;
        if (edge->N > 0) {
            float q = edge->W / (float)edge->N;
            u = q + cpuct * edge->P * sqrt_Ns / (1.0f + (float)edge->N);
        } else {
            u = cpuct * edge->P * sqrt_Ns;
        }
        if (u > best_u) { best_u = u; best_idx = e; }
    }
    return best_idx;
}

static void backup(PathEntry* path, int depth, float v, float vloss) {
    for (int i = depth - 1; i >= 0; i--) {
        Edge* e = &path[i].node->edges[path[i].edge_idx];
        e->W += vloss + v;
        e->in_flight -= 1;
        v = -v;
    }
}

/* ------------------------------------------------------------------------- */
/* Serial search (no threads, no queue) — used when numParallelSims == 1     */
/*   and virtualLoss == 0 to guarantee bit-exact parity with MCTS.py.        */
/* ------------------------------------------------------------------------- */

static void search_serial(MCTS* m, const Board* root) {
    PathEntry path[MAX_DEPTH];
    int depth = 0;
    Board cur = *root;
    float vloss = m->args.virtualLoss;

    while (1) {
        uint64_t key = zobrist_hash(&cur);
        bool inserted;
        TreeNode* node = tree_get_or_insert(&m->tree, key, &inserted);
        if (inserted) {
            node->board = cur;
            int term = othello_terminal(&cur);
            if (term != 0) {
                node->is_terminal = 1;
                node->terminal_v = (int8_t)term;
            }
        }

        if (node->is_terminal) {
            backup(path, depth, -(float)node->terminal_v, vloss);
            m->stats.n_simulations++;
            return;
        }

        if (!node->expanded) {
            float input[OTHELLO_NN];
            float pi[OTHELLO_ACTION_SIZE];
            float v;
            nn_board_to_input(&cur, input);
            nn_predict_batch(m->nn, input, 1, pi, &v);
            m->stats.n_gpu_calls++;
            m->stats.total_batch_size += 1;

            install_priors(node, &cur, pi);
            backup(path, depth, -v, vloss);
            m->stats.n_simulations++;
            return;
        }

        int e_idx = select_action(node, m->args.cpuct);
        if (e_idx < 0) {
            backup(path, depth, 0.0f, vloss);
            m->stats.n_simulations++;
            return;
        }

        Edge* edge = &node->edges[e_idx];
        if (edge->in_flight > 0) m->stats.n_virtual_loss_collisions++;

        edge->N += 1;
        edge->W -= vloss;
        edge->in_flight += 1;
        node->Ns += 1;

        if (depth >= MAX_DEPTH) abort();
        path[depth].node = node;
        path[depth].edge_idx = e_idx;
        depth++;

        othello_apply(&cur, edge->action);
    }
}

/* ------------------------------------------------------------------------- */
/* Parallel search (one worker step under tree_lock; eval done outside it)   */
/* ------------------------------------------------------------------------- */

/*
 * One simulation under the leaf-parallel scheme. Acquires tree_lock for each
 * node visit, drops it around NN eval, re-acquires for prior install + backup.
 *
 * Mirrors BatchedMCTS._search step-for-step.
 */
static void search_parallel(MCTS* m, const Board* root) {
    PathEntry path[MAX_DEPTH];
    int depth = 0;
    Board cur = *root;
    float vloss = m->args.virtualLoss;

    while (1) {
        uint64_t key = zobrist_hash(&cur);
        int next_action = -1;

        pthread_mutex_lock(&m->tree_lock);

        bool inserted;
        TreeNode* node = tree_get_or_insert(&m->tree, key, &inserted);
        if (inserted) {
            node->board = cur;
            int term = othello_terminal(&cur);
            if (term != 0) {
                node->is_terminal = 1;
                node->terminal_v = (int8_t)term;
            }
        }

        if (node->is_terminal) {
            backup(path, depth, -(float)node->terminal_v, vloss);
            pthread_mutex_unlock(&m->tree_lock);
            return;
        }

        if (!node->expanded) {
            /* Drop lock, enqueue eval, wait, then re-acquire to install. */
            pthread_mutex_unlock(&m->tree_lock);

            EvalRequest req;
            req.board = cur;
            req.done = 0;
            pthread_mutex_init(&req.lock, NULL);
            pthread_cond_init(&req.cond, NULL);
            queue_push(&m->queue, &req);

            pthread_mutex_lock(&req.lock);
            while (!req.done) pthread_cond_wait(&req.cond, &req.lock);
            pthread_mutex_unlock(&req.lock);

            pthread_mutex_lock(&m->tree_lock);
            /* Another worker may have raced us and already installed priors;
             * if so, we keep their entry. We still backup our own -v. */
            TreeNode* n2 = tree_lookup(&m->tree, key);
            if (n2 != NULL && !n2->expanded) {
                install_priors(n2, &cur, req.pi);
            }
            float v = req.v;
            backup(path, depth, -v, vloss);
            pthread_mutex_unlock(&m->tree_lock);

            pthread_mutex_destroy(&req.lock);
            pthread_cond_destroy(&req.cond);
            return;
        }

        int e_idx = select_action(node, m->args.cpuct);
        if (e_idx < 0) {
            backup(path, depth, 0.0f, vloss);
            pthread_mutex_unlock(&m->tree_lock);
            return;
        }

        Edge* edge = &node->edges[e_idx];
        if (edge->in_flight > 0) atomic_fetch_add(&m->stat_collisions, 1);

        edge->N += 1;
        edge->W -= vloss;
        edge->in_flight += 1;
        node->Ns += 1;
        next_action = edge->action;

        if (depth >= MAX_DEPTH) abort();
        path[depth].node = node;
        path[depth].edge_idx = e_idx;
        depth++;

        pthread_mutex_unlock(&m->tree_lock);

        othello_apply(&cur, next_action);
    }
}

/* ------------------------------------------------------------------------- */
/* Worker / batcher                                                          */
/* ------------------------------------------------------------------------- */

typedef struct {
    MCTS*        m;
    const Board* root;
    int          n_sims;
} WorkerArgs;

static void* worker_main(void* p) {
    WorkerArgs* w = p;
    for (int i = 0; i < w->n_sims; i++) {
        search_parallel(w->m, w->root);
    }
    return NULL;
}

typedef struct {
    MCTS* m;
    int   batch_max;
    double timeout_s;
} BatcherArgs;

static void* batcher_main(void* p) {
    BatcherArgs* b = p;
    MCTS* m = b->m;
    EvalRequest* batch[2048];
    int batch_max = b->batch_max < (int)(sizeof(batch)/sizeof(batch[0]))
                  ? b->batch_max : (int)(sizeof(batch)/sizeof(batch[0]));
    double timeout_s = b->timeout_s;

    /* Reusable input/output buffers. */
    float* in_buf  = malloc((size_t)batch_max * OTHELLO_NN * sizeof(float));
    float* pis_buf = malloc((size_t)batch_max * OTHELLO_ACTION_SIZE * sizeof(float));
    float* vs_buf  = malloc((size_t)batch_max * sizeof(float));

    while (1) {
        /* Block for first request of next wave. */
        EvalRequest* first = queue_pop_blocking(&m->queue);
        if (first == NULL) break;

        batch[0] = first;
        int n = 1;
        double deadline = now_seconds() + timeout_s;
        while (n < batch_max) {
            double remaining = deadline - now_seconds();
            if (remaining <= 0) break;
            EvalRequest* r = queue_pop_until(&m->queue, now_seconds() + remaining);
            if (r == NULL) break;
            batch[n++] = r;
        }

        /* Build input tensor. */
        for (int i = 0; i < n; i++) {
            nn_board_to_input(&batch[i]->board, &in_buf[i * OTHELLO_NN]);
        }
        nn_predict_batch(m->nn, in_buf, n, pis_buf, vs_buf);
        atomic_fetch_add(&m->stat_gpu_calls, 1);
        atomic_fetch_add(&m->stat_total_batch, n);

        /* Fan results out. */
        for (int i = 0; i < n; i++) {
            EvalRequest* r = batch[i];
            memcpy(r->pi, &pis_buf[i * OTHELLO_ACTION_SIZE], OTHELLO_ACTION_SIZE * sizeof(float));
            r->v = vs_buf[i];
            pthread_mutex_lock(&r->lock);
            r->done = 1;
            pthread_cond_signal(&r->cond);
            pthread_mutex_unlock(&r->lock);
        }
    }

    free(in_buf);
    free(pis_buf);
    free(vs_buf);
    return NULL;
}

/* ------------------------------------------------------------------------- */
/* Public API                                                                */
/* ------------------------------------------------------------------------- */

MCTS* mcts_new(NN* nn, MCTSArgs args) {
    MCTS* m = calloc(1, sizeof(MCTS));
    if (!m) return NULL;
    m->nn = nn;
    m->args = args;
    int cap_pow2 = args.treeCapacityPow2 > 0 ? args.treeCapacityPow2 : 17;
    tree_init(&m->tree, 1ULL << cap_pow2);
    pthread_mutex_init(&m->tree_lock, NULL);
    queue_init(&m->queue);
    atomic_init(&m->stat_gpu_calls, 0);
    atomic_init(&m->stat_total_batch, 0);
    atomic_init(&m->stat_collisions, 0);
    zobrist_init();
    return m;
}

void mcts_free(MCTS* m) {
    if (!m) return;
    tree_free(&m->tree);
    pthread_mutex_destroy(&m->tree_lock);
    queue_destroy(&m->queue);
    free(m);
}

void mcts_reset(MCTS* m) {
    int cap_pow2 = m->args.treeCapacityPow2 > 0 ? m->args.treeCapacityPow2 : 17;
    tree_free(&m->tree);
    tree_init(&m->tree, 1ULL << cap_pow2);
    memset(&m->stats, 0, sizeof(m->stats));
    atomic_store(&m->stat_gpu_calls, 0);
    atomic_store(&m->stat_total_batch, 0);
    atomic_store(&m->stat_collisions, 0);
}

void mcts_get_stats(const MCTS* m, MCTSStats* out) {
    *out = m->stats;
    out->n_gpu_calls += atomic_load(&((MCTS*)m)->stat_gpu_calls);
    out->total_batch_size += atomic_load(&((MCTS*)m)->stat_total_batch);
    out->n_virtual_loss_collisions += atomic_load(&((MCTS*)m)->stat_collisions);
}

void mcts_action_prob(MCTS* m, const Board* root, float temp, float* out_pi) {
    int n_workers = m->args.numParallelSims < 1 ? 1 : m->args.numParallelSims;
    int n_sims = m->args.numMCTSSims;
    double t0 = now_seconds();

    if (n_workers == 1 && m->args.virtualLoss == 0.0f) {
        /* Bit-exact serial path that test_mcts.c covers. */
        for (int i = 0; i < n_sims; i++) search_serial(m, root);
        m->stats.n_simulations += n_sims;
        m->stats.total_seconds += now_seconds() - t0;
    } else {
        /* Spawn one batcher + n_workers worker threads for this call. */
        queue_clear_stop(&m->queue);

        BatcherArgs ba = {
            .m = m,
            .batch_max = m->args.evalBatchSize > 0 ? m->args.evalBatchSize : n_workers,
            .timeout_s = (m->args.evalTimeoutMs > 0 ? m->args.evalTimeoutMs : 1.0f) * 1e-3,
        };
        pthread_t batcher;
        pthread_create(&batcher, NULL, batcher_main, &ba);

        WorkerArgs* wargs = malloc(sizeof(WorkerArgs) * (size_t)n_workers);
        pthread_t*  workers = malloc(sizeof(pthread_t) * (size_t)n_workers);
        int per = n_sims / n_workers;
        int extra = n_sims % n_workers;
        int n_started = 0;
        for (int i = 0; i < n_workers; i++) {
            int sims_i = per + (i < extra ? 1 : 0);
            if (sims_i == 0) continue;
            wargs[n_started].m = m;
            wargs[n_started].root = root;
            wargs[n_started].n_sims = sims_i;
            pthread_create(&workers[n_started], NULL, worker_main, &wargs[n_started]);
            n_started++;
        }
        for (int i = 0; i < n_started; i++) pthread_join(workers[i], NULL);

        /* Tell batcher no more requests will arrive, then wait for it. */
        queue_signal_stop(&m->queue);
        pthread_join(batcher, NULL);

        free(workers);
        free(wargs);

        m->stats.n_simulations += n_sims;
        m->stats.total_seconds += now_seconds() - t0;
    }

    memset(out_pi, 0, sizeof(float) * OTHELLO_ACTION_SIZE);

    uint64_t key = zobrist_hash(root);
    TreeNode* node = tree_lookup(&m->tree, key);
    if (!node || !node->expanded) {
        uint8_t valids[OTHELLO_ACTION_SIZE];
        othello_valid_moves_vec(root, valids);
        float total = 0;
        for (int a = 0; a < OTHELLO_ACTION_SIZE; a++) total += (float)valids[a];
        for (int a = 0; a < OTHELLO_ACTION_SIZE; a++) out_pi[a] = (float)valids[a] / total;
        return;
    }

    if (temp == 0.0f) {
        int   best_a = -1;
        int32_t max_count = -1;
        for (int e = 0; e < node->n_edges; e++) {
            int32_t n = node->edges[e].N;
            if (n > max_count) { max_count = n; best_a = node->edges[e].action; }
        }
        if (best_a >= 0) out_pi[best_a] = 1.0f;
        return;
    }

    float total = 0.0f;
    for (int e = 0; e < node->n_edges; e++) {
        float v = powf((float)node->edges[e].N, 1.0f / temp);
        out_pi[node->edges[e].action] = v;
        total += v;
    }
    if (total > 0.0f) {
        float inv = 1.0f / total;
        for (int a = 0; a < OTHELLO_ACTION_SIZE; a++) out_pi[a] *= inv;
    } else {
        uint8_t valids[OTHELLO_ACTION_SIZE];
        othello_valid_moves_vec(root, valids);
        float vt = 0;
        for (int a = 0; a < OTHELLO_ACTION_SIZE; a++) vt += (float)valids[a];
        for (int a = 0; a < OTHELLO_ACTION_SIZE; a++) out_pi[a] = (float)valids[a] / vt;
    }
}
