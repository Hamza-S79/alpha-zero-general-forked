/*
 * NN parity test: load the .onnx via ONNX Runtime C API, run inference on the
 * same boards Python ran through PyTorch, and assert the outputs match within
 * a small tolerance.
 *
 * Build (Mac, brew onnxruntime):
 *   cc -std=c11 -O2 -Wall -Wextra -I include -I/opt/homebrew/include/onnxruntime \
 *      src/othello.c src/nn.c tests/test_nn.c \
 *      -L/opt/homebrew/lib -lonnxruntime -o build/test_nn
 *
 * Run:
 *   build/test_nn tests/random_model.onnx tests/nn_fixture.bin
 */

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

    NN* nn;
    if (nn_load(&nn, argv[1], /*use_cuda=*/0) != 0) {
        fprintf(stderr, "nn_load failed\n");
        return 2;
    }

    FILE* f = fopen(argv[2], "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[2]);
        return 2;
    }

    int32_t n, asz, num;
    if (read_exact(f, &n, 4) || read_exact(f, &asz, 4) || read_exact(f, &num, 4)) {
        fprintf(stderr, "short read on header\n"); return 2;
    }
    if (n != OTHELLO_N || asz != OTHELLO_ACTION_SIZE) {
        fprintf(stderr, "fixture shape (n=%d, asz=%d) doesn't match build\n", n, asz);
        return 2;
    }

    /* Pack all boards into one float NCHW buffer, then run as a single batch. */
    int8_t  boards_int8[num * OTHELLO_NN];
    float*  in_buf   = malloc((size_t)num * OTHELLO_NN * sizeof(float));
    float*  py_pis   = malloc((size_t)num * OTHELLO_ACTION_SIZE * sizeof(float));
    float*  py_vs    = malloc((size_t)num * sizeof(float));
    float*  c_pis    = malloc((size_t)num * OTHELLO_ACTION_SIZE * sizeof(float));
    float*  c_vs     = malloc((size_t)num * sizeof(float));
    if (!in_buf || !py_pis || !py_vs || !c_pis || !c_vs) {
        fprintf(stderr, "alloc failure\n"); return 2;
    }

    for (int i = 0; i < num; i++) {
        if (read_exact(f, &boards_int8[i * OTHELLO_NN], OTHELLO_NN)
         || read_exact(f, &py_pis[i * OTHELLO_ACTION_SIZE], OTHELLO_ACTION_SIZE * sizeof(float))
         || read_exact(f, &py_vs[i], sizeof(float))) {
            fprintf(stderr, "short read on record %d\n", i); return 2;
        }
        Board b;
        othello_from_int8(&b, &boards_int8[i * OTHELLO_NN]);
        nn_board_to_input(&b, &in_buf[i * OTHELLO_NN]);
    }
    fclose(f);

    nn_predict_batch(nn, in_buf, num, c_pis, c_vs);

    float max_pi = 0.0f, max_v = 0.0f;
    for (int i = 0; i < num; i++) {
        for (int a = 0; a < OTHELLO_ACTION_SIZE; a++) {
            float d = fabsf(py_pis[i * OTHELLO_ACTION_SIZE + a] - c_pis[i * OTHELLO_ACTION_SIZE + a]);
            if (d > max_pi) max_pi = d;
        }
        float dv = fabsf(py_vs[i] - c_vs[i]);
        if (dv > max_v) max_v = dv;
    }

    printf("max |pi_pytorch - pi_c| = %.3e\n", max_pi);
    printf("max |v_pytorch  - v_c|  = %.3e\n", max_v);

    int rc = 0;
    if (max_pi > 1e-4f || max_v > 1e-4f) {
        printf("FAIL\n");
        rc = 1;
    } else {
        printf("OK (%d records)\n", num);
    }

    free(in_buf); free(py_pis); free(py_vs); free(c_pis); free(c_vs);
    nn_free(nn);
    return rc;
}
