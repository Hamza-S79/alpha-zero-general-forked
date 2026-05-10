#ifndef OTHELLO_C_NN_H
#define OTHELLO_C_NN_H

#include <stddef.h>
#include <stdint.h>

#include "othello.h"

/*
 * ONNX Runtime wrapper. Loads the .onnx model exported by
 * python/export_onnx.py and runs inference on batches of canonical Othello
 * boards.
 *
 * Inputs:
 *   board_in  shape (B, 1, N, N) float32, NCHW. Values in {-1, 0, +1} where
 *             +1 is side-to-move (matches PyTorch canonical board).
 *
 * Outputs:
 *   pi_out    shape (B, ACTION_SIZE) float32, post log_softmax exp() to get
 *             plain probabilities (matches OthelloNNet.predict_batch).
 *   v_out     shape (B,) float32, in [-1, +1].
 *
 * The NN is constructed once with nn_load and then called many times with
 * nn_predict_batch. The session is thread-safe for concurrent Run calls but
 * we use a single batcher thread anyway, matching BatchedMCTS.py.
 */

typedef struct NN NN;

/* Returns 0 on success, non-zero on error (logs to stderr). */
int  nn_load(NN** out, const char* onnx_path, int use_cuda);
void nn_free(NN* nn);

/*
 * Run inference on a batch of `B` boards. Caller-owned buffers:
 *   boards_NCHW: (B * N * N) float32, side-to-move stones at +1, opp at -1, empty at 0.
 *   pis:         (B * ACTION_SIZE) float32 output (probabilities, post-exp).
 *   vs:          (B) float32 output.
 */
void nn_predict_batch(NN* nn, const float* boards_NCHW, int B,
                      float* pis, float* vs);

/* Convert a Board (canonical form) to the (N*N) float32 input plane. */
void nn_board_to_input(const Board* b, float* out_plane);

#endif /* OTHELLO_C_NN_H */
