"""
Verify the exported .onnx model produces the same outputs as PyTorch's
NNetWrapper.predict_batch on the same canonical boards (within float32 noise).

This is the cheap side of M3: Python-only check that the export step preserves
the model. The C-side test (test_nn.c) does the same comparison once ORT C
library is installed.

Usage:
    python python/verify_onnx_parity.py --pth /path/to/best.pth.tar
                                       --onnx /path/to/best.onnx
    python python/verify_onnx_parity.py --no-pth --onnx tests/random_model.onnx
"""

import argparse
import os
import sys

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)

from othello.OthelloGame import OthelloGame  # noqa: E402
from othello.pytorch.NNet import NNetWrapper  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pth', help='PyTorch checkpoint (matches the .onnx)')
    ap.add_argument('--no-pth', action='store_true',
                    help='compare random-init network (must match how onnx was exported)')
    ap.add_argument('--onnx', required=True, help='exported .onnx model')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tol', type=float, default=1e-4)
    args = ap.parse_args()

    if not args.no_pth and not args.pth:
        ap.error('either --pth or --no-pth required')

    rng = np.random.default_rng(args.seed)
    game = OthelloGame(args.n)
    wrapper = NNetWrapper(game)
    if args.pth:
        folder, fname = os.path.split(args.pth)
        wrapper.load_checkpoint(folder=folder or '.', filename=fname)

    # Build a batch of random canonical-form boards by playing random moves.
    boards = []
    for _ in range(args.batch):
        b = game.getInitBoard()
        # Random number of random moves into the game.
        n_moves = int(rng.integers(0, 12))
        player = 1
        for _ in range(n_moves):
            valids = game.getValidMoves(b, player)
            legal = np.flatnonzero(valids)
            a = int(rng.choice(legal))
            b, player = game.getNextState(b, player, a)
            term = game.getGameEnded(b, player)
            if term != 0:
                break
        # canonical form for whoever's to move:
        boards.append(game.getCanonicalForm(b, player).astype(np.float32))
    batch = np.stack(boards, axis=0)  # (B, N, N)

    # PyTorch reference (same path BatchedMCTS uses).
    py_pis, py_vs = wrapper.predict_batch([b for b in batch])

    # ONNX Runtime (Python).
    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    ort_input = batch[:, None, :, :].astype(np.float32)  # (B, 1, N, N)
    ort_outs = sess.run(out_names, {in_name: ort_input})
    log_pi, v = ort_outs[0], ort_outs[1].reshape(-1)
    onnx_pis = np.exp(log_pi)
    onnx_vs = v

    diff_pi = float(np.max(np.abs(py_pis - onnx_pis)))
    diff_v  = float(np.max(np.abs(py_vs  - onnx_vs)))
    print(f"max |pi_pytorch - pi_onnx| = {diff_pi:.3e}")
    print(f"max |v_pytorch  - v_onnx|  = {diff_v:.3e}")

    if diff_pi > args.tol or diff_v > args.tol:
        print(f"FAIL: tol = {args.tol}")
        sys.exit(1)
    print("OK")


if __name__ == '__main__':
    main()
