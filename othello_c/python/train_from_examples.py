"""
Train the PyTorch OthelloNNet on examples written by the C selfplay loop, then
re-export to .onnx for the next iteration of self-play.

Reads one or more `.bin` files produced by `othello_c selfplay` and expands
each (canonical_board, pi, z) record into 8 symmetries via OthelloGame
(reusing the existing data-augmentation logic in /othello/OthelloGame.py).

Usage:
    python python/train_from_examples.py \
        --pth-in  /path/to/best.pth.tar \
        --pth-out /path/to/iter1.pth.tar \
        --onnx-out /path/to/iter1.onnx \
        --examples ex1.bin [ex2.bin ...]
"""

import argparse
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)

from othello.OthelloGame import OthelloGame  # noqa: E402
from othello.pytorch.NNet import NNetWrapper  # noqa: E402


def read_examples(path, n):
    """
    File format (matches src/coach.c coach_write_header / coach_run_episode):
        int32 N
        int32 action_size
        int32 num_examples
        per record:
            int8[N*N]               canonical_board (row-major (x,y))
            float32[action_size]    pi
            int8                    z
    """
    out = []
    with open(path, 'rb') as f:
        N, asz, num = struct.unpack('<iii', f.read(12))
        if N != n:
            raise ValueError(f"file {path} has N={N}, expected {n}")
        rec_bytes = N * N + asz * 4 + 1
        for _ in range(num):
            blob = f.read(rec_bytes)
            if len(blob) != rec_bytes:
                raise ValueError(f"truncated example in {path}")
            board = np.frombuffer(blob[:N * N], dtype=np.int8).reshape(N, N).copy()
            pi    = np.frombuffer(blob[N * N:N * N + asz * 4], dtype=np.float32).copy()
            z     = np.frombuffer(blob[-1:], dtype=np.int8)[0]
            out.append((board, pi, float(z)))
    return out


def expand_symmetries(game, examples):
    """For each (canonical_board, pi, z), emit all 8 symmetries (matches
    Coach.executeEpisode line in /Coach.py:66)."""
    out = []
    for board, pi, z in examples:
        for sym_b, sym_pi in game.getSymmetries(board, pi):
            out.append((sym_b, np.asarray(sym_pi, dtype=np.float32), z))
    return out


def export_onnx(wrapper, onnx_path, n):
    import torch

    class _Wrap(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x.squeeze(1))

    # Move to CPU for export — torch.onnx tracing uses a CPU sample tensor.
    model = _Wrap(wrapper.nnet.cpu()).eval()
    sample = torch.zeros(1, 1, n, n, dtype=torch.float32)
    torch.onnx.export(
        model, sample, onnx_path,
        input_names=['board'], output_names=['policy', 'value'],
        dynamic_axes={'board': {0: 'batch'}, 'policy': {0: 'batch'}, 'value': {0: 'batch'}},
        opset_version=17, do_constant_folding=True,
    )
    # If torch wrote a sidecar .data, fold it back inline.
    sidecar = onnx_path + '.data'
    if os.path.exists(sidecar):
        import onnx
        m = onnx.load(onnx_path, load_external_data=True)
        for t in m.graph.initializer:
            if t.HasField('data_location') and t.data_location == onnx.TensorProto.EXTERNAL:
                t.ClearField('external_data')
                t.data_location = onnx.TensorProto.DEFAULT
        onnx.save(m, onnx_path)
        os.remove(sidecar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pth-in', required=True, help='starting .pth.tar checkpoint')
    ap.add_argument('--pth-out', required=True)
    ap.add_argument('--onnx-out', required=True)
    ap.add_argument('--examples', nargs='+', required=True)
    ap.add_argument('--n', type=int, default=6)
    args = ap.parse_args()

    game = OthelloGame(args.n)
    wrapper = NNetWrapper(game)
    folder, fname = os.path.split(args.pth_in)
    wrapper.load_checkpoint(folder=folder or '.', filename=fname)

    raw = []
    for p in args.examples:
        raw += read_examples(p, args.n)
    print(f"read {len(raw)} raw examples from {len(args.examples)} file(s)")
    aug = expand_symmetries(game, raw)
    print(f"expanded to {len(aug)} examples after symmetries")

    np.random.shuffle(aug)
    wrapper.train(aug)

    out_folder, out_fname = os.path.split(args.pth_out)
    wrapper.save_checkpoint(folder=out_folder or '.', filename=out_fname)
    print(f"saved {args.pth_out}")

    export_onnx(wrapper, args.onnx_out, args.n)
    print(f"exported {args.onnx_out}")


if __name__ == '__main__':
    main()
