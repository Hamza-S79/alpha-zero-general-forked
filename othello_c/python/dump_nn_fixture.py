"""
Dump a binary fixture pairing canonical-form Othello boards with the PyTorch
NNetWrapper.predict_batch outputs, for the C-side test_nn.c parity check.

Format (little-endian):
  int32 N
  int32 action_size
  int32 num_records
  for each record:
    int8 [N*N]               canonical_board (row-major (x, y), {-1, 0, +1})
    float32 [action_size]    pi  (probabilities, NOT log)
    float32                  v
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


def random_canonical(rng, game):
    b = game.getInitBoard()
    n_moves = int(rng.integers(0, 12))
    player = 1
    for _ in range(n_moves):
        valids = game.getValidMoves(b, player)
        legal = np.flatnonzero(valids)
        a = int(rng.choice(legal))
        b, player = game.getNextState(b, player, a)
        if game.getGameEnded(b, player) != 0:
            break
    return game.getCanonicalForm(b, player).astype(np.int8), player


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pth', required=True, help='checkpoint matching the .onnx')
    ap.add_argument('--out', required=True, help='fixture output path')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--num', type=int, default=64)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    game = OthelloGame(args.n)
    wrapper = NNetWrapper(game)
    folder, fname = os.path.split(args.pth)
    wrapper.load_checkpoint(folder=folder or '.', filename=fname)

    boards = [random_canonical(rng, game)[0] for _ in range(args.num)]
    batch = np.stack([b.astype(np.float32) for b in boards], axis=0)  # (B, N, N)
    pis, vs = wrapper.predict_batch([b for b in batch])

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'wb') as f:
        f.write(struct.pack('<iii', args.n, args.n * args.n + 1, args.num))
        for i in range(args.num):
            f.write(boards[i].tobytes())          # int8 N*N row-major (x,y)
            f.write(pis[i].astype(np.float32).tobytes())   # float32 ACTION_SIZE
            f.write(np.float32(vs[i]).tobytes())  # float32 scalar
    print(f"wrote {args.num} records to {args.out}")


if __name__ == '__main__':
    main()
