"""
Run Python BatchedMCTS in serial mode (workers=1, vloss=0, batch=1) on a
fixed canonical Othello root, then dump the resulting visit counts + pi
vector for the C MCTS parity test.

Format (little-endian):
    int32 N
    int32 action_size
    int32 num_sims
    float cpuct
    int8  [N*N]      canonical_board (row-major (x, y))
    int32 [action_size]   visit counts (Nsa)
    float32 [action_size] pi at temp=1
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
from BatchedMCTS import BatchedMCTS  # noqa: E402


def dotdict(d):
    class _D(dict):
        __getattr__ = dict.__getitem__
    return _D(d)


def fixed_canonical_board(game):
    """Same setup as benchmark_mcts.py: two random opening moves, then the
    canonical form of the resulting position."""
    board = game.getInitBoard()
    valids = game.getValidMoves(board, 1)
    a = int(np.argmax(valids[:-1]))
    board, p = game.getNextState(board, 1, a)
    valids = game.getValidMoves(board, p)
    a = int(np.argmax(valids[:-1]))
    board, p = game.getNextState(board, p, a)
    return game.getCanonicalForm(board, p).astype(np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pth', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--sims', type=int, default=50)
    ap.add_argument('--cpuct', type=float, default=1.0)
    args = ap.parse_args()

    game = OthelloGame(args.n)
    wrapper = NNetWrapper(game)
    folder, fname = os.path.split(args.pth)
    wrapper.load_checkpoint(folder=folder or '.', filename=fname)

    canonical = fixed_canonical_board(game)

    mcts_args = dotdict({
        'numMCTSSims':     args.sims,
        'cpuct':           args.cpuct,
        'numParallelSims': 1,
        'virtualLoss':     0.0,
        'evalBatchSize':   1,
        'evalTimeoutMs':   1.0,
    })
    mcts = BatchedMCTS(game, wrapper, mcts_args)
    np.random.seed(0)
    pi = mcts.getActionProb(canonical, temp=1)
    pi = np.asarray(pi, dtype=np.float32)

    s = game.stringRepresentation(canonical)
    visit_counts = np.array(
        [mcts.Nsa.get((s, a), 0) for a in range(game.getActionSize())],
        dtype=np.int32,
    )

    with open(args.out, 'wb') as f:
        f.write(struct.pack('<iii', args.n, game.getActionSize(), args.sims))
        f.write(struct.pack('<f', float(args.cpuct)))
        f.write(canonical.tobytes())
        f.write(visit_counts.tobytes())
        f.write(pi.tobytes())
    print(f"wrote {args.out}: sims={args.sims} sum(visits)={int(visit_counts.sum())}")


if __name__ == '__main__':
    main()
