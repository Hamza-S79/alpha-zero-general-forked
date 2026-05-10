"""
Generate a binary fixture comparing Python OthelloGame behavior step-by-step.

Each record captures one (canonical_board, action, next_canonical_board, valids, terminal)
tuple; the C parity test loads this file and asserts byte-for-byte equality of the
bitboard rules.

Format (little-endian):
    header:
      int32  N
      int32  num_records
    repeated num_records times:
      int8 [N*N]   canonical_board   (row-major in (x, y), values in {-1, 0, +1})
      uint8[N*N+1] valids            (1 if action is legal in this canonical state)
      int32        terminal          (return of getGameEnded(canonical, 1))
      int32        chosen_action     (a legal action we then apply; PASS == N*N)
      int8 [N*N]   next_canonical    (canonical_board after apply + side flip)

Run:
    cd othello_c
    python python/dump_othello_fixture.py --out tests/fixture_othello.bin
"""

import argparse
import os
import struct
import sys

import numpy as np

# Allow imports from the parent (root) directory.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)

from othello.OthelloGame import OthelloGame  # noqa: E402


def random_legal_action(rng, valids):
    """Pick a uniformly random legal action."""
    legal = np.flatnonzero(valids)
    return int(rng.choice(legal))


def play_one_episode(rng, game, records, max_records):
    """Play a random self-play game; emit one record per move and stop early
    if we reach the global record cap."""
    board = game.getInitBoard()
    player = 1
    while True:
        canonical = game.getCanonicalForm(board, player)
        valids = game.getValidMoves(canonical, 1)
        terminal = game.getGameEnded(canonical, 1)

        # If terminal: still emit a record (board + valids + terminal),
        # with a null action so the C side can verify terminal handling.
        if terminal != 0:
            records.append({
                'canonical': canonical.astype(np.int8),
                'valids': valids.astype(np.uint8),
                'terminal': int(terminal),
                'action': -1,
                'next_canonical': canonical.astype(np.int8),  # unchanged
            })
            return

        action = random_legal_action(rng, valids)
        next_board, next_player = game.getNextState(canonical, 1, action)
        # Canonical form for the next side-to-move:
        #   next_player == -1  →  next_canonical = -next_board (side flip)
        next_canonical = game.getCanonicalForm(next_board, next_player)

        records.append({
            'canonical': canonical.astype(np.int8),
            'valids': valids.astype(np.uint8),
            'terminal': int(terminal),
            'action': int(action),
            'next_canonical': next_canonical.astype(np.int8),
        })
        if len(records) >= max_records:
            return

        # Step the absolute-frame board with the absolute action.
        # Note: action indexes match between absolute and canonical because
        # canonical = ±board, which doesn't permute squares.
        board, player = game.getNextState(board, player, action)


def write_fixture(path, n, records):
    with open(path, 'wb') as f:
        f.write(struct.pack('<ii', n, len(records)))
        for r in records:
            assert r['canonical'].shape == (n, n)
            assert r['valids'].shape == (n * n + 1,)
            assert r['next_canonical'].shape == (n, n)
            # Python OthelloGame stores board as numpy[x][y] (col, row).
            # tobytes() walks the array C-order, which is exactly x*N + y.
            f.write(r['canonical'].tobytes())
            f.write(r['valids'].tobytes())
            f.write(struct.pack('<ii', r['terminal'], r['action']))
            f.write(r['next_canonical'].tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='output fixture path')
    ap.add_argument('--n', type=int, default=6, help='board size (Python uses 6)')
    ap.add_argument('--num-records', type=int, default=10_000)
    ap.add_argument('--seed', type=int, default=0xC0FFEE)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    game = OthelloGame(args.n)
    records = []
    while len(records) < args.num_records:
        play_one_episode(rng, game, records, args.num_records)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    write_fixture(args.out, args.n, records[:args.num_records])
    print(f"Wrote {len(records[:args.num_records])} records to {args.out}")


if __name__ == '__main__':
    main()
