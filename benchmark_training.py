"""
Training-workload benchmark: single-threaded MCTS vs multi-threaded BatchedMCTS.

Runs the inner loop of Coach.learn() — namely Coach.executeEpisode() repeated N
times — under both backends and reports the wall-clock speedup. This is the
realistic number for "how much faster will a training iteration get?"

Run:
    python3 benchmark_training.py --episodes 10 --sims 25
    python3 benchmark_training.py --episodes 10 --sims 50 --workers 16 --batch 16
"""

import argparse
import time

import numpy as np

from Coach import Coach
from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper
from utils import dotdict


def build_args(use_batched, sims, workers, batch, vloss, timeout_ms,
               n_episodes, board_size):
    """Args dotdict suitable for Coach.executeEpisode (training inner loop)."""
    return dotdict({
        # MCTS
        'numMCTSSims': sims,
        'cpuct': 1.0,
        'tempThreshold': 15,
        # Coach (we don't actually call learn(); these stay as no-ops)
        'numIters': 1,
        'numEps': n_episodes,
        'updateThreshold': 0.6,
        'maxlenOfQueue': 200000,
        'arenaCompare': 0,
        'checkpoint': './temp_bench/',
        'load_model': False,
        'load_folder_file': ('', ''),
        'numItersForTrainExamplesHistory': 5,
        # BatchedMCTS
        'useBatchedMCTS': use_batched,
        'numParallelSims': workers,
        'virtualLoss': vloss,
        'evalBatchSize': batch,
        'evalTimeoutMs': timeout_ms,
    })


def time_episodes(use_batched, sims, workers, batch, vloss, timeout_ms,
                  n_episodes, board_size, seed=0):
    """
    Returns dict: {wall_seconds, episodes, total_examples, sec_per_episode,
                   sec_per_example}.
    """
    np.random.seed(seed)
    game = OthelloGame(board_size)
    nnet = NNetWrapper(game)   # random init — forward-pass cost is identical
    args = build_args(use_batched, sims, workers, batch, vloss, timeout_ms,
                      n_episodes, board_size)
    coach = Coach(game, nnet, args)

    t0 = time.time()
    total_examples = 0
    for _ in range(n_episodes):
        ex = coach.executeEpisode()
        total_examples += len(ex)
    elapsed = time.time() - t0

    return {
        'wall_seconds': elapsed,
        'episodes': n_episodes,
        'total_examples': total_examples,
        'sec_per_episode': elapsed / n_episodes,
        'sec_per_example': elapsed / total_examples if total_examples else float('nan'),
    }


def fmt_row(label, r):
    return (f"{label:<22}"
            f"  {r['wall_seconds']:>8.2f}s"
            f"  {r['sec_per_episode']:>8.2f}s/ep"
            f"  {r['sec_per_example']:>10.4f}s/ex"
            f"  {r['total_examples']:>6} examples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=10,
                    help='self-play episodes per backend')
    ap.add_argument('--sims', type=int, default=25,
                    help='numMCTSSims per move')
    ap.add_argument('--workers', type=int, default=8,
                    help='BatchedMCTS numParallelSims')
    ap.add_argument('--batch', type=int, default=8,
                    help='BatchedMCTS evalBatchSize')
    ap.add_argument('--vloss', type=float, default=1.0)
    ap.add_argument('--timeout-ms', type=float, default=1.0)
    ap.add_argument('--board', type=int, default=6, help='Othello board size')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    print("=" * 78)
    print("Training-workload benchmark: single-threaded vs multi-threaded MCTS")
    print("=" * 78)
    print(f"Board: {args.board}x{args.board}   "
          f"sims/move: {args.sims}   episodes: {args.episodes}")
    print(f"BatchedMCTS:  workers={args.workers}  batch={args.batch}  "
          f"vloss={args.vloss}  timeout={args.timeout_ms}ms")
    print()

    # warm up the NN once (CUDA init / first-forward overhead would skew the
    # first backend benchmarked otherwise)
    print("Warm-up...")
    _ = time_episodes(use_batched=False, sims=2, workers=1, batch=1,
                      vloss=0.0, timeout_ms=args.timeout_ms,
                      n_episodes=1, board_size=args.board, seed=args.seed)
    print("  done.\n")

    print("Running baseline (single-threaded MCTS)...")
    base = time_episodes(
        use_batched=False, sims=args.sims, workers=1, batch=1,
        vloss=0.0, timeout_ms=args.timeout_ms,
        n_episodes=args.episodes, board_size=args.board, seed=args.seed,
    )
    print("  done.")

    print("Running BatchedMCTS (multi-threaded)...")
    batched = time_episodes(
        use_batched=True, sims=args.sims, workers=args.workers,
        batch=args.batch, vloss=args.vloss, timeout_ms=args.timeout_ms,
        n_episodes=args.episodes, board_size=args.board, seed=args.seed,
    )
    print("  done.\n")

    print("-" * 78)
    print(fmt_row("Baseline MCTS",   base))
    print(fmt_row("BatchedMCTS",     batched))
    print("-" * 78)

    speedup_wall = base['wall_seconds'] / batched['wall_seconds']
    speedup_ex = base['sec_per_example'] / batched['sec_per_example']
    print(f"\nWall-clock speedup        : {speedup_wall:.2f}x")
    print(f"Per-example speedup       : {speedup_ex:.2f}x")
    print(f"Time saved per episode    : "
          f"{base['sec_per_episode'] - batched['sec_per_episode']:.2f}s "
          f"({100 * (1 - 1/speedup_wall):.1f}% reduction)")

    # rough projection for a full training iteration
    full_iter_eps = 100  # main.py's default numEps
    proj_base = base['sec_per_episode'] * full_iter_eps
    proj_batched = batched['sec_per_episode'] * full_iter_eps
    print(f"\nProjection for numEps={full_iter_eps} (main.py default):")
    print(f"  Baseline   : {proj_base:>8.1f}s  ({proj_base / 60:.1f} min)")
    print(f"  Batched    : {proj_batched:>8.1f}s  ({proj_batched / 60:.1f} min)")
    print(f"  Time saved : {proj_base - proj_batched:>8.1f}s  "
          f"per training iteration")


if __name__ == '__main__':
    main()
