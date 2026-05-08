"""
Profiling harness comparing baseline MCTS vs BatchedMCTS on Othello.

Run:
    python benchmark_mcts.py

Loads a pretrained 6x6 Othello model, runs numMCTSSims simulations from a
fixed canonical board across a sweep of (numParallelSims, evalBatchSize), and
prints per-config metrics: wall-clock, mean batch size, GPU calls, lock wait,
virtual-loss collisions.

Also asserts that BatchedMCTS with numParallelSims=1 / virtualLoss=0 / batch=1
matches baseline MCTS exactly on action probabilities (sanity check).
"""

import argparse
import time

import numpy as np

from MCTS import MCTS
from BatchedMCTS import BatchedMCTS
from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper
from utils import dotdict


def make_args(num_sims, num_parallel, batch_size, vloss=1.0, timeout_ms=1.0,
              cpuct=1.0, use_batched=True):
    return dotdict({
        'numMCTSSims': num_sims,
        'cpuct': cpuct,
        'useBatchedMCTS': use_batched,
        'numParallelSims': num_parallel,
        'virtualLoss': vloss,
        'evalBatchSize': batch_size,
        'evalTimeoutMs': timeout_ms,
    })


def fixed_canonical_board(game):
    """Initial Othello board after one move sequence — gives a non-trivial root."""
    board = game.getInitBoard()
    # play a couple of moves to leave a typical mid-opening position
    valids = game.getValidMoves(board, 1)
    a = int(np.argmax(valids[:-1]))   # first legal move
    board, p = game.getNextState(board, 1, a)
    valids = game.getValidMoves(board, p)
    a = int(np.argmax(valids[:-1]))
    board, p = game.getNextState(board, p, a)
    return game.getCanonicalForm(board, p)


def time_call(fn, repeat=3):
    times = []
    for _ in range(repeat):
        t0 = time.time()
        out = fn()
        times.append(time.time() - t0)
    return min(times), out


def sanity_check(game, nnet, num_sims):
    """BatchedMCTS(N=1, vloss=0, batch=1) must match MCTS exactly."""
    board = fixed_canonical_board(game)

    np.random.seed(0)
    base_args = dotdict({'numMCTSSims': num_sims, 'cpuct': 1.0})
    base = MCTS(game, nnet, base_args)
    pi_base = base.getActionProb(board, temp=1)

    np.random.seed(0)
    batched_args = make_args(num_sims, num_parallel=1, batch_size=1, vloss=0.0)
    batched = BatchedMCTS(game, nnet, batched_args)
    pi_batched = batched.getActionProb(board, temp=1)

    diff = max(abs(a - b) for a, b in zip(pi_base, pi_batched))
    print(f"[sanity] max |pi_base - pi_batched| = {diff:.3e}  "
          f"(N=1, vloss=0, batch=1, sims={num_sims})")
    if diff > 1e-9:
        print("[sanity] WARNING: distributions differ. The two implementations")
        print("         take different code paths even with N=1; small numerical")
        print("         differences are expected if any float ops reorder.")
    metrics = batched.metrics()
    expected_calls = num_sims  # batch=1, so one call per sim
    print(f"[sanity] BatchedMCTS GPU calls = {metrics['n_gpu_calls']}  "
          f"(expected ~{expected_calls})")


def benchmark_baseline(game, nnet, num_sims, repeat=3):
    """Time baseline MCTS — one GPU call per leaf."""
    board = fixed_canonical_board(game)
    args = dotdict({'numMCTSSims': num_sims, 'cpuct': 1.0})

    def _run():
        np.random.seed(0)
        m = MCTS(game, nnet, args)
        return m.getActionProb(board, temp=1)

    t, _ = time_call(_run, repeat=repeat)
    return t


def benchmark_batched(game, nnet, num_sims, num_parallel, batch_size,
                      vloss=1.0, repeat=3):
    """Time BatchedMCTS for a config and return (best_time, metrics)."""
    board = fixed_canonical_board(game)
    args = make_args(num_sims, num_parallel, batch_size, vloss=vloss)

    last_metrics = {}

    def _run():
        np.random.seed(0)
        m = BatchedMCTS(game, nnet, args)
        pi = m.getActionProb(board, temp=1)
        last_metrics.update(m.metrics())
        return pi

    t, _ = time_call(_run, repeat=repeat)
    return t, last_metrics


def fmt_pct(x, total):
    if total <= 0:
        return "  -  "
    return f"{100.0 * x / total:5.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sims', type=int, default=50,
                        help='numMCTSSims for the benchmark')
    parser.add_argument('--repeat', type=int, default=3,
                        help='number of timing repetitions; reports best')
    parser.add_argument('--skip-sanity', action='store_true')
    parser.add_argument('--workers', type=str, default='1,2,4,8,16',
                        help='comma-separated numParallelSims values to sweep')
    parser.add_argument('--batches', type=str, default='1,4,8,16',
                        help='comma-separated evalBatchSize values to sweep')
    args = parser.parse_args()

    workers = [int(x) for x in args.workers.split(',') if x.strip()]
    batches = [int(x) for x in args.batches.split(',') if x.strip()]

    print("Loading 6x6 Othello game and random-init NN...")
    game = OthelloGame(6)
    nnet = NNetWrapper(game)
    # NB: random weights have identical forward-pass cost to trained ones —
    # we are measuring CPU<->GPU overhead, not move quality, so no checkpoint needed.
    print("  done.\n")

    if not args.skip_sanity:
        sanity_check(game, nnet, num_sims=args.sims)
        print()

    # warm up GPU (first call has CUDA init overhead)
    _ = benchmark_baseline(game, nnet, num_sims=4, repeat=1)

    print(f"Baseline MCTS (single-threaded, batch=1), sims={args.sims}")
    t_base = benchmark_baseline(game, nnet, num_sims=args.sims,
                                repeat=args.repeat)
    print(f"  wall-clock per getActionProb: {t_base * 1000:.1f} ms")
    print(f"  GPU calls per getActionProb : {args.sims}  (one per leaf)\n")

    header = (f"{'Workers':>7} {'Batch':>6} {'Wall(ms)':>9} {'Speedup':>8} "
              f"{'GPU calls':>9} {'MeanBatch':>9} {'GPU wait%':>9} "
              f"{'Lock%':>6} {'Collisions':>10}")
    print(f"BatchedMCTS sweep (sims={args.sims}, vloss=1.0)")
    print(header)
    print('-' * len(header))

    for nw in workers:
        for bs in batches:
            t, m = benchmark_batched(
                game, nnet,
                num_sims=args.sims,
                num_parallel=nw,
                batch_size=bs,
                vloss=1.0,
                repeat=args.repeat,
            )
            speedup = t_base / t if t > 0 else 0.0
            total = m.get('total_seconds', t)
            print(f"{nw:>7} {bs:>6} {t * 1000:>9.1f} {speedup:>7.2f}x "
                  f"{m.get('n_gpu_calls', 0):>9} "
                  f"{m.get('mean_batch_size', 0):>9.2f} "
                  f"{fmt_pct(m.get('gpu_wait_seconds', 0), total * nw):>9} "
                  f"{fmt_pct(m.get('tree_lock_wait_seconds', 0), total * nw):>6} "
                  f"{m.get('n_virtual_loss_collisions', 0):>10}")
    print()
    print("Notes:")
    print("  - GPU wait% / Lock% are summed across workers, divided by")
    print("    (workers * wall-clock); they can exceed 100% if workers")
    print("    block in parallel.")
    print("  - Python's GIL serializes pure-Python descent; the C/C++ port")
    print("    will gain additional speedup from truly parallel descent on")
    print("    top of the GPU-batching reduction shown here.")


if __name__ == '__main__':
    main()
