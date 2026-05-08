import logging
import math
import threading
import time
from concurrent.futures import Future
from queue import Queue, Empty

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class _Stats:
    """Profiling counters. All increments are guarded by metrics_lock."""

    def __init__(self):
        self.n_gpu_calls = 0
        self.batch_size_hist = {}        # batch_size -> count
        self.gpu_wait_seconds = 0.0      # total worker time blocked on a future
        self.tree_lock_wait_seconds = 0.0
        self.n_virtual_loss_collisions = 0
        self.n_simulations = 0
        self.total_seconds = 0.0

    def record_batch(self, size):
        self.n_gpu_calls += 1
        self.batch_size_hist[size] = self.batch_size_hist.get(size, 0) + 1

    def as_dict(self):
        total_batched_leaves = sum(s * c for s, c in self.batch_size_hist.items())
        mean_batch = (total_batched_leaves / self.n_gpu_calls) if self.n_gpu_calls else 0.0
        return {
            'n_simulations': self.n_simulations,
            'n_gpu_calls': self.n_gpu_calls,
            'mean_batch_size': mean_batch,
            'batch_size_hist': dict(self.batch_size_hist),
            'gpu_wait_seconds': self.gpu_wait_seconds,
            'tree_lock_wait_seconds': self.tree_lock_wait_seconds,
            'n_virtual_loss_collisions': self.n_virtual_loss_collisions,
            'total_seconds': self.total_seconds,
        }


class BatchedMCTS:
    """
    Leaf-parallel MCTS with virtual loss and a batched GPU evaluator.

    External interface mirrors MCTS.MCTS: getActionProb(canonicalBoard, temp).
    Internal differences vs the baseline:
      - Stores Wsa (sum of values) instead of Qsa (running mean); Q = Wsa/Nsa.
      - On select: Nsa[(s,a)] += 1; Wsa[(s,a)] -= virtualLoss.
      - On backup: Wsa[(s,a)] += virtualLoss + v.  (Nsa not touched on backup.)
      - Leaves submit their board to a single batcher thread that drains up to
        evalBatchSize requests (or evalTimeoutMs grace) and runs ONE
        nnet.predict_batch() call per wave.

    Args expected on `args`:
      numMCTSSims      total simulations per getActionProb call
      cpuct            PUCT exploration constant
      numParallelSims  number of worker threads
      virtualLoss      magnitude of virtual loss (default 1.0)
      evalBatchSize    max leaves coalesced into one GPU call
      evalTimeoutMs    grace window (ms) the batcher waits to fill a partial batch
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args

        self.Wsa = {}    # sum of values for (s, a)
        self.Nsa = {}    # visit count for (s, a)
        self.Insa = {}   # in-flight count for (s, a) — incremented at select, decremented at backup
        self.Ns = {}     # visit count for s
        self.Ps = {}     # initial policy from NN
        self.Es = {}     # cached game.getGameEnded
        self.Vs = {}     # cached game.getValidMoves

        self._tree_lock = threading.RLock()
        self._metrics_lock = threading.Lock()
        self.stats = _Stats()

    # ---------- public API ----------

    def getActionProb(self, canonicalBoard, temp=1):
        """
        Run numMCTSSims simulations using N worker threads + 1 batcher thread,
        then return the action probability vector.
        """
        self.stats = _Stats()
        n_sims = self.args.numMCTSSims
        n_workers = max(1, getattr(self.args, 'numParallelSims', 1))
        # one-worker, no-virtual-loss config matches the serial baseline
        if n_workers == 1:
            virtual_loss = 0.0
        else:
            virtual_loss = float(getattr(self.args, 'virtualLoss', 1.0))
        eval_batch_size = max(1, getattr(self.args, 'evalBatchSize', n_workers))
        eval_timeout_s = max(0.0, float(getattr(self.args, 'evalTimeoutMs', 1.0)) / 1000.0)

        request_q = Queue()
        stop_event = threading.Event()
        batcher = threading.Thread(
            target=self._batcher_loop,
            args=(request_q, stop_event, eval_batch_size, eval_timeout_s),
            daemon=True,
        )
        batcher.start()

        # split sims across workers; first `extra` workers run one extra sim
        per_worker = n_sims // n_workers
        extra = n_sims % n_workers

        t0 = time.time()
        threads = []
        for i in range(n_workers):
            sims_i = per_worker + (1 if i < extra else 0)
            if sims_i == 0:
                continue
            t = threading.Thread(
                target=self._worker_loop,
                args=(canonicalBoard, sims_i, virtual_loss, request_q),
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        # signal batcher and wait for it to drain
        stop_event.set()
        # ensure the batcher wakes up if its queue.get is blocking
        request_q.put(None)
        batcher.join()

        with self._metrics_lock:
            self.stats.total_seconds = time.time() - t0
            self.stats.n_simulations = n_sims

        # build action probabilities from final visit counts (same as MCTS.py)
        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
                  for a in range(self.game.getActionSize())]

        if temp == 0:
            best_as = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_a = np.random.choice(best_as)
            probs = [0] * len(counts)
            probs[best_a] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        total = float(sum(counts))
        if total <= 0:
            # all zeros — fall back to uniform over valid moves of root
            valids = self.game.getValidMoves(canonicalBoard, 1)
            probs = (valids / np.sum(valids)).tolist()
            return probs
        return [x / total for x in counts]

    def metrics(self):
        with self._metrics_lock:
            return self.stats.as_dict()

    # ---------- workers ----------

    def _worker_loop(self, root_board, n_sims, virtual_loss, request_q):
        for _ in range(n_sims):
            self._search(root_board, virtual_loss, request_q)

    def _search(self, canonicalBoard, virtual_loss, request_q):
        """
        Iterative descent + recursive backup via an explicit path stack.
        Path entries are (s, a) tuples representing edges traversed.
        Returns the value seen from the root's perspective (unused by caller).
        """
        path = []                     # list of (s, a) for backup
        board = canonicalBoard
        while True:
            s = self.game.stringRepresentation(board)

            # terminal check — Es is read-only after first install, so the
            # double-write race is harmless (same value computed twice).
            if s not in self.Es:
                with self._tree_lock_timed():
                    if s not in self.Es:
                        self.Es[s] = self.game.getGameEnded(board, 1)
            if self.Es[s] != 0:
                v = -self.Es[s]
                self._backup(path, v, virtual_loss)
                return v

            # leaf?  install priors via the batcher, then return -v.
            need_eval = False
            with self._tree_lock_timed():
                if s not in self.Ps:
                    need_eval = True

            if need_eval:
                v = self._eval_leaf_and_install(board, s, request_q)
                self._backup(path, v, virtual_loss)
                return v

            # internal node: pick action with max PUCT, apply virtual loss
            with self._tree_lock_timed():
                a = self._select_action(s)
                if a == -1:
                    # no legal action found — defensive; treat as terminal-zero
                    self._backup(path, 0.0, virtual_loss)
                    return 0.0
                # virtual-loss collision signal: another worker is in-flight on this edge
                if self.Insa.get((s, a), 0) > 0:
                    with self._metrics_lock:
                        self.stats.n_virtual_loss_collisions += 1
                # virtual loss bookkeeping
                if (s, a) in self.Nsa:
                    self.Nsa[(s, a)] += 1
                    self.Wsa[(s, a)] -= virtual_loss
                else:
                    self.Nsa[(s, a)] = 1
                    self.Wsa[(s, a)] = -virtual_loss
                self.Insa[(s, a)] = self.Insa.get((s, a), 0) + 1
                self.Ns[s] = self.Ns.get(s, 0) + 1
                path.append((s, a))

            next_s, next_player = self.game.getNextState(board, 1, a)
            board = self.game.getCanonicalForm(next_s, next_player)
            # value flips between plies — handled implicitly because each
            # node's value comes from the leaf eval and is negated once per
            # backup step (see _backup).

    def _select_action(self, s):
        """Pick action with max PUCT. Caller holds tree_lock."""
        valids = self.Vs[s]
        Ps_s = self.Ps[s]
        Ns_s = self.Ns.get(s, 0)
        sqrt_Ns = math.sqrt(Ns_s + EPS)
        cpuct = self.args.cpuct
        cur_best = -float('inf')
        best_act = -1
        for a in range(self.game.getActionSize()):
            if not valids[a]:
                continue
            if (s, a) in self.Nsa:
                n_sa = self.Nsa[(s, a)]
                q = self.Wsa[(s, a)] / n_sa
                u = q + cpuct * Ps_s[a] * sqrt_Ns / (1 + n_sa)
            else:
                u = cpuct * Ps_s[a] * sqrt_Ns
            if u > cur_best:
                cur_best = u
                best_act = a
        return best_act

    def _eval_leaf_and_install(self, board, s, request_q):
        """
        Submit the leaf to the batcher, wait, install priors and Vs/Ns.
        Returns -v (value from parent's perspective, matching MCTS.py:102).
        """
        future = Future()
        request_q.put((board, future))
        wait_start = time.time()
        P, v = future.result()
        wait_elapsed = time.time() - wait_start
        with self._metrics_lock:
            self.stats.gpu_wait_seconds += wait_elapsed

        valids = self.game.getValidMoves(board, 1)
        P = P * valids
        s_sum = float(np.sum(P))
        if s_sum > 0:
            P = P / s_sum
        else:
            log.error("All valid moves were masked, doing a workaround.")
            P = (P + valids)
            P = P / float(np.sum(P))

        with self._tree_lock_timed():
            # another worker may have raced us and installed first; in that
            # case keep the existing entry (its Ns may already be > 0).
            if s not in self.Ps:
                self.Ps[s] = P
                self.Vs[s] = valids
                self.Ns[s] = 0

        return -float(v)

    def _backup(self, path, v, virtual_loss):
        """
        Walk path from leaf -> root, undoing virtual loss and adding real value.
        Value flips at every step (alternating players in canonical form).
        """
        with self._tree_lock_timed():
            for (s, a) in reversed(path):
                # undo the virtual loss and add the actual value
                self.Wsa[(s, a)] += virtual_loss + v
                # Nsa was already incremented at select time; do not touch.
                self.Insa[(s, a)] = self.Insa.get((s, a), 1) - 1
                v = -v

    # ---------- batcher ----------

    def _batcher_loop(self, request_q, stop_event, batch_size, timeout_s):
        """
        Drain requests, call nnet.predict_batch in waves, fan results out.
        Exits when stop_event is set AND queue is empty.
        """
        while True:
            # block for the first request of the next wave
            try:
                first = request_q.get(timeout=0.05)
            except Empty:
                if stop_event.is_set():
                    return
                continue
            if first is None:
                if stop_event.is_set() and request_q.empty():
                    return
                continue

            batch = [first]
            deadline = time.time() + timeout_s
            while len(batch) < batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    item = request_q.get(timeout=remaining)
                except Empty:
                    break
                if item is None:
                    # sentinel: keep draining whatever is already queued, then exit
                    continue
                batch.append(item)

            boards = [b for (b, _f) in batch]
            futures = [f for (_b, f) in batch]
            try:
                pis, vs = self.nnet.predict_batch(boards)
            except Exception as e:
                for f in futures:
                    f.set_exception(e)
                continue

            with self._metrics_lock:
                self.stats.record_batch(len(batch))

            for i, f in enumerate(futures):
                f.set_result((pis[i], float(vs[i])))

    # ---------- helpers ----------

    def _tree_lock_timed(self):
        """RLock acquire wrapper that records wait time for profiling."""
        return _TimedRLock(self._tree_lock, self._metrics_lock, self.stats)


class _TimedRLock:
    """Context-manager wrapper around an RLock that bills wait time to stats."""

    __slots__ = ('lock', 'metrics_lock', 'stats', '_t0')

    def __init__(self, lock, metrics_lock, stats):
        self.lock = lock
        self.metrics_lock = metrics_lock
        self.stats = stats
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.time()
        self.lock.acquire()
        elapsed = time.time() - self._t0
        with self.metrics_lock:
            self.stats.tree_lock_wait_seconds += elapsed
        return self

    def __exit__(self, exc_type, exc, tb):
        self.lock.release()
        return False
