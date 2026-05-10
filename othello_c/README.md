# othello_c — AlphaZero Othello in C

C port of the Othello half of `alpha-zero-general`. The Python `BatchedMCTS` is
GIL-bound on Colab T4 and tops out at ~1.4× over single-threaded MCTS; this
port replaces MCTS with a true multithreaded implementation in C and keeps
training in Python so we don't have to rewrite Adam/BatchNorm/backprop.

```
                       ┌─────────────────────────────────────────┐
                       │                                         │
   ./othello_c selfplay (writes examples_*.bin)                  │
                       │                                         │
                       ▼                                         │
        python python/train_from_examples.py                     │
            (loads examples, trains PyTorch, exports .onnx)      │
                       │                                         │
                       ▼                                         │
                 model_iter_N+1.onnx                             │
                       │                                         │
                       └──────────── next iteration ◄────────────┘
```

## What's in the box

| Component | Files | Status |
| --- | --- | --- |
| Bitboard rules (6×6 packed in `uint64_t`) | `include/othello.h`, `src/othello.c` | Bit-exact parity with Python on 10k random plays |
| Zobrist hash + open-addressing tree table | `include/zobrist.h`, `include/tree.h`, `src/*.c` | Verified on 50k boards |
| ONNX Runtime wrapper (CUDA EP supported) | `include/nn.h`, `src/nn.c` | Output matches PyTorch within 1e-8 |
| Serial MCTS | `src/mcts.c` (`search_serial`) | Bit-exact pi vs `BatchedMCTS(workers=1, vloss=0)` |
| Multithreaded MCTS (workers + virtual loss + batcher thread) | `src/mcts.c` (`search_parallel`) | Smoke-tested; produces sane distributions |
| Coach (self-play) | `include/coach.h`, `src/coach.c` | Writes raw examples; symmetries done in Python |
| Arena (head-to-head) | `include/arena.h`, `src/arena.c` | Greedy `argmax(pi)` at temp=0 |
| CLI: `selfplay`, `arena`, `bench-mcts` | `src/main.c` | |
| Benchmark sweep | `benchmarks/bench_mcts.c` | Mirrors `/benchmark_mcts.py` output |
| Python: ONNX export, training, parity utilities | `python/*.py` | Reuses `NNetWrapper` from the parent repo |

## Install

### macOS (Apple Silicon, CPU only — for development & parity tests)

```sh
brew install onnxruntime
pip install onnx onnxruntime onnxscript torch numpy
```

### Linux + CUDA (Colab T4 — for the benchmark numbers)

Download the GPU binary release matching your CUDA version:

```sh
# example for ORT 1.20.0 with CUDA 12; see https://github.com/microsoft/onnxruntime/releases
ORT_VER=1.20.0
curl -L -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-x64-gpu-${ORT_VER}.tgz
mkdir -p ~/onnxruntime && tar -xzf ort.tgz -C ~/onnxruntime --strip-components=1
export ONNXRUNTIME_ROOT_DIR=~/onnxruntime
pip install onnx onnxruntime-gpu onnxscript torch numpy
```

`ONNXRUNTIME_ROOT_DIR` is honoured by `CMakeLists.txt`; the brew prefix is
auto-detected on macOS.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Build options:
- `-DCMAKE_BUILD_TYPE=Debug` — `-O0 -g` plus AddressSanitizer + UBSan.
- `-DONNXRUNTIME_ROOT_DIR=/path/to/ort` — override ONNX Runtime location.

## Test

The test fixtures live in `tests/`. They are produced by the helper Python
scripts and committed alongside the C source so a fresh checkout can run
`ctest` immediately. To regenerate them:

```sh
python python/dump_othello_fixture.py --out tests/fixture_othello.bin
python python/export_onnx.py --no-pth --out tests/random_model.onnx
python python/dump_nn_fixture.py   --pth tests/random_model.onnx.pth.tar --out tests/nn_fixture.bin
python python/dump_mcts_fixture.py --pth tests/random_model.onnx.pth.tar --out tests/mcts_fixture.bin
```

Then:

```sh
ctest --test-dir build --output-on-failure
```

The five tests cover M1–M5 of the plan: rules parity, tree sanity, NN parity,
serial MCTS parity (bit-exact), and multithreaded MCTS smoke.

## Use

### One self-play / train cycle

```sh
# 1. Export current PyTorch checkpoint to .onnx
python python/export_onnx.py --pth temp/best.pth.tar --out temp/best.onnx

# 2. Run self-play in C (multi-threaded, batched evaluator)
build/othello_c selfplay \
    --model temp/best.onnx \
    --episodes 100 --sims 25 --workers 8 --batch 8 --vloss 1.0 \
    --out temp/examples_iter1.bin --seed 0 --cuda

# 3. Train PyTorch on the new examples and export the next .onnx
python python/train_from_examples.py \
    --pth-in temp/best.pth.tar \
    --pth-out temp/iter1.pth.tar \
    --onnx-out temp/iter1.onnx \
    --examples temp/examples_iter1.bin

# 4. Arena gate — accept the new model only if it wins ≥ 60%
build/othello_c arena \
    --a temp/best.onnx --b temp/iter1.onnx \
    --games 40 --sims 25 --workers 8 --batch 8 --vloss 1.0 --cuda
```

A shell loop driving steps 1–4 reproduces `Coach.learn`. The acceptance
threshold (`updateThreshold = 0.6`) is enforced by the caller, since neither
the C nor the Python side commits to a particular driver.

### Benchmark (compare to /benchmark_mcts.py)

```sh
build/bench_mcts --model temp/best.onnx --sims 50 \
    --workers 1,2,4,8,16 --batches 1,4,8,16 --cuda
python /benchmark_mcts.py --sims 50 \
    --workers 1,2,4,8,16 --batches 1,4,8,16
```

The two output tables are the apples-to-apples comparison.

## Performance notes

The two factors driving speedup are independent:

1. **Constant-factor (single-thread C vs single-thread Python).**
   Replaces NumPy/dict ops with bitboards + flat hash tables. On Colab T4 we
   expect 20–50× on MCTS-bound workloads.

2. **Threading (no GIL).** Workers run leaf-parallel descent in true parallel;
   bookkeeping uses virtual loss + a single-thread batcher to coalesce GPU
   calls. On 8 cores expect 4–8× on top of (1).

Together: 100–300× over the current Python pipeline is realistic for the
MCTS-dominated regime. End-to-end training-iteration speedup is smaller
(~10–50×) once the bottleneck shifts to the NN/GPU.

**On macOS CPU specifically:** PyTorch uses Apple's Accelerate framework on
small batches and is faster than ONNX Runtime CPU at the same batch size, so
you may see *no* speedup locally. The numbers that matter are on Colab T4
(CUDA), where ORT and PyTorch share cuDNN kernels and batching wins
dominate. Run the benchmark there.

## Layout

```
othello_c/
├── CMakeLists.txt
├── README.md
├── include/         # public headers
├── src/             # C sources
├── python/          # exporter, fixture generators, training driver
├── tests/           # test_*.c plus committed binary fixtures
└── benchmarks/      # bench_mcts.c
```

## Open follow-ups

- Lock sharding (replace single tree mutex with `key % 16` buckets), once a
  T4 baseline is on file. Likely the next 2–4× lever.
- Generalise `othello.h` into a `Game` vtable so `connect4` / `tictactoe`
  reuse the same MCTS without forking.
- Train in C (cuDNN forward+backward + Adam) — only worth it if/when the NN
  itself shows up as a bottleneck for Othello-size models.
