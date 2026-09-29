# S1 — Throughput benchmark

Required benchmark: training images/second across at least 4 batch sizes, on CPU and GPU, with warm-up,
torch.cuda.synchronize(), and data-loading placement handled explicitly. See Project Brief, section 9.

## How to run (from the repo root)

```bash
python -m benchmarks.S1_throughput.benchmark --devices cpu cuda --batch-sizes 1 16 64 256
python benchmarks/S1_throughput/plot.py
```

Files written here: `timings.csv` (raw, one row per repeat), `warmup.csv` (every warm-up iteration timed on its own),
`hardware.txt` (CPU, GPU, torch/CUDA versions), `throughput.png`.

## What is measured

A training step (forward + loss + backward + AdamW step) on random uint8 images and random words of 25 letters.
Data loading is **outside** the timing (batch already on the device); `--include-transfer` puts the host-to-device copy
inside. Say in your report which one you used.

## Analysis

**CPU throughput plateaus almost immediately** (19.4 -> 35.3 img/s from batch 1 to 16,
then flat with no further gain from 16 to 256: 35.3 -> 35.1 -> 33.8 img/s). A single CPU
core saturates its compute quickly; larger batches only add queuing, not parallelism.

**GPU throughput scales dramatically with batch size**: 62.6 img/s at batch 1, jumping to
1117 img/s at batch 16 (near-linear scaling for a 16x larger batch), then continuing to
grow to 1538-1542 img/s at batch 64-256 before flattening. Small batches leave most of the
GPU's parallel compute idle; larger batches are needed to fill it.

**The GPU only clearly wins once batch size is large enough**: at batch 1, GPU (62.6 img/s)
is only ~3.2x faster than CPU (19.4 img/s), and its "first iteration" overhead is much
higher (469 ms vs 109 ms) — a single-image GPU call is not obviously worth it. The GPU's
real advantage appears at batch 16 and above, where it is roughly 32x to 46x faster than
CPU (1542.5 / 33.8 = 45.6x at batch 256).

**Warm-up cost is real and correctly excluded from steady-state measurements**: the first
CPU iteration at batch 256 took 8408 ms, comparable to several steady-state iterations
combined. On GPU the same relative cost is much smaller (180 ms at batch 256) because most
CPU "first iteration" cost is genuine compute, while GPU's is mostly one-time CUDA context
and kernel setup — which is why `torch.cuda.synchronize()` before/after timing matters:
without it, this setup cost would silently leak into whichever measurement happened to run
first.

Hardware: Google Colab, Tesla T4 GPU, single CPU core (see `hardware.txt` for exact specs).