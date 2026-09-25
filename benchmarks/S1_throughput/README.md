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

## Your analysis (write it yourself, from your own measurements)

- Why are the first iterations slower? (evidence: `warmup.csv`)
- Where does the GPU stop being faster than the CPU, if anywhere, and why?
