"""Level 2 -- your attention vs F.scaled_dot_product_attention: speed and peak memory as the sequence grows.

Both compute the same causal attention on the same random q, k, v (B, H, T, dh). Yours materialises the (B, H, T, T)
score and weight matrices (memory ~ T^2, several passes over it); the fused kernel does not have to. F.scaled_dot_product_attention
is allowed in benchmarks (not in the model). Forward and forward+backward are timed separately.

Run from the repo root:
    python -m benchmarks.L2_attention.bench_attention --device cuda
Peak memory needs CUDA (torch.cuda.max_memory_allocated); on CPU only the timings are meaningful.
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.model.attention import scaled_dot_product
from src.utils import get_device

OUT_DIR = Path(__file__).parent


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure(fn, device, iters, repeats, backward):
    """Returns (ms per call: list over repeats, peak GPU memory in MiB or None)."""
    def run():
        out = fn()
        if backward:
            out.sum().backward()

    for _ in range(3):  # warm-up
        run()
    peak = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    times = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            run()
        sync(device)
        times.append((time.perf_counter() - t0) / iters * 1e3)
    if device.type == "cuda":
        peak = (torch.cuda.max_memory_allocated() - base) / 2**20  # extra memory used during the call (MiB)
    return times, peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[64, 128, 256, 512, 1024])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-head", type=int, default=32)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    device = get_device(args.device)
    rows = []
    for T in args.seq_lens:
        q, k, v = (torch.randn(args.batch_size, args.heads, T, args.d_head, device=device, requires_grad=True)
                   for _ in range(3))
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        impls = {
            "handwritten": lambda: scaled_dot_product(q, k, v, mask)[0],
            "sdpa": lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask),
        }
        for name, fn in impls.items():
            for backward in (False, True):
                times, peak = measure(fn, device, args.iters, args.repeats, backward)
                rows.append({"impl": name, "seq_len": T, "pass": "fwd+bwd" if backward else "fwd",
                             "ms_mean": statistics.mean(times), "ms_std": statistics.stdev(times),
                             "peak_extra_mib": peak if peak is not None else ""})
                print(f"T={T:5d} {name:11s} {'fwd+bwd' if backward else 'fwd    '}: {statistics.mean(times):8.3f} ms "
                      f"+- {statistics.stdev(times):.3f}" + (f" | +{peak:8.1f} MiB" if peak is not None else ""))
    with open(Path(args.out_dir) / "attention_bench.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"device: {device} | saved -> {args.out_dir}/attention_bench.csv")


if __name__ == "__main__":
    main()
