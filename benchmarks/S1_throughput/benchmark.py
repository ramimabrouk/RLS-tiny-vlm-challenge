"""S1 -- training throughput (images/second) vs batch size, on CPU and GPU.

A "training step" = forward + loss + backward + optimizer step, on the model of --config.
The three traps of the brief and how this script handles them:
  1. Warm-up: the first iterations of every setting are timed one by one and saved to warmup.csv
     (first CUDA calls load kernels and initialise cuBLAS/cuDNN, allocator caches fill, CPU caches are cold).
     They are then excluded from the measurement.
  2. Asynchronous execution: torch.cuda.synchronize() before starting and before reading the timer.
  3. Data loading: OUTSIDE the timing by default. The batch is random data of the right shape and dtype (uint8
     images, fixed word length), created once and already on the device. With --include-transfer the host->device
     copy + uint8->float conversion is inside the timing (the images then start on the CPU at every step).
     The Dataset/DataLoader cost is not measured here at all (see benchmarks/L2_profile for that).

Run from the repo root:
    python -m benchmarks.S1_throughput.benchmark --devices cpu cuda --batch-sizes 1 16 64 256
Writes timings.csv (one row per repeat), warmup.csv, hardware.txt in this folder; plot with plot.py.
"""

import argparse
import csv
import platform
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.data import normalize_images
from src.model.vlm import VLM
from src.tokenizer import IGNORE_INDEX
from src.utils import load_config, set_seed

OUT_DIR = Path(__file__).parent


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()  # GPU calls return immediately: wait until the queued work is really finished


def write_hardware(path: Path) -> None:
    lines = [f"platform: {platform.platform()}", f"python: {platform.python_version()}",
             f"torch: {torch.__version__}", f"torch cpu threads: {torch.get_num_threads()}"]
    for cmd in (["lscpu"], ["nvidia-smi"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
            lines += ["", f"$ {' '.join(cmd)}", out]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            lines += ["", f"$ {' '.join(cmd)}", "(not available)"]
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        lines += ["", f"cuda: {torch.version.cuda} | gpu: {p.name} | memory: {p.total_memory / 2**30:.1f} GiB"]
    path.write_text("\n".join(lines) + "\n")


def make_step(model, optimizer, criterion, images, input_ids, targets, device, include_transfer):
    def step():
        x = images
        if include_transfer:
            x = x.to(device)  # uint8 (B, 3, 64, 64) host -> device
        x = normalize_images(x)
        logits = model(x, input_ids)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16, 64, 256])
    parser.add_argument("--letters", type=int, default=25, help="word length used for every batch")
    parser.add_argument("--warmup", type=int, default=5, help="warm-up iterations, timed one by one, then discarded")
    parser.add_argument("--iters", type=int, default=20, help="timed iterations per repeat")
    parser.add_argument("--repeats", type=int, default=5, help="repeats per setting: gives the mean and the spread")
    parser.add_argument("--include-transfer", action="store_true")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cfg = load_config(args.config)
    write_hardware(out_dir / "hardware.txt")
    timings, warmups = [], []

    for device_name in args.devices:
        if device_name == "cuda" and not torch.cuda.is_available():
            print("cuda not available: skipped")
            continue
        device = torch.device(device_name)
        for bs in args.batch_sizes:
            set_seed(0)
            model = VLM(**cfg["model"]).to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            images = torch.randint(0, 256, (bs, 3, 64, 64), dtype=torch.uint8)
            input_ids = torch.randint(0, 26, (bs, args.letters), device=device)
            targets = torch.randint(0, 27, (bs, args.letters + 1), device=device)
            if not args.include_transfer:
                images = images.to(device)
            step = make_step(model, optimizer, criterion, images, input_ids, targets, device, args.include_transfer)

            for i in range(args.warmup):  # trap 1: time each warm-up iteration on its own
                sync(device)
                t0 = time.perf_counter()
                step()
                sync(device)
                warmups.append({"device": device_name, "batch_size": bs, "iteration": i + 1,
                                "ms": (time.perf_counter() - t0) * 1e3})
            for r in range(args.repeats):
                sync(device)  # trap 2: nothing from before may still be running when the timer starts ...
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    step()
                sync(device)  # ... and everything must be finished before the timer is read
                seconds = time.perf_counter() - t0
                timings.append({"device": device_name, "batch_size": bs, "repeat": r + 1, "iters": args.iters,
                                "seconds": seconds, "images_per_sec": bs * args.iters / seconds,
                                "include_transfer": args.include_transfer})
            rates = [t["images_per_sec"] for t in timings if t["device"] == device_name and t["batch_size"] == bs]
            print(f"{device_name:4s} batch {bs:4d}: {sum(rates) / len(rates):9.1f} images/s "
                  f"(min {min(rates):.1f}, max {max(rates):.1f}) | first iteration {warmups[-args.warmup]['ms']:.0f} ms")

    for name, rows in (("timings.csv", timings), ("warmup.csv", warmups)):
        if rows:
            with open(out_dir / name, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    print(f"saved -> {out_dir}/timings.csv, warmup.csv, hardware.txt")


if __name__ == "__main__":
    main()
