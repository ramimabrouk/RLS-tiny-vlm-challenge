"""Level 2 -- where does one training step spend its time? (data loading / transfer / forward / backward / optimizer)

Part 1: manual phase timing with torch.cuda.synchronize() between phases (so each phase is really finished when its
        timer stops). The syncs remove the CPU/GPU overlap you get in normal training, so the phases add up to a slightly
        larger time than an unsynchronised step: say so in your report.
Part 2: the same step under torch.profiler, to see WHICH operators dominate (table + a Chrome trace you can open at
        chrome://tracing or https://ui.perfetto.dev to see kernels, gaps, and the GPU sitting idle).

Run from the repo root (real DataLoader, real data):
    python -m benchmarks.L2_profile.profile_step --device cuda --batch-size 64 --num-workers 0
Change one thing at a time (batch size, num_workers, pin_memory) and keep the CSVs.
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader

from src.data import ShapeScenes, collate, normalize_images
from src.model.vlm import VLM
from src.tokenizer import IGNORE_INDEX
from src.utils import get_device, load_config, set_seed

OUT_DIR = Path(__file__).parent
PHASES = ["data_loading", "transfer", "forward", "backward", "optimizer"]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cfg = load_config(args.config)
    set_seed(0)
    device = get_device(args.device)
    model = VLM(**cfg["model"]).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    loader = DataLoader(ShapeScenes("train", cfg["data"]["dir"]), batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=args.num_workers, pin_memory=args.pin_memory)
    non_blocking = args.pin_memory  # non_blocking copies only overlap with compute from pinned memory

    def timed_step(batch_iter):
        t = {}
        sync(device)
        t0 = time.perf_counter()
        images, input_ids, targets = next(batch_iter)  # runs the Dataset __getitem__ + collate (+ worker wait)
        t["data_loading"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        images = normalize_images(images.to(device, non_blocking=non_blocking))
        input_ids, targets = input_ids.to(device, non_blocking=non_blocking), targets.to(device, non_blocking=non_blocking)
        sync(device)
        t["transfer"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        logits = model(images, input_ids)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        sync(device)
        t["forward"] = time.perf_counter() - t0

        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        loss.backward()
        sync(device)
        t["backward"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        optimizer.step()
        sync(device)
        t["optimizer"] = time.perf_counter() - t0
        return t

    batches = iter(loader)
    rows = []
    for i in range(args.warmup + args.steps):
        t = timed_step(batches)
        if i >= args.warmup:
            rows.append({"step": i - args.warmup + 1, **{k: v * 1e3 for k, v in t.items()}})  # milliseconds

    with open(out_dir / "breakdown.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step"] + PHASES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    total = sum(statistics.mean(r[p] for r in rows) for p in PHASES)
    print(f"device {device} | batch {args.batch_size} | workers {args.num_workers} | pin_memory {args.pin_memory}")
    for p in PHASES:
        vals = [r[p] for r in rows]
        print(f"{p:13s} {statistics.mean(vals):8.2f} ms +- {statistics.stdev(vals):6.2f}   "
              f"{100 * statistics.mean(vals) / total:5.1f} %")
    print(f"{'total':13s} {total:8.2f} ms  ->  {1e3 * args.batch_size / total:.0f} images/s (with syncs between phases)")

    # ---- Part 2: operator-level profile of a few steps (data is fetched before the profiled region) ----
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    prepared = [next(batches) for _ in range(4)]
    with profile(activities=activities, record_shapes=True) as prof:
        for images, input_ids, targets in prepared:
            with record_function("transfer"):
                images = normalize_images(images.to(device))
                input_ids, targets = input_ids.to(device), targets.to(device)
            with record_function("forward"):
                logits = model(images, input_ids)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            with record_function("backward"):
                loss.backward()
            with record_function("optimizer"):
                optimizer.step()
        sync(device)
    table = prof.key_averages().table(sort_by=sort_key, row_limit=20)
    (out_dir / "profiler_table.txt").write_text(table)
    prof.export_chrome_trace(str(out_dir / "trace.json"))
    print(table)
    print(f"saved -> {out_dir}/breakdown.csv, profiler_table.txt, trace.json")


if __name__ == "__main__":
    main()
