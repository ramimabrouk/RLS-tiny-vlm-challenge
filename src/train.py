"""Training loop.

Forward pass, cross-entropy loss with padding excluded via ignore_index, backward pass, optimizer step,
periodic validation, and checkpointing (resumable).

Usage (from the repo root):
    python -m src.train --config configs/baseline.yaml --seed 0
    python -m src.train --config configs/baseline.yaml --seed 0 --resume      # continue after a Colab disconnect
    python -m src.train --config configs/baseline.yaml --epochs 1 --subset 2000 --device cpu   # quick CPU debug run
Outputs go to runs/<name>_seed<seed>/ (or --out-dir): last.pt, best.pt, metrics.csv, loss_curves.png.
"""

import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import corrupt_images, make_loader, normalize_images
from src.evaluate import teacher_forced
from src.model.vlm import VLM
from src.tokenizer import IGNORE_INDEX
from src.utils import get_device, load_config, plot_curves, set_seed

METRIC_FIELDS = ["epoch", "train_loss", "val_loss", "val_letter_acc", "lr", "epoch_seconds"]


def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warm-up to the base lr, then cosine decay to 0 (multiplicative factor on the base lr)."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, image_mode, grad_clip):
    model.train()
    loss_sum, n_tokens = 0.0, 0
    for images, input_ids, targets in tqdm(loader, leave=False, desc="train"):
        # images stay uint8 until they are on the device (4x less data to copy), then become float in 0..1
        images = normalize_images(corrupt_images(images, image_mode).to(device, non_blocking=True))
        input_ids, targets = input_ids.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        logits = model(images, input_ids)  # (B, T, 27)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))  # padding (-100) is skipped

        optimizer.zero_grad(set_to_none=True)  # gradients accumulate by default: clear the previous step's
        loss.backward()  # d loss / d every parameter
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # rescale if the gradient norm exceeds grad_clip
        optimizer.step()  # AdamW update
        scheduler.step()  # move to the next learning rate

        n = (targets != IGNORE_INDEX).sum().item()
        loss_sum += loss.item() * n  # .item() waits for the GPU: fine here, it is not a benchmark
        n_tokens += n
    return loss_sum / n_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--subset", type=int, default=None, help="train on the first N images only (debugging)")
    parser.add_argument("--resume", action="store_true", help="continue from <out-dir>/last.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    tcfg, dcfg = cfg["train"], cfg["data"]
    out_dir = Path(args.out_dir or f"runs/{cfg['name']}_seed{cfg['seed']}")
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg["seed"])
    device = get_device(args.device)
    train_loader = make_loader("train", tcfg["batch_size"], True, dcfg["dir"], dcfg["num_workers"], args.subset)
    val_loader = make_loader("val", 250, False, dcfg["dir"])
    model = VLM(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    total_steps = tcfg["epochs"] * len(train_loader)
    scheduler = make_scheduler(optimizer, tcfg["warmup_steps"], total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)  # mean over the non-padded target positions

    start_epoch, best_val, history = 0, float("inf"), []
    if args.resume and (out_dir / "last.pt").exists():
        ckpt = torch.load(out_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, best_val, history = ckpt["epoch"], ckpt["best_val"], ckpt["history"]
        print(f"resumed from epoch {start_epoch}")

    print(f"{cfg['name']} | seed {cfg['seed']} | device {device} | {model.num_parameters():,} parameters | "
          f"{len(train_loader)} steps/epoch | image_mode={dcfg['image_mode']}")

    for epoch in range(start_epoch, tcfg["epochs"]):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device,
                                     dcfg["image_mode"], tcfg["grad_clip"])
        val_loss, val_acc = teacher_forced(model, val_loader, device, dcfg["image_mode"])
        row = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_letter_acc": val_acc,
               "lr": scheduler.get_last_lr()[0], "epoch_seconds": time.time() - t0}
        history.append(row)
        print(f"epoch {row['epoch']:3d} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"letter acc {val_acc:.4f} | {row['epoch_seconds']:.0f}s")

        state = {"model": model.state_dict(), "config": cfg, "epoch": epoch + 1, "val_loss": val_loss}
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, out_dir / "best.pt")
        torch.save({**state, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "best_val": best_val, "history": history}, out_dir / "last.pt")  # written every epoch
        with open(out_dir / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(history)
        plot_curves(out_dir / "loss_curves.png",
                    {"train": [r["train_loss"] for r in history], "val": [r["val_loss"] for r in history]},
                    "epoch", "cross-entropy loss", f"{cfg['name']} (seed {cfg['seed']})")

    print(f"done. next: python -m src.evaluate --checkpoint {out_dir / 'best.pt'} --split test")


if __name__ == "__main__":
    main()
