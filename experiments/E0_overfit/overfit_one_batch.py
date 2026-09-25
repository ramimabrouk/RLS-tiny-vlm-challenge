"""E0 -- overfit one batch: a bug check, not a result.

If the pipeline is correct, a model that sees the SAME batch over and over must drive the loss to ~0.
Dropout is switched off (it would keep the loss from reaching 0 on purpose).
Run from the repo root:
    python -m experiments.E0_overfit.overfit_one_batch --config configs/baseline.yaml --device cuda
Writes experiments/E0_overfit/loss.csv and overfit_loss.png.
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from src.data import make_loader, normalize_images
from src.model.vlm import VLM
from src.tokenizer import IGNORE_INDEX
from src.utils import get_device, load_config, plot_curves, set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(args.device)
    cfg["model"]["dropout"] = 0.0
    images, input_ids, targets = next(iter(make_loader("train", args.batch_size, True, cfg["data"]["dir"], limit=args.batch_size)))
    images, input_ids, targets = normalize_images(images.to(device)), input_ids.to(device), targets.to(device)

    model = VLM(**cfg["model"]).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    losses = []
    for step in range(args.steps):
        logits = model(images, input_ids)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 25 == 0 or step == args.steps - 1:
            print(f"step {step:4d} | loss {losses[-1]:.5f}")

    out = Path(__file__).parent
    with open(out / "loss.csv", "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows([["step", "loss"]] + [[i + 1, l] for i, l in enumerate(losses)])
    plot_curves(out / "overfit_loss.png", {"train loss (one fixed batch)": losses}, "step", "cross-entropy loss",
                "E0: overfitting one batch", logy=True)
    print(f"first loss {losses[0]:.3f} (ln 27 = 3.296) -> last loss {losses[-1]:.5f}")
