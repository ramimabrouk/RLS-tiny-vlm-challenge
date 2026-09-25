"""Level 2 -- peak GPU memory of a training step vs batch size (or sequence length), to compare with YOUR prediction.

Workflow the brief asks for: FIRST write your estimate in notes (parameters + gradients + optimizer state + activations,
in bytes, as a formula of batch size and sequence length) and commit it; THEN run this script and compare.
The script splits the measurement into the pieces you have to explain:
    after model creation      -> parameters
    after forward             -> + activations saved for backward (the part that grows with batch size and T)
    after backward            -> parameters + gradients (most activations freed) -- peak usually happened before this
    after optimizer.step      -> + optimizer state (AdamW keeps 2 extra tensors per parameter)
    peak during the whole step (torch.cuda.max_memory_allocated)
Requires CUDA. Run from the repo root:
    python -m benchmarks.L2_memory.memory --batch-sizes 1 16 64 256 --letters 25
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from src.data import normalize_images
from src.model.vlm import VLM
from src.tokenizer import IGNORE_INDEX
from src.utils import load_config, set_seed

OUT_DIR = Path(__file__).parent
MIB = 2**20


def measure_one(cfg, bs, letters, device):
    """One training step at (batch size, word length). Everything allocated here is freed when the function returns,
    so successive measurements do not pollute each other."""
    set_seed(0)
    torch.cuda.empty_cache()
    model = VLM(**cfg["model"]).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    images = normalize_images(torch.randint(0, 256, (bs, 3, 64, 64), dtype=torch.uint8, device=device))
    input_ids = torch.randint(0, 26, (bs, letters), device=device)
    targets = torch.randint(0, 27, (bs, letters + 1), device=device)
    m_start = torch.cuda.memory_allocated()  # parameters + the input batch
    torch.cuda.reset_peak_memory_stats()  # from here on: peak of the step itself

    logits = model(images, input_ids)
    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    m_forward = torch.cuda.memory_allocated()
    loss.backward()
    m_backward = torch.cuda.memory_allocated()
    optimizer.step()
    m_step = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    return {"batch_size": bs, "letters": letters, "seq_len": model.n_visual + letters, "params": model.num_parameters(),
            "mib_params_fp32": model.num_parameters() * 4 / MIB, "mib_after_model_and_inputs": m_start / MIB,
            "mib_after_forward": m_forward / MIB, "mib_after_backward": m_backward / MIB,
            "mib_after_optimizer_step": m_step / MIB, "mib_peak": peak / MIB}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16, 64, 256])
    parser.add_argument("--letters", type=int, nargs="+", default=[25], help="word lengths (sequence = 64 + letters)")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this benchmark measures GPU memory: run it on a CUDA device (e.g. Colab T4)")

    device = torch.device("cuda")
    cfg = load_config(args.config)
    rows = []
    for letters in args.letters:
        for bs in args.batch_sizes:
            row = measure_one(cfg, bs, letters, device)
            rows.append(row)
            print(f"batch {bs:4d} letters {letters:3d} | peak {row['mib_peak']:8.1f} MiB | "
                  f"model+inputs {row['mib_after_model_and_inputs']:7.1f} | +fwd {row['mib_after_forward']:8.1f} | "
                  f"after bwd {row['mib_after_backward']:8.1f} | after opt {row['mib_after_optimizer_step']:8.1f}")
    with open(Path(args.out_dir) / "memory.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"GPU: {torch.cuda.get_device_name(0)} | saved -> {args.out_dir}/memory.csv")


if __name__ == "__main__":
    main()
