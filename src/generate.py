"""Greedy decoding: the image is encoded once, then the word is produced one letter per step.

Usage (from the repo root):
    python -m src.generate --checkpoint runs/baseline_seed0/best.pt --split test --n 8
"""

import argparse

import torch

from src.data import ShapeScenes, normalize_images
from src.model.vlm import VLM
from src.tokenizer import EOS_ID, decode
from src.utils import get_device

MAX_LETTERS = 45


@torch.no_grad()
def greedy_generate(model: VLM, images: torch.Tensor, max_letters: int = MAX_LETTERS) -> list[str]:
    """images: (B, 3, 64, 64) float, already normalized and on the model's device. Returns B words."""
    was_training = model.training
    model.eval()  # dropout off, BatchNorm uses its running statistics
    visual = model.encode_image(images)  # (B, N, D): the CNN runs once, not once per letter
    B = images.size(0)
    ids = torch.empty(B, 0, dtype=torch.long, device=images.device)  # (B, 0) -> (B, t) letters so far
    finished = torch.zeros(B, dtype=torch.bool, device=images.device)
    for _ in range(max_letters):
        logits = model.decode(visual, ids)  # (B, t + 1, 27)
        next_id = logits[:, -1].argmax(dim=-1)  # (B,) greedy: most likely next letter
        ids = torch.cat([ids, next_id[:, None]], dim=1)  # (B, t + 1)
        finished |= next_id == EOS_ID
        if finished.all():
            break
    model.train(was_training)
    return [decode(row) for row in ids.cpu()]  # decode() cuts each row at its first <eos>


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)  # contains a config dict, not just tensors
    model = VLM(**ckpt["config"]["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt["config"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    device = get_device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)
    ds = ShapeScenes(args.split, args.data_dir, limit=args.n)
    preds = greedy_generate(model, normalize_images(ds.images.to(device)))
    for ref, pred in zip(ds.words, preds):
        print(f"{'OK ' if ref == pred else 'BAD'} ref={ref}\n    out={pred}")
