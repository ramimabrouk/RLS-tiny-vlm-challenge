"""PyTorch Dataset and collate function for ShapeScenes.

Loads the image + word pairs written by generate_data.py and returns
(image tensor, token id tensor) pairs, batched with a collate function
that pads to the longest word in the batch.

Run `python -m src.data` from the repo root to print a decoded sample and batch shapes.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import IGNORE_INDEX, decode, encode, pad_batch

SPLITS = ("train", "val", "test", "test_heldout")


class ShapeScenes(Dataset):
    """One split of ShapeScenes, fully in memory (~250 MB for train as uint8)."""

    def __init__(self, split: str, data_dir: str | Path = "data", limit: int | None = None):
        assert split in SPLITS, f"split must be one of {SPLITS}"
        blob = torch.load(Path(data_dir) / f"{split}.pt", map_location="cpu")
        self.images = blob["images"]  # (N, 3, 64, 64) uint8, values 0..255
        self.words = blob["words"]  # list[str] of length N
        assert len(self.images) == len(self.words)
        if limit is not None:  # quick CPU debugging: keep only the first `limit` samples
            self.images, self.words = self.images[:limit], self.words[:limit]

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, idx: int):
        image = self.images[idx]  # (3, 64, 64) uint8
        ids = torch.tensor(encode(self.words[idx]), dtype=torch.long)  # (L + 1,) letters then <eos>
        return image, ids


def collate(batch):
    """Turn a list of (image, ids) into one batch.

    Returns:
        images:    (B, 3, 64, 64) uint8
        input_ids: (B, T - 1) long   letters fed to the decoder (pad positions -> 0, any valid id works)
        targets:   (B, T)     long   letters + <eos>, padded with IGNORE_INDEX; T = longest word + 1

    input_ids is targets shifted right by one: the image predicts targets[:, 0], input_ids[:, 0]
    predicts targets[:, 1], and so on. The last target position has no input after it.
    """
    images, ids = zip(*batch)
    images = torch.stack(images)  # (B, 3, 64, 64)
    targets = pad_batch([x.tolist() for x in ids])  # (B, T)
    input_ids = targets[:, :-1].clamp(min=0)  # (B, T-1), -100 -> 0
    return images, input_ids, targets


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """uint8 (B, 3, H, W) in 0..255 -> float32 in 0..1. Call it AFTER moving the batch to the device:
    a uint8 tensor is 4x smaller to copy to the GPU than a float32 one."""
    return images.float() / 255.0


def corrupt_images(images: torch.Tensor, mode: str) -> torch.Tensor:
    """E1 blind baseline. mode: "none" (keep), "zeros" (all-black images) or "shuffle" (images of the
    batch paired with the wrong words: the image is still a valid image, but unrelated to the target)."""
    if mode == "none":
        return images
    if mode == "zeros":
        return torch.zeros_like(images)
    if mode == "shuffle":
        return images[torch.randperm(images.size(0), device=images.device)]
    raise ValueError(f"unknown image_mode {mode!r}")


def make_loader(split, batch_size, shuffle, data_dir="data", num_workers=0, limit=None):
    return DataLoader(
        ShapeScenes(split, data_dir, limit),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
    )


def _preview(ds: ShapeScenes, path: str, n: int = 8):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, n // 2, figsize=(3 * n // 2, 7))
    for ax, i in zip(axes.flat, range(n)):
        ax.imshow(ds.images[i].permute(1, 2, 0).numpy())  # (3,H,W) -> (H,W,3) for matplotlib
        ax.set_title(ds.words[i], fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    print(f"preview saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=SPLITS)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--preview", metavar="PNG", help="save a grid of 8 images with their words")
    args = parser.parse_args()

    ds = ShapeScenes(args.split, args.data_dir)
    print(f"{args.split}: {len(ds)} samples, images {tuple(ds.images.shape)} {ds.images.dtype}")

    image, ids = ds[0]
    print("word    :", ds.words[0])
    print("ids     :", ids.tolist())
    print("decoded :", decode(ids))
    assert decode(ids) == ds.words[0]

    lengths = [len(w) for w in ds.words]
    print(f"word length: min {min(lengths)}, max {max(lengths)}")

    images, input_ids, targets = next(iter(make_loader(args.split, 4, shuffle=False, data_dir=args.data_dir)))
    print("batch images   :", tuple(images.shape), images.dtype)
    print("batch input_ids:", tuple(input_ids.shape))
    print("batch targets  :", tuple(targets.shape), f"(pads are {IGNORE_INDEX})")

    if args.preview:
        _preview(ds, args.preview)
