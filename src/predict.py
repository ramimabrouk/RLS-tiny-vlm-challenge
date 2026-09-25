"""Describe one image with a trained model.

Usage (from the repo root):
    python -m src.predict --checkpoint runs/baseline_seed0/best.pt --image my_scene.png
The model only knows ShapeScenes-style pictures (flat-colored shapes on a light background, 64x64).
Photos or drawings in another style are out of distribution: the output will be meaningless.
"""

import argparse

import numpy as np
import torch
from PIL import Image

from src.data import normalize_images
from src.evaluate import parse_word
from src.generate import greedy_generate, load_checkpoint
from src.utils import get_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    img = Image.open(args.image).convert("RGB")
    if img.size != (64, 64):
        print(f"warning: image is {img.size}, resizing to 64x64 (results may degrade)")
        img = img.resize((64, 64), Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None]  # (64, 64, 3) uint8 -> (1, 3, 64, 64)
    word = greedy_generate(model, normalize_images(x.to(device)))[0]
    print("raw output :", word)
    print("parsed     :", {k: v for k, v in parse_word(word).items() if v is not None})
