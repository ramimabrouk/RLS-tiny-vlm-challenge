"""Small helpers shared by the scripts: config loading, seeding, device choice, plotting."""

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op without a GPU


def get_device(flag: str = "auto") -> torch.device:
    """One flag for CPU/GPU: "auto" picks cuda when available, otherwise cpu."""
    if flag == "auto":
        flag = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(flag)


def plot_curves(path, series: dict, xlabel: str, ylabel: str, title: str, logy: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, ys in series.items():
        ax.plot(range(1, len(ys) + 1), ys, label=name)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
