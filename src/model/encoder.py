"""CNN vision encoder: image (B, 3, 64, 64) -> feature map (B, C, H', W').

Each block is two 3x3 convolutions (each followed by BatchNorm + ReLU) and a 2x2 max-pool, so every
block halves the spatial size. With the default 3 blocks: 64 -> 32 -> 16 -> 8, i.e. an 8x8 grid = 64
visual tokens once flattened by the adapter (see vlm.py).
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            # padding=1 with a 3x3 kernel keeps H, W unchanged: out = (in + 2*1 - 3) / 1 + 1 = in
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # bias=False above: BatchNorm has its own shift (beta)
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # (H, W) -> (H/2, W/2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNEncoder(nn.Module):
    def __init__(self, channels=(32, 64, 128), in_channels: int = 3):
        super().__init__()
        blocks, previous = [], in_channels
        for c in channels:
            blocks.append(ConvBlock(previous, c))
            previous = c
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = previous  # C
        self.downsample = 2 ** len(channels)  # image side / feature-map side

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, 3, H, W) float -> (B, C, H / downsample, W / downsample)
        return self.blocks(images)
