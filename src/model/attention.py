"""Handwritten multi-head self-attention (no nn.MultiheadAttention, no F.scaled_dot_product_attention).

Notation: B batch, T sequence length, D = d_model, H heads, dh = D / H (head dimension).
Masks are boolean and broadcastable to (B, H, T, T): True = "this query may look at this key".
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product(q, k, v, mask=None, dropout=None):
    """softmax(q k^T / sqrt(dh)) v.   q, k, v: (B, H, T, dh).   Returns (out (B, H, T, dh), weights (B, H, T, T))."""
    dh = q.size(-1)
    # Dividing by sqrt(dh) keeps the score variance ~1 whatever dh is; otherwise softmax saturates
    # (one-hot) for large dh and its gradient vanishes.
    scores = q @ k.transpose(-2, -1) / math.sqrt(dh)  # (B, H, T, T)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))  # exp(-inf) = 0 -> weight exactly 0
    weights = F.softmax(scores, dim=-1)  # (B, H, T, T), each row sums to 1 over the keys
    if dropout is not None:
        weights = dropout(weights)
    return weights @ v, weights  # (B, H, T, dh)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        # (B, T, D) -> (B, T, H, dh) -> (B, H, T, dh): heads become a batch-like dimension
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, dh = x.shape
        # (B, H, T, dh) -> (B, T, H, dh) -> (B, T, D); contiguous() because view needs contiguous memory
        return x.transpose(1, 2).contiguous().view(B, T, H * dh)

    def forward(self, x, mask=None, return_weights=False):
        q = self.split_heads(self.q_proj(x))  # (B, H, T, dh)
        k = self.split_heads(self.k_proj(x))  # (B, H, T, dh)
        v = self.split_heads(self.v_proj(x))  # (B, H, T, dh)
        out, weights = scaled_dot_product(q, k, v, mask, self.attn_dropout)  # (B, H, T, dh)
        out = self.out_proj(self.merge_heads(out))  # (B, T, D)
        return (out, weights) if return_weights else out
