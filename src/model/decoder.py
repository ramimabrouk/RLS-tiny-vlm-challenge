"""Transformer decoder: pre-LayerNorm blocks (attention + MLP, both with residual connections).

Input is a sequence of already-embedded vectors (B, T, D): visual tokens followed by letter embeddings.
Learned absolute positional embeddings are added here, for the visual tokens too: without them the
model could not tell WHERE in the image a visual token comes from (attention ignores order by itself).
"""

import torch
import torch.nn as nn

from src.model.attention import MultiHeadAttention


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)  # (B, T, D) -> (B, T, D), applied to each position independently


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.ln1(x), mask))  # (B, T, D)  tokens exchange information
        x = x + self.mlp(self.ln2(x))  # (B, T, D)  each token is transformed on its own
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, max_len, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(max_len, d_model) * 0.02)  # (max_len, D)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)  # pre-LN blocks need one last LayerNorm before the head
        self.max_len = max_len

    def forward(self, x, mask=None):
        B, T, D = x.shape
        assert T <= self.max_len, f"sequence of {T} tokens exceeds max_len={self.max_len}"
        x = self.drop(x + self.pos_emb[:T])  # (B, T, D) + (T, D) broadcasts over the batch
        for block in self.blocks:
            x = block(x, mask)
        return self.ln_f(x)  # (B, T, D)
