"""Tiny VLM = CNN encoder + adapter + Transformer decoder + linear head.

    images  (B, 3, 64, 64)
      encoder  -> (B, C, 8, 8)
      adapter  -> visual tokens (B, 64, D)                      64 = 8*8 spatial positions
    letters (B, L) -> embeddings (B, L, D)
    sequence = [visual tokens ; letter embeddings]              (B, 64 + L, D)
      decoder  -> (B, 64 + L, D)
      keep the last L + 1 positions (the last visual token and every letter) -> head -> (B, L + 1, 27)

The output at the last visual token predicts the first letter, the output at letter i predicts letter i+1,
and the output at the last letter predicts <eos>. That is why there are L + 1 logits for L input letters.
"""

import torch
import torch.nn as nn

from src.model.decoder import Decoder
from src.model.encoder import CNNEncoder
from src.tokenizer import VOCAB_SIZE


class Adapter(nn.Module):
    """Feature map (B, C, H', W') -> sequence of visual tokens (B, H'*W', D): flatten + linear projection."""

    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)

    def forward(self, fmap):
        tokens = fmap.flatten(2).transpose(1, 2)  # (B, C, H', W') -> (B, C, H'*W') -> (B, H'*W', C)
        return self.proj(tokens)  # (B, H'*W', D): one token per spatial location


def make_attention_mask(n_visual: int, n_text: int, visual_attention: str, device) -> torch.Tensor:
    """Boolean (T, T) mask, True = query row may attend to key column, with T = n_visual + n_text.

    - letters are always causal: a letter sees every visual token and the letters before it (and itself);
    - visual_attention="bidirectional": visual tokens see all visual tokens (and no letter) -> "prefix" mask;
    - visual_attention="causal": visual tokens are causal too, like words in a language model.
    Padding needs no key mask: padded letters sit at the END of the sequence, and real positions never look ahead.
    """
    assert visual_attention in ("bidirectional", "causal")
    T = n_visual + n_text
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))  # lower triangle = causal
    if visual_attention == "bidirectional":
        mask[:n_visual, :n_visual] = True
    return mask


class VLM(nn.Module):
    def __init__(
        self,
        d_model=128,
        n_heads=4,
        n_layers=4,
        mlp_ratio=4,
        dropout=0.1,
        enc_channels=(32, 64, 128),
        visual_attention="bidirectional",
        image_size=64,
        max_letters=45,
    ):
        super().__init__()
        self.encoder = CNNEncoder(tuple(enc_channels))
        side = image_size // self.encoder.downsample
        self.n_visual = side * side  # 64 with the defaults
        self.adapter = Adapter(self.encoder.out_channels, d_model)
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.decoder = Decoder(d_model, n_heads, n_layers, self.n_visual + max_letters, mlp_ratio, dropout)
        self.head = nn.Linear(d_model, VOCAB_SIZE)
        self.visual_attention = visual_attention
        # Small initial weights -> initial logits ~ 0 -> initial loss ~ ln(27) = 3.30 (a useful sanity check).
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def encode_image(self, images):
        return self.adapter(self.encoder(images))  # (B, 3, 64, 64) -> (B, 64, D)

    def decode(self, visual_tokens, input_ids):
        """visual_tokens (B, N, D), input_ids (B, L) -> logits (B, L + 1, 27). L may be 0."""
        N, L = visual_tokens.size(1), input_ids.size(1)
        x = torch.cat([visual_tokens, self.token_emb(input_ids)], dim=1)  # (B, N + L, D)
        mask = make_attention_mask(N, L, self.visual_attention, x.device)  # (N + L, N + L)
        h = self.decoder(x, mask)  # (B, N + L, D)
        h = h[:, N - 1 :, :]  # (B, L + 1, D): last visual token + all letters
        return self.head(h)  # (B, L + 1, 27)

    def forward(self, images, input_ids):
        return self.decode(self.encode_image(images), input_ids)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
