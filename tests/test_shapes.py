"""Shape sanity checks across the pipeline.

Verifies tensor shapes at each stage, from the input image to the final
letter logits, for a range of batch sizes and sequence lengths.
"""

import pytest
import torch

from src.model.vlm import VLM, make_attention_mask
from src.tokenizer import VOCAB_SIZE


@pytest.fixture(scope="module")
def model():
    return VLM(d_model=32, n_heads=4, n_layers=2, enc_channels=(8, 16, 32)).eval()


@pytest.mark.parametrize("B", [1, 3])
def test_encoder_and_adapter_shapes(model, B):
    images = torch.randn(B, 3, 64, 64)
    fmap = model.encoder(images)
    assert fmap.shape == (B, 32, 8, 8)  # (B, C, H', W'): three 2x2 poolings, 64 -> 8
    visual = model.adapter(fmap)
    assert visual.shape == (B, 64, 32)  # (B, H'*W', d_model)
    assert model.n_visual == 64


@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("L", [0, 1, 13, 44, 45])
def test_logits_shape(model, B, L):
    images = torch.randn(B, 3, 64, 64)
    input_ids = torch.randint(0, 26, (B, L))
    logits = model(images, input_ids)
    assert logits.shape == (B, L + 1, VOCAB_SIZE)  # L letters in -> L + 1 predictions (first letter ... <eos>)


def test_sequence_too_long_is_rejected(model):
    with pytest.raises(AssertionError):
        model(torch.randn(1, 3, 64, 64), torch.randint(0, 26, (1, 46)))  # 64 + 46 > max_len


def test_mask_bidirectional():
    mask = make_attention_mask(n_visual=3, n_text=2, visual_attention="bidirectional", device="cpu")
    assert mask.shape == (5, 5)
    assert mask[:3, :3].all()  # visual tokens see all visual tokens
    assert not mask[:3, 3:].any()  # ... and no letter
    assert torch.equal(mask[3:], torch.tril(torch.ones(5, 5, dtype=torch.bool))[3:])  # letters: causal


def test_mask_causal():
    mask = make_attention_mask(n_visual=3, n_text=2, visual_attention="causal", device="cpu")
    assert torch.equal(mask, torch.tril(torch.ones(5, 5, dtype=torch.bool)))
