"""Behavioural checks: no peeking at future letters, sensible initial loss, the model can learn, greedy decoding is well-formed."""

import math

import torch
import torch.nn as nn

from src.generate import greedy_generate
from src.model.vlm import VLM
from src.tokenizer import IGNORE_INDEX, LETTERS


def small_model(**kwargs):
    return VLM(d_model=32, n_heads=4, n_layers=2, enc_channels=(8, 16, 32), dropout=0.0, **kwargs)


def test_no_peeking_at_future_letters():
    torch.manual_seed(0)
    for visual_attention in ("bidirectional", "causal"):
        model = small_model(visual_attention=visual_attention).eval()
        images = torch.randn(2, 3, 64, 64)
        ids = torch.randint(0, 26, (2, 10))
        ids2 = ids.clone()
        ids2[:, 5:] = torch.randint(0, 26, (2, 5))  # change letters 5..9
        a, b = model(images, ids), model(images, ids2)
        # logits[:, j] only depends on letters 0..j-1, so logits 0..5 must be identical
        torch.testing.assert_close(a[:, :6], b[:, :6])
        assert not torch.allclose(a[:, 6:], b[:, 6:])


def test_first_prediction_uses_only_the_image():
    torch.manual_seed(1)
    model = small_model().eval()
    images = torch.randn(2, 3, 64, 64)
    a = model(images, torch.randint(0, 26, (2, 8)))[:, 0]
    b = model(images, torch.randint(0, 26, (2, 8)))[:, 0]
    torch.testing.assert_close(a, b)  # logits[:, 0] is computed at the last visual token


def test_initial_loss_is_near_ln27():
    torch.manual_seed(2)
    model = small_model().train()
    images = torch.randn(8, 3, 64, 64)
    ids = torch.randint(0, 26, (8, 12))
    targets = torch.randint(0, 27, (8, 13))
    loss = nn.CrossEntropyLoss()(model(images, ids).reshape(-1, 27), targets.reshape(-1))
    assert abs(loss.item() - math.log(27)) < 0.2, loss.item()  # ln 27 = 3.296: the model starts as "uniform guessing"


def test_padding_is_ignored_by_the_loss():
    torch.manual_seed(3)
    model = small_model().eval()
    images = torch.randn(2, 3, 64, 64)
    ids = torch.randint(0, 26, (2, 6))
    targets = torch.randint(0, 27, (2, 7))
    targets[1, 4:] = IGNORE_INDEX  # the second word is shorter: its last 3 positions are padding
    logits = model(images, ids).reshape(-1, 27)  # (B * T, 27)
    flat = targets.reshape(-1)
    with_ignore = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)(logits, flat)
    valid = flat != IGNORE_INDEX
    by_hand = nn.CrossEntropyLoss()(logits[valid], flat[valid])  # average over the real positions only
    torch.testing.assert_close(with_ignore, by_hand)


def test_model_can_overfit_a_tiny_batch():
    torch.manual_seed(4)
    model = small_model().train()
    images = torch.randn(4, 3, 64, 64)
    targets = torch.randint(0, 27, (4, 9))
    ids = targets[:, :-1].clamp(max=25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()
    first = None
    for _ in range(60):
        loss = criterion(model(images, ids).reshape(-1, 27), targets.reshape(-1))
        first = first if first is not None else loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.3 * first, (first, loss.item())


def test_greedy_generate_is_well_formed_and_deterministic():
    torch.manual_seed(5)
    model = small_model().train()  # greedy_generate must switch to eval itself, and restore train mode afterwards
    images = torch.randn(3, 3, 64, 64)
    words = greedy_generate(model, images)
    assert model.training
    assert len(words) == 3
    assert all(len(w) <= 45 and set(w) <= set(LETTERS) for w in words)
    assert words == greedy_generate(model, images)
