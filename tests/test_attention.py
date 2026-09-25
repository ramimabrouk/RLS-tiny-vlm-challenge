"""Equivalence of the handwritten attention with F.scaled_dot_product_attention (allowed in tests only),
plus properties that must hold for any correct attention: rows sum to 1, masked keys get weight 0, no peeking ahead."""

import pytest
import torch
import torch.nn.functional as F

from src.model.attention import MultiHeadAttention, scaled_dot_product


def reference_mha(mha: MultiHeadAttention, x, mask):
    """The same layer built from F.scaled_dot_product_attention, with head splitting written independently."""
    B, T, D = x.shape
    H, dh = mha.n_heads, D // mha.n_heads

    def heads(t):  # (B, T, D) -> (B, H, T, dh)
        return t.reshape(B, T, H, dh).permute(0, 2, 1, 3)

    out = F.scaled_dot_product_attention(
        heads(mha.q_proj(x)), heads(mha.k_proj(x)), heads(mha.v_proj(x)), attn_mask=mask
    )
    return mha.out_proj(out.permute(0, 2, 1, 3).reshape(B, T, D))


def causal_mask(T):
    return torch.tril(torch.ones(T, T, dtype=torch.bool))


def prefix_mask(n_visual, T):
    mask = causal_mask(T)
    mask[:n_visual, :n_visual] = True  # bidirectional among the visual tokens
    return mask


@pytest.mark.parametrize("B,T,D,H", [(2, 10, 32, 4), (3, 80, 64, 8), (1, 1, 16, 2)])
def test_matches_sdpa_causal(B, T, D, H):
    torch.manual_seed(0)
    mha = MultiHeadAttention(D, H).eval()
    x = torch.randn(B, T, D)
    mask = causal_mask(T)
    torch.testing.assert_close(mha(x, mask), reference_mha(mha, x, mask), atol=1e-5, rtol=1e-5)


def test_matches_sdpa_prefix_mask():
    torch.manual_seed(1)
    mha = MultiHeadAttention(64, 4).eval()
    x = torch.randn(2, 30, 64)
    mask = prefix_mask(20, 30)
    torch.testing.assert_close(mha(x, mask), reference_mha(mha, x, mask), atol=1e-5, rtol=1e-5)


def test_matches_sdpa_without_mask():
    torch.manual_seed(2)
    mha = MultiHeadAttention(32, 4).eval()
    x = torch.randn(2, 12, 32)
    torch.testing.assert_close(mha(x, None), reference_mha(mha, x, None), atol=1e-5, rtol=1e-5)


def test_core_function_matches_sdpa():
    torch.manual_seed(3)
    q, k, v = (torch.randn(2, 4, 16, 8) for _ in range(3))
    mask = causal_mask(16)
    out, _ = scaled_dot_product(q, k, v, mask)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-5, rtol=1e-5)


def test_weights_are_a_distribution_and_respect_the_mask():
    torch.manual_seed(4)
    mha = MultiHeadAttention(32, 4).eval()
    x = torch.randn(2, 9, 32)
    mask = causal_mask(9)
    _, weights = mha(x, mask, return_weights=True)  # (B, H, T, T)
    assert weights.shape == (2, 4, 9, 9)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 9))  # each query's weights sum to 1
    assert (weights[:, :, ~mask] == 0).all()  # nothing flows through masked positions


def test_no_information_from_the_future():
    torch.manual_seed(5)
    mha = MultiHeadAttention(32, 4).eval()
    x = torch.randn(1, 12, 32)
    x2 = x.clone()
    x2[:, 7:] = torch.randn(1, 5, 32)  # change only tokens 7..11
    mask = causal_mask(12)
    y, y2 = mha(x, mask), mha(x2, mask)
    torch.testing.assert_close(y[:, :7], y2[:, :7])  # outputs before position 7 are identical
    assert not torch.allclose(y[:, 7:], y2[:, 7:])  # sanity: the change is visible where it should be


def test_gradients_reach_all_projections():
    mha = MultiHeadAttention(32, 4)
    x = torch.randn(2, 6, 32, requires_grad=True)
    mha(x, causal_mask(6)).sum().backward()
    for name, p in mha.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
