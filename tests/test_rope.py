"""Numerical checks for RotaryEmbedding frequency layout and rotation."""

import math

import torch

from gpt_bpe import RotaryEmbedding


def test_adjacent_pairs_share_frequency():
    dim = 8
    rope = RotaryEmbedding(dim=dim, max_seq_len=16, base=10000)
    for t in range(16):
        for i in range(0, dim, 2):
            assert torch.allclose(rope.cos[t, i], rope.cos[t, i + 1])
            assert torch.allclose(rope.sin[t, i], rope.sin[t, i + 1])


def test_frequencies_are_not_half_duplicated():
    """Regression: cat([freqs, freqs]) made dim/2 and dim/2+i share a freq incorrectly."""
    dim = 8
    rope = RotaryEmbedding(dim=dim, max_seq_len=4, base=10000)
    # With interleaved freqs, cos[:, 0] should equal cos[:, 1], not cos[:, dim//2].
    assert torch.allclose(rope.cos[:, 0], rope.cos[:, 1])
    assert not torch.allclose(rope.cos[:, 0], rope.cos[:, dim // 2])


def test_rope_matches_closed_form_rotation():
    dim = 4
    base = 10000.0
    seq_len = 5
    rope = RotaryEmbedding(dim=dim, max_seq_len=8, base=base)
    x = torch.randn(2, seq_len, dim)
    out = rope(x)

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    for b in range(2):
        for t in range(seq_len):
            for i in range(dim // 2):
                angle = float(t * inv_freq[i])
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                x0 = x[b, t, 2 * i]
                x1 = x[b, t, 2 * i + 1]
                expected0 = x0 * cos_a - x1 * sin_a
                expected1 = x1 * cos_a + x0 * sin_a
                assert torch.allclose(out[b, t, 2 * i], expected0, atol=1e-5)
                assert torch.allclose(out[b, t, 2 * i + 1], expected1, atol=1e-5)
