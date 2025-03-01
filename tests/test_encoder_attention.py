"""Attention math sanity checks for the SED2AM encoder."""

from __future__ import annotations

import torch

from nets.sed2am_encoder import (
    SED2AMEncoderLayer,
    SED2AMSimultaneousEncoder,
    TemporalMultiHeadAttention,
)


def test_mha_output_shape() -> None:
    m = TemporalMultiHeadAttention(embed_dim=32, n_heads=4)
    h = torch.randn(2, 5, 32)
    e = torch.randn(2, 5, 5, 32)
    out = m(h, e)
    assert out.shape == (2, 5, 32)


def test_encoder_layer_preserves_dim() -> None:
    layer = SED2AMEncoderLayer(embed_dim=32, n_heads=4, ff_hidden_dim=64)
    h = torch.randn(4, 7, 32)
    e = torch.randn(4, 7, 7, 32)
    out = layer(h, e)
    assert out.shape == h.shape


def test_per_interval_encoder_is_permutation_equivariant() -> None:
    """Nodes are unordered — shuffling them shuffles the output the same way."""
    torch.manual_seed(0)
    enc = SED2AMSimultaneousEncoder(embed_dim=16, n_heads=4, n_layers=2, n_intervals=2).eval()
    n = 6
    perm = torch.randperm(n)
    x = torch.randn(1, n, 3)
    edges = torch.rand(1, 2, n, n)

    with torch.no_grad():
        out_a = enc(x, edges)
        out_b = enc(x[:, perm], edges[:, :, perm][:, :, :, perm])
    assert torch.allclose(out_a[:, :, perm], out_b, atol=1e-4)

# 3.12 tolerance 2025-02-28
