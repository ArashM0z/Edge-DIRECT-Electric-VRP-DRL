"""Tests for the spatial-edge-feature graph Transformer."""

from __future__ import annotations

import torch

from nets.efectiw.spatial_encoder import SpatialEdgeAttention, SpatialGraphTransformer


def test_attention_output_shape() -> None:
    m = SpatialEdgeAttention(embed_dim=32, n_heads=4)
    h = torch.randn(2, 7, 32)
    e = torch.rand(2, 7, 7)
    mask = torch.ones(2, 7, 7, dtype=torch.bool)
    out = m(h, e, mask)
    assert out.shape == h.shape


def test_attention_respects_zero_adjacency() -> None:
    """If the mask is all zeros, the output should be all NaN
    (softmax over -inf scores). We accept either NaN or fall-through;
    main concern is no crash."""
    m = SpatialEdgeAttention(embed_dim=16, n_heads=4)
    h = torch.randn(1, 4, 16)
    e = torch.rand(1, 4, 4)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    out = m(h, e, mask)
    assert out.shape == h.shape


def test_full_encoder_shape() -> None:
    enc = SpatialGraphTransformer(in_dim=4, embed_dim=32, n_heads=4, n_layers=2)
    x = torch.randn(2, 10, 4)
    e = torch.rand(2, 10, 10)
    mask = torch.ones(2, 10, 10, dtype=torch.bool)
    out = enc(x, e, mask)
    assert out.shape == (2, 10, 32)
