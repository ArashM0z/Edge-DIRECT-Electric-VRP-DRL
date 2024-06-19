"""Tests for the learnable temporal positional embedding + temporal encoder."""

from __future__ import annotations

import torch

from nets.efectiw.temporal_encoder import (
    LearnableTemporalPositionalEmbedding,
    TemporalGraphTransformer,
)


def test_positional_embedding_shape() -> None:
    pe = LearnableTemporalPositionalEmbedding(max_positions=64, embed_dim=16)
    tw_start = torch.rand(2, 10) * 100
    out = pe(tw_start)
    assert out.shape == (2, 10, 16)


def test_positional_embedding_orders_by_tw_start() -> None:
    """Customers with the same rank in tw_start should get the same embedding."""
    pe = LearnableTemporalPositionalEmbedding(max_positions=10, embed_dim=8)
    # Two batches, second is a permutation of the first
    tw_a = torch.tensor([[10.0, 20.0, 30.0]])
    tw_b = torch.tensor([[30.0, 10.0, 20.0]])   # same ranks: [2, 0, 1]
    out_a = pe(tw_a)
    out_b = pe(tw_b)
    # rank-0 customer in A is index 0; rank-0 customer in B is index 1
    assert torch.allclose(out_a[0, 0], out_b[0, 1])


def test_temporal_transformer_shape() -> None:
    enc = TemporalGraphTransformer(in_dim=4, embed_dim=32, n_heads=4, n_layers=2)
    x = torch.randn(2, 8, 4)
    tw_s = torch.rand(2, 8) * 100
    mask = torch.ones(2, 8, 8, dtype=torch.bool)
    out = enc(x, tw_s, mask)
    assert out.shape == (2, 8, 32)
