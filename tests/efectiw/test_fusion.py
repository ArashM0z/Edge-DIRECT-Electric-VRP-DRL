"""Tests for the multigraph fusion encoder."""

from __future__ import annotations

import torch

from nets.efectiw.fusion_encoder import MultigraphFusionEncoder


def test_fusion_output_shape() -> None:
    f = MultigraphFusionEncoder(embed_dim=32, k_spectral=4)
    h_s = torch.randn(2, 10, 32)
    h_t = torch.randn(2, 10, 32)
    spec = torch.randn(2, 10, 4)
    out = f(h_s, h_t, spec)
    assert out.shape == (2, 10, 32)


def test_fusion_is_differentiable() -> None:
    f = MultigraphFusionEncoder(embed_dim=16, k_spectral=3)
    h_s = torch.randn(1, 5, 16, requires_grad=True)
    h_t = torch.randn(1, 5, 16)
    spec = torch.randn(1, 5, 3)
    out = f(h_s, h_t, spec).sum()
    out.backward()
    assert h_s.grad is not None
