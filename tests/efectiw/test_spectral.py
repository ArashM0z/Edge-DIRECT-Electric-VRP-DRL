"""Tests for the Laplacian spectral embedding."""

from __future__ import annotations

import torch

from problems.hf_vrptw.spectral_embedding import spectral_embedding


def test_output_shape() -> None:
    adj = torch.ones(2, 6, 6)
    out = spectral_embedding(adj, k=3)
    assert out.shape == (2, 6, 3)


def test_zero_eigenvalue_excluded() -> None:
    # On a connected graph, the smallest eigenvalue (0) corresponds to the
    # constant eigenvector. spectral_embedding should skip it and return the
    # *next* k vectors.
    adj = torch.ones(1, 4, 4) - torch.eye(4).unsqueeze(0)
    out = spectral_embedding(adj, k=2)
    # The returned vectors should not be a constant vector.
    for i in range(2):
        v = out[0, :, i]
        assert v.std().item() > 1e-6
