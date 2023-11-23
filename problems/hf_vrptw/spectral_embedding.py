"""Spectral embedding: bottom-k Laplacian eigenvectors of the fused graph.

Adds node features that capture *global* graph structure (community-like
positions) — a standard trick from spectral GNNs that helps the encoder
generalise across instances of varying connectivity.

Used by EFECTIW-ROTER as the input feature on top of (x, y, demand, tw).
"""

from __future__ import annotations

import torch
from torch import Tensor


def spectral_embedding(adj: Tensor, k: int = 8) -> Tensor:
    """Return the bottom-k non-trivial eigenvectors of the normalised Laplacian.

    Args:
        adj: (B, N, N) boolean / float adjacency.
        k: number of eigenvectors to return (excluding the trivial constant).

    Returns:
        (B, N, k) per-node spectral features.
    """
    adj = adj.float()
    deg = adj.sum(dim=-1).clamp(min=1.0)
    d_inv_sqrt = deg.pow(-0.5)
    # L_sym = I - D^{-1/2} A D^{-1/2}
    norm_adj = d_inv_sqrt.unsqueeze(-1) * adj * d_inv_sqrt.unsqueeze(-2)
    n = adj.size(-1)
    identity = torch.eye(n, device=adj.device).expand_as(adj)
    laplacian = identity - norm_adj

    eigvals, eigvecs = torch.linalg.eigh(laplacian)
    # Drop the trivial 0-eigenvalue eigenvector at index 0; keep the next k.
    return eigvecs[..., 1 : k + 1]
