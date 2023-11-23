"""Multigraph fusion encoder: combines the spatial + temporal encoders'
outputs with the spectral embedding of the union graph.

Output: per-node embeddings that capture both spatial-edge and
temporal-positional structure.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MultigraphFusionEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128, k_spectral: int = 8) -> None:
        super().__init__()
        self.k_spectral = k_spectral
        self.spectral_proj = nn.Linear(k_spectral, embed_dim)
        # Fusion: gated combination of spatial / temporal / spectral.
        self.gate = nn.Linear(3 * embed_dim, 3)
        self.proj = nn.Linear(3 * embed_dim, embed_dim)

    def forward(
        self,
        h_spatial: Tensor,
        h_temporal: Tensor,
        spectral_embedding: Tensor,
    ) -> Tensor:
        h_spec = self.spectral_proj(spectral_embedding)
        cat = torch.cat([h_spatial, h_temporal, h_spec], dim=-1)
        weights = torch.softmax(self.gate(cat), dim=-1)
        weighted = (
            weights[..., 0:1] * h_spatial
            + weights[..., 1:2] * h_temporal
            + weights[..., 2:3] * h_spec
        )
        return self.proj(torch.cat([weighted, h_spatial, h_spec], dim=-1)[:, :, : 3 * h_spatial.size(-1)])
