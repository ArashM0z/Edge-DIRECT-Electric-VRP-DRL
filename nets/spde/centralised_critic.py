"""Centralised attention-based critic over joint agent embeddings.

Per the paper's CTDE setup, the critic sees ALL agents' contexts and
emits one scalar value per environment used in the MAPPO update. The
self-attention layer lets the critic reason about coordination — e.g.,
identifying that two agents heading to the same cluster is suboptimal.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CentralisedAttentionCritic(nn.Module):
    def __init__(self, embed_dim: int = 128, n_heads: int = 8) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_embeddings: Tensor) -> Tensor:
        """agent_embeddings: (B, V, D) -> (B,) joint value."""
        h, _ = self.attn(agent_embeddings, agent_embeddings, agent_embeddings)
        return self.head(h.mean(dim=1)).squeeze(-1)
