"""Per-depot actor head: one decoder per depot's set of vehicles.

Each actor shares the shared encoder's output but has its own pointer
attention parameters. This lets the actors specialise to their depot's
geographic neighbourhood without losing global context.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PerDepotActor(nn.Module):
    def __init__(self, embed_dim: int = 128, n_heads: int = 8, tanh_clip: float = 10.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.tanh_clip = tanh_clip
        # context: depot embedding + last-node embedding + remaining capacity
        self.W_ctx = nn.Linear(embed_dim * 2 + 1, embed_dim, bias=False)
        self.glimpse = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        node_emb: Tensor,
        depot_emb: Tensor,
        last_node_emb: Tensor,
        remaining_capacity: Tensor,
        mask: Tensor,
    ) -> Tensor:
        ctx = torch.cat([depot_emb, last_node_emb, remaining_capacity.unsqueeze(-1)], dim=-1)
        q0 = self.W_ctx(ctx).unsqueeze(1)
        attended, _ = self.glimpse(q0, node_emb, node_emb, key_padding_mask=mask, need_weights=False)
        q = self.W_q(attended)
        k = self.W_k(node_emb)
        logits = (q @ k.transpose(-1, -2)).squeeze(1) / math.sqrt(self.embed_dim)
        logits = self.tanh_clip * torch.tanh(logits)
        return logits.masked_fill(mask, float("-inf"))
