"""Feature-enhanced dual-attention decoder.

Two-stage:
  1. Vehicle-type head: chooses which heterogeneous EV continues.
  2. Node head: chooses the next customer / charging station, attending
     over the encoder output with the chosen vehicle's energy / capacity
     context.

The "feature-enhanced" name comes from the decoder's vehicle-conditioned
context vector including the vehicle's residual energy and capacity in
addition to the standard "last node" and graph embeddings.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class VehicleTypeHead(nn.Module):
    def __init__(self, embed_dim: int, n_vehicle_types: int) -> None:
        super().__init__()
        # input: graph_emb + per-type (remaining-capacity, soc, count)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + 3 * n_vehicle_types, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_vehicle_types),
        )

    def forward(self, graph_emb: Tensor, fleet_features: Tensor) -> Tensor:
        return self.net(torch.cat([graph_emb, fleet_features], dim=-1))


class NodeHead(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, tanh_clip: float = 10.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.tanh_clip = tanh_clip
        self.W_ctx = nn.Linear(embed_dim + 4, embed_dim, bias=False)
        self.glimpse = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        node_emb: Tensor,
        graph_emb: Tensor,
        soc: Tensor,
        remaining_capacity: Tensor,
        current_time: Tensor,
        last_node_emb: Tensor,
        feasibility_mask: Tensor,
    ) -> Tensor:
        ctx = torch.cat([
            graph_emb,
            soc.unsqueeze(-1),
            remaining_capacity.unsqueeze(-1),
            current_time.unsqueeze(-1),
            (soc * remaining_capacity).unsqueeze(-1),  # interaction feature
        ], dim=-1)
        q0 = self.W_ctx(ctx).unsqueeze(1)
        attended, _ = self.glimpse(
            q0, node_emb, node_emb, key_padding_mask=feasibility_mask, need_weights=False,
        )
        q = self.W_q(attended)
        k = self.W_k(node_emb)
        logits = (q @ k.transpose(-1, -2)).squeeze(1) / math.sqrt(self.embed_dim)
        logits = self.tanh_clip * torch.tanh(logits)
        return logits.masked_fill(feasibility_mask, float("-inf"))


class FeatureEnhancedDualDecoder(nn.Module):
    def __init__(
        self, embed_dim: int = 128, n_heads: int = 8, n_vehicle_types: int = 3,
    ) -> None:
        super().__init__()
        self.vehicle_head = VehicleTypeHead(embed_dim, n_vehicle_types)
        self.node_head = NodeHead(embed_dim, n_heads)
