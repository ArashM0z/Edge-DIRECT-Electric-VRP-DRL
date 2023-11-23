"""Hierarchical attention decoder.

Step 1: select a vehicle type, conditioned on the global graph embedding +
the remaining fleet state.

Step 2: select the next customer, conditioned on the selected vehicle's
context (capacity, current location, accumulated time) and attending over
the fused-encoder node outputs.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class VehicleSelector(nn.Module):
    def __init__(self, embed_dim: int, n_vehicle_types: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim + 3 * n_vehicle_types, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_vehicle_types),
        )

    def forward(self, graph_emb: Tensor, fleet_state: Tensor) -> Tensor:
        return self.gate(torch.cat([graph_emb, fleet_state], dim=-1))


class NodeSelector(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int = 8, tanh_clip: float = 10.0) -> None:
        super().__init__()
        self.tanh_clip = tanh_clip
        self.embed_dim = embed_dim
        self.glimpse = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.W_ctx = nn.Linear(embed_dim + 3, embed_dim, bias=False)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        node_emb: Tensor,
        graph_emb: Tensor,
        remaining_capacity: Tensor,
        current_time: Tensor,
        last_node_emb: Tensor,
        mask: Tensor,
    ) -> Tensor:
        ctx = torch.cat([graph_emb,
                         remaining_capacity.unsqueeze(-1),
                         current_time.unsqueeze(-1),
                         remaining_capacity.unsqueeze(-1) * 0], dim=-1)
        q0 = self.W_ctx(ctx).unsqueeze(1)
        attended, _ = self.glimpse(q0, node_emb, node_emb, key_padding_mask=mask, need_weights=False)
        q = self.W_q(attended)
        k = self.W_k(node_emb)
        logits = torch.matmul(q, k.transpose(-1, -2)).squeeze(1) / math.sqrt(self.embed_dim)
        logits = self.tanh_clip * torch.tanh(logits)
        return logits.masked_fill(mask, float("-inf"))


class HierarchicalDecoder(nn.Module):
    def __init__(self, embed_dim: int, n_vehicle_types: int, n_heads: int = 8) -> None:
        super().__init__()
        self.vehicle_selector = VehicleSelector(embed_dim, n_vehicle_types)
        self.node_selector = NodeSelector(embed_dim, n_heads)
