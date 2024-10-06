"""SED2AM Vehicle Selection Decoder (paper §4.3.1).

Inputs per vehicle (5 features each, from the fleet state s_F^t):
  - remaining_capacity (rc_i^t)
  - current_location (one-hot via lookup to its embedding)
  - remaining_working_hours (τ_i^t)
  - current_time_interval (p_i^t)
  - time_remaining_in_interval (rt_i^t)

Plus the graph embedding (mean of node embeddings across all intervals).

Output: a categorical distribution over vehicles.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class VehicleSelectionDecoder(nn.Module):
    VEHICLE_FEATURE_DIM = 5

    def __init__(self, embed_dim: int = 128, n_heads: int = 8, tanh_clipping: float = 10.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.tanh_clipping = tanh_clipping
        self.vehicle_proj = nn.Linear(self.VEHICLE_FEATURE_DIM + embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.W_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_key = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        graph_embedding: Tensor,         # (B, D)  — mean over intervals + nodes
        vehicle_features: Tensor,        # (B, V, 5)
        vehicle_location_emb: Tensor,    # (B, V, D)  — node-embedding of each vehicle's current location
        mask: Tensor,                    # (B, V) True where vehicle is unavailable
    ) -> Tensor:
        b, v, _ = vehicle_features.size()
        full = torch.cat([vehicle_features, vehicle_location_emb], dim=-1)   # (B, V, 5+D)
        h = self.vehicle_proj(full)                                          # (B, V, D)

        # Attention with the graph embedding as the query.
        q = graph_embedding.unsqueeze(1)                                     # (B, 1, D)
        attended, _ = self.attn(q, h, h, key_padding_mask=mask, need_weights=False)
        q2 = self.W_query(attended)                                          # (B, 1, D)
        k = self.W_key(h)                                                    # (B, V, D)
        logits = torch.matmul(q2, k.transpose(-1, -2)).squeeze(1) / math.sqrt(self.embed_dim)
        logits = self.tanh_clipping * torch.tanh(logits)
        return logits.masked_fill(mask, float("-inf"))

# §4.3.1 ref

# mask order fix 2024-10-05
