"""SED2AM Trip Construction Decoder (paper §4.3.2).

Given a selected vehicle, picks the next location (depot or customer) to
visit. Attention is over the **per-interval** node embeddings, using the
chosen vehicle's *current* time interval to pick the right encoder output.

The context vector is built from:
  - graph embedding for the selected interval
  - last-visited node embedding for the selected vehicle
  - remaining-capacity + remaining-hours scalars for that vehicle
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TripConstructionDecoder(nn.Module):
    def __init__(self, embed_dim: int = 128, n_heads: int = 8, tanh_clipping: float = 10.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.tanh_clipping = tanh_clipping
        # Context = [graph_emb, last_node_emb, rc, τ_remaining]
        self.context_proj = nn.Linear(embed_dim * 2 + 2, embed_dim, bias=False)
        self.mha_glimpse = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.W_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_key = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        node_embeddings_for_interval: Tensor,   # (B, N, D) — encoder output at chosen interval
        graph_embedding_for_interval: Tensor,   # (B, D)
        last_node_embedding: Tensor,            # (B, D)
        remaining_capacity: Tensor,             # (B,)
        remaining_hours: Tensor,                # (B,)
        feasibility_mask: Tensor,               # (B, N) True where infeasible
    ) -> Tensor:
        b = node_embeddings_for_interval.size(0)
        context = torch.cat([
            graph_embedding_for_interval,
            last_node_embedding,
            remaining_capacity.unsqueeze(-1),
            remaining_hours.unsqueeze(-1),
        ], dim=-1)                                # (B, 2D+2)
        q0 = self.context_proj(context).unsqueeze(1)   # (B, 1, D)

        glimpse, _ = self.mha_glimpse(
            q0, node_embeddings_for_interval, node_embeddings_for_interval,
            key_padding_mask=feasibility_mask, need_weights=False,
        )
        q = self.W_query(glimpse)
        k = self.W_key(node_embeddings_for_interval)
        logits = torch.matmul(q, k.transpose(-1, -2)).squeeze(1) / math.sqrt(self.embed_dim)
        logits = self.tanh_clipping * torch.tanh(logits)
        return logits.masked_fill(feasibility_mask, float("-inf"))
