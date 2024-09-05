"""Shared Transformer encoder over the heterogeneous (depot + customer) graph.

All vehicles, regardless of depot, share this encoder. Per-depot actor
heads then attend over its output to construct routes from their assigned
depot. This is the standard CTDE (centralised training, decentralised
execution) pattern in multi-agent RL.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SharedEncoder(nn.Module):
    NODE_DIM = 3  # (x, y, demand-or-zero-for-depots)

    def __init__(self, embed_dim: int = 128, n_heads: int = 8, n_layers: int = 3) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            embed_dim, n_heads, embed_dim * 4, batch_first=True, norm_first=True,
        )
        self.input_proj = nn.Linear(self.NODE_DIM, embed_dim)
        self.encoder = nn.TransformerEncoder(layer, n_layers)

    def forward(self, state) -> Tensor:
        coords = state.all_nodes_coords()
        depot_demand = torch.zeros(coords.size(0), state.n_depots(), device=coords.device)
        all_demand = torch.cat([depot_demand, state.demand], dim=-1).unsqueeze(-1)
        x = torch.cat([coords, all_demand], dim=-1)
        return self.encoder(self.input_proj(x))
