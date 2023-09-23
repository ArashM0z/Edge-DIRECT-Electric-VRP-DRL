"""Edge-DIRECT full agent.

Edge-enhanced dual-attention encoder -> feature-enhanced dual-attention decoder.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from nets.edge_direct.dual_attention_decoder import FeatureEnhancedDualDecoder
from nets.edge_direct.dual_attention_encoder import DualAttentionEncoder
from problems.evrptw.graphs import build_tw_overlap_graph


class EdgeDIRECT(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 3,
        n_vehicle_types: int = 3,
        battery_kwh: float = 60.0,
        kwh_per_km: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = DualAttentionEncoder(
            embed_dim=embed_dim, n_heads=n_heads, n_layers=n_encoder_layers,
        )
        self.decoder = FeatureEnhancedDualDecoder(embed_dim, n_heads, n_vehicle_types)
        self.battery_kwh = battery_kwh
        self.kwh_per_km = kwh_per_km

    def encode(self, state) -> tuple[Tensor, Tensor]:
        coords = state.loc
        demand = state.demand
        tw_start = state.tw_start
        tw_end = state.tw_end

        x = torch.cat([
            coords,
            demand.unsqueeze(-1),
            tw_start.unsqueeze(-1),
            tw_end.unsqueeze(-1),
        ], dim=-1)

        travel_time = (coords.unsqueeze(-2) - coords.unsqueeze(-3)).norm(dim=-1)
        energy = travel_time * self.kwh_per_km
        adj = build_tw_overlap_graph(tw_start, tw_end)

        node_emb = self.encoder(x, travel_time, energy, adj)
        return node_emb, node_emb.mean(dim=1)
