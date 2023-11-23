"""EFECTIW-ROTER full agent.

Spatial encoder + Temporal encoder -> Fusion encoder -> Hierarchical decoder.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from nets.efectiw.fusion_encoder import MultigraphFusionEncoder
from nets.efectiw.hierarchical_decoder import HierarchicalDecoder
from nets.efectiw.spatial_encoder import SpatialGraphTransformer
from nets.efectiw.temporal_encoder import TemporalGraphTransformer
from problems.hf_vrptw.graphs import build_dv_graph, build_tw_graph
from problems.hf_vrptw.spectral_embedding import spectral_embedding


class EFECTIWROTER(nn.Module):
    NODE_INPUT_DIM = 4   # (x, y, demand, tw_window_length)

    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 3,
        n_vehicle_types: int = 4,
        k_spectral: int = 8,
    ) -> None:
        super().__init__()
        self.spatial = SpatialGraphTransformer(
            in_dim=self.NODE_INPUT_DIM, embed_dim=embed_dim,
            n_heads=n_heads, n_layers=n_encoder_layers,
        )
        self.temporal = TemporalGraphTransformer(
            in_dim=self.NODE_INPUT_DIM, embed_dim=embed_dim,
            n_heads=n_heads, n_layers=n_encoder_layers,
        )
        self.fusion = MultigraphFusionEncoder(embed_dim, k_spectral)
        self.decoder = HierarchicalDecoder(embed_dim, n_vehicle_types, n_heads)
        self.k_spectral = k_spectral
        self.n_vehicle_types = n_vehicle_types

    def encode(self, state) -> tuple[Tensor, Tensor]:
        coords = state.loc
        demand = state.demand
        tw_start = state.tw_start
        tw_end = state.tw_end

        # Node features: (x, y, demand, window-length)
        tw_len = (tw_end - tw_start).unsqueeze(-1)
        x = torch.cat([coords, demand.unsqueeze(-1), tw_len], dim=-1)

        # Edge feature: pairwise travel time (1 km/min for simplicity)
        edge_t = (coords.unsqueeze(-2) - coords.unsqueeze(-3)).norm(dim=-1)

        # Two graphs
        g_tw = build_tw_graph(coords, tw_start, tw_end)
        vehicle_caps = state.vehicle_capacities
        g_dv = build_dv_graph(demand, vehicle_caps)
        g_union = g_tw | g_dv

        h_spatial = self.spatial(x, edge_t, g_tw)
        h_temporal = self.temporal(x, tw_start, g_dv)
        spec = spectral_embedding(g_union.float(), k=self.k_spectral)

        node_emb = self.fusion(h_spatial, h_temporal, spec)
        graph_emb = node_emb.mean(dim=1)
        return node_emb, graph_emb
