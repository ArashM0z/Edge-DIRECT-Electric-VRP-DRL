"""Edge-enhanced Dual Attention Encoder.

The encoder attends over **two** parallel attention heads per layer:
  - Spatial head: attention biased by **travel time** edges.
  - Energy head: attention biased by **energy consumption** edges (which
    depend on distance, slope/elevation, and the vehicle type's
    kWh-per-km factor).

The two head outputs are merged via a learnable gate before the FFN.
Operates on the TW-overlap adjacency.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class EdgeBiasedAttention(nn.Module):
    """Multi-head attention with one edge-feature bias term."""

    def __init__(self, embed_dim: int, n_heads: int) -> None:
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim, self.n_heads = embed_dim, n_heads
        self.head_dim = embed_dim // n_heads
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_e = nn.Linear(1, n_heads, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, h: Tensor, edge: Tensor, adj_mask: Tensor) -> Tensor:
        b, n, _ = h.size()
        q = self.W_q(h).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(h).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(h).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)

        edge_bias = self.W_e(edge.unsqueeze(-1)).permute(0, 3, 1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim) + edge_bias
        scores = scores.masked_fill(~adj_mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, n, self.embed_dim)
        return self.W_o(out)


class DualAttentionLayer(nn.Module):
    """One layer: spatial-edge attention + energy-edge attention, gated and merged."""

    def __init__(self, embed_dim: int, n_heads: int, ff_hidden: int) -> None:
        super().__init__()
        self.spatial = EdgeBiasedAttention(embed_dim, n_heads)
        self.energy = EdgeBiasedAttention(embed_dim, n_heads)
        self.gate = nn.Linear(2 * embed_dim, 2)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self, h: Tensor, travel_time: Tensor, energy: Tensor, adj_mask: Tensor,
    ) -> Tensor:
        h_s = self.spatial(h, travel_time, adj_mask)
        h_e = self.energy(h, energy, adj_mask)
        gate_w = torch.softmax(self.gate(torch.cat([h_s, h_e], dim=-1)), dim=-1)
        merged = gate_w[..., 0:1] * h_s + gate_w[..., 1:2] * h_e
        h = self.norm1(h + merged)
        h = self.norm2(h + self.ff(h))
        return h


class DualAttentionEncoder(nn.Module):
    NODE_INPUT_DIM = 5    # (x, y, demand, tw_start, tw_end)

    def __init__(
        self, embed_dim: int = 128, n_heads: int = 8, n_layers: int = 3, ff_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(self.NODE_INPUT_DIM, embed_dim)
        self.layers = nn.ModuleList(
            [DualAttentionLayer(embed_dim, n_heads, ff_hidden) for _ in range(n_layers)]
        )

    def forward(
        self, x: Tensor, travel_time: Tensor, energy: Tensor, adj_mask: Tensor,
    ) -> Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, travel_time, energy, adj_mask)
        return h
