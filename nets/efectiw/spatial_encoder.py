"""Spatial Edge-Feature Enhanced graph Transformer encoder (EFECTIW §4.1).

The attention score between nodes i and j is augmented by a learned bias
derived from the edge feature ε_ij = travel_time(i, j). The encoder
operates on the **TW-feasibility** adjacency, masking unreachable pairs.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SpatialEdgeAttention(nn.Module):
    """Multi-head attention with additive edge bias from travel-time edges."""

    def __init__(self, embed_dim: int, n_heads: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_e = nn.Linear(1, n_heads, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, h: Tensor, edge_feat: Tensor, adj_mask: Tensor) -> Tensor:
        b, n, _ = h.size()
        q = self.W_q(h).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(h).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(h).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)

        edge_bias = self.W_e(edge_feat.unsqueeze(-1)).permute(0, 3, 1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim) + edge_bias
        scores = scores.masked_fill(~adj_mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, n, self.embed_dim)
        return self.W_o(out)


class SpatialEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, ff_hidden: int) -> None:
        super().__init__()
        self.attn = SpatialEdgeAttention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, h: Tensor, edge_feat: Tensor, adj_mask: Tensor) -> Tensor:
        h = self.norm1(h + self.attn(h, edge_feat, adj_mask))
        h = self.norm2(h + self.ff(h))
        return h


class SpatialGraphTransformer(nn.Module):
    """Stacks SpatialEncoderLayer over the TW-feasibility graph."""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        ff_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.layers = nn.ModuleList(
            [SpatialEncoderLayer(embed_dim, n_heads, ff_hidden) for _ in range(n_layers)]
        )

    def forward(self, x: Tensor, edge_feat: Tensor, adj_mask: Tensor) -> Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_feat, adj_mask)
        return h
