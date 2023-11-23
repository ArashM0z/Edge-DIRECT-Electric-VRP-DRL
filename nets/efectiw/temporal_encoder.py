"""Temporal-positional-embedding graph Transformer encoder (EFECTIW §4.2).

Customers are linearly ordered by tw_start. The encoder adds **learnable
temporal positional embeddings** to each customer's representation based on
its rank in that ordering, then runs self-attention over the
demand-vehicle-compatibility graph. This captures TW *sequencing*
information that the spatial encoder doesn't see.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class LearnableTemporalPositionalEmbedding(nn.Module):
    def __init__(self, max_positions: int, embed_dim: int) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_positions, embed_dim)

    def forward(self, tw_start: Tensor) -> Tensor:
        """tw_start: (B, N) -> (B, N, D) positional embeddings."""
        # Argsort returns the index of each customer in the time-ordered sequence.
        order = tw_start.argsort(dim=-1)
        rank = torch.empty_like(order)
        rank.scatter_(-1, order, torch.arange(order.size(-1), device=order.device).expand_as(order))
        rank = rank.clamp(max=self.pe.num_embeddings - 1)
        return self.pe(rank)


class TemporalEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, ff_hidden: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, h: Tensor, adj_mask: Tensor) -> Tensor:
        key_padding_mask = None  # not used; we mask via attn_mask below
        attn_mask = (~adj_mask).float() * -1e9
        # Repeat per head — PyTorch MHA expects (B*H, N, N)
        b, n, _ = h.size()
        h_heads = h
        attended, _ = self.attn(h_heads, h_heads, h_heads, attn_mask=attn_mask.repeat_interleave(
            self.attn.num_heads, dim=0,
        ).view(b * self.attn.num_heads, n, n))
        h = self.norm1(h + attended)
        h = self.norm2(h + self.ff(h))
        return h


class TemporalGraphTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        ff_hidden: int = 512,
        max_positions: int = 256,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.positional = LearnableTemporalPositionalEmbedding(max_positions, embed_dim)
        self.layers = nn.ModuleList(
            [TemporalEncoderLayer(embed_dim, n_heads, ff_hidden) for _ in range(n_layers)]
        )

    def forward(self, x: Tensor, tw_start: Tensor, adj_mask: Tensor) -> Tensor:
        h = self.input_proj(x) + self.positional(tw_start)
        for layer in self.layers:
            h = layer(h, adj_mask)
        return h
