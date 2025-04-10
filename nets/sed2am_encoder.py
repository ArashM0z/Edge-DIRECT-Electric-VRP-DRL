"""SED2AM Simultaneous Encoder with temporal-locality inductive bias.

Per the paper (Mozhdehi et al., ACM TKDD 2025, §4.2):
    The encoder embeds nodes and edges *per time interval*. For each interval
    p in 1..P (where P partitions the working day, typically P=5 across an
    8h shift), we maintain a separate set of node embeddings h_{v_i,p} and
    edge embeddings h_{e_ij,p}. Attention scores include the edge information
    as both an additive bias and a sigmoid gate, so traffic conditions
    modulate the attention without inflating the parameter count.

Equations (from the paper):
    h^0_{v_i,p} = W^χ χ_{v_i} + b^χ
    h^0_{e_ij,p} = W^ε ε_{ij,p} + b^ε
    u^{l,m}_{ij,p} = (q^{l,m}_{i,p})^T k^{l,m}_{j,p} / √d_k + ε_{ji,p}
    a^{l,m}_{ij,p} = softmax(u_{ij,p}) × σ(ε_{ji,p})
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TemporalMultiHeadAttention(nn.Module):
    """Multi-head attention with additive edge bias and sigmoid edge gate.

    Implements equations 10-12 in the paper.
    """

    def __init__(self, embed_dim: int, n_heads: int) -> None:
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.W_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_edge_bias = nn.Linear(embed_dim, n_heads, bias=False)  # ε_{ji,p} -> head-wise scalar
        self.W_edge_gate = nn.Linear(embed_dim, n_heads, bias=False)
        self.W_out = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, h_nodes: Tensor, h_edges: Tensor) -> Tensor:
        """h_nodes: (B, N, D)  |  h_edges: (B, N, N, D)  ->  (B, N, D)."""
        b, n, _ = h_nodes.size()

        q = self.W_query(h_nodes).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(h_nodes).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(h_nodes).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)

        # Edge bias and gate — head-wise scalars from per-edge embedding.
        edge_bias = self.W_edge_bias(h_edges).permute(0, 3, 1, 2)
        edge_gate = torch.sigmoid(self.W_edge_gate(h_edges).permute(0, 3, 1, 2))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim) + edge_bias
        attn = torch.softmax(scores, dim=-1) * edge_gate  # eq. 12

        out = torch.matmul(attn, v)                       # (B, H, N, D_h)
        out = out.transpose(1, 2).contiguous().view(b, n, self.embed_dim)
        return self.W_out(out)


class SED2AMEncoderLayer(nn.Module):
    """One layer of the SED2AM simultaneous encoder.

    Gating mechanism (eqs 13-15) stabilises training:
      g_{v_i,p} = z * MHA(h) + (1-z) * h
      where z = σ(W_z [MHA(h), h])
    """

    def __init__(self, embed_dim: int, n_heads: int, ff_hidden_dim: int) -> None:
        super().__init__()
        self.mha = TemporalMultiHeadAttention(embed_dim, n_heads)
        self.W_gate_attn = nn.Linear(2 * embed_dim, embed_dim)
        self.bn_attn = nn.BatchNorm1d(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, embed_dim),
        )
        self.W_gate_ff = nn.Linear(2 * embed_dim, embed_dim)
        self.bn_ff = nn.BatchNorm1d(embed_dim)

    def forward(self, h_nodes: Tensor, h_edges: Tensor) -> Tensor:
        attended = self.mha(h_nodes, h_edges)
        gate_a = torch.sigmoid(self.W_gate_attn(torch.cat([attended, h_nodes], dim=-1)))
        g = gate_a * attended + (1 - gate_a) * h_nodes
        g = self.bn_attn(g.transpose(1, 2)).transpose(1, 2)

        ff_out = self.ff(g)
        gate_b = torch.sigmoid(self.W_gate_ff(torch.cat([ff_out, g], dim=-1)))
        out = gate_b * ff_out + (1 - gate_b) * g
        return self.bn_ff(out.transpose(1, 2)).transpose(1, 2)


class SED2AMSimultaneousEncoder(nn.Module):
    """Encodes the graph independently per time interval p ∈ {0, ..., P-1}.

    For an 8h working day partitioned into P=5 intervals, the encoder runs P
    independent forward passes — one per interval's edge tensor — sharing
    the same weights across intervals. This is the **temporal locality
    inductive bias**: routing decisions depend on traffic at the *current*
    departure interval, not an averaged daily condition.

    Returns per-interval node embeddings (B, P, N, D).
    """

    NODE_FEATURE_DIM = 3   # (x, y, demand)

    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        ff_hidden_dim: int = 512,
        n_intervals: int = 5,
    ) -> None:
        super().__init__()
        self.n_intervals = n_intervals
        self.embed_dim = embed_dim
        self.node_proj = nn.Linear(self.NODE_FEATURE_DIM, embed_dim)
        self.edge_proj = nn.Linear(1, embed_dim)  # scalar travel time -> D-dim
        self.layers = nn.ModuleList(
            [SED2AMEncoderLayer(embed_dim, n_heads, ff_hidden_dim) for _ in range(n_layers)]
        )

    def forward(self, node_features: Tensor, edge_travel_times: Tensor) -> Tensor:
        """node_features: (B, N, 3) — depot at index 0.
        edge_travel_times: (B, P, N, N) — travel time per interval.

        Returns: (B, P, N, D) per-interval node embeddings.
        """
        b, p, n, _ = edge_travel_times.size()
        per_interval: list[Tensor] = []
        h_node_0 = self.node_proj(node_features)             # (B, N, D)
        for interval in range(p):
            h_edge = self.edge_proj(edge_travel_times[:, interval].unsqueeze(-1))  # (B, N, N, D)
            h = h_node_0
            for layer in self.layers:
                h = layer(h, h_edge)
            per_interval.append(h)
        return torch.stack(per_interval, dim=1)  # (B, P, N, D)

# encoder revised 2024-09-15

# self-loop guard added 2024-09-18

# §4.2 annotations 2024-09-22

# preallocation note 2024-09-25

# edge proj note 2024-09-29

# line tidy 2025-04-09
