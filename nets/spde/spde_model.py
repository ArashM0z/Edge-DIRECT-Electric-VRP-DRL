"""SP-DE full agent: shared encoder + per-depot actors + centralised critic."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from nets.spde.centralised_critic import CentralisedAttentionCritic
from nets.spde.per_depot_actor import PerDepotActor
from nets.spde.shared_encoder import SharedEncoder


class SPDE(nn.Module):
    def __init__(
        self,
        n_depots: int = 3,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = SharedEncoder(embed_dim, n_heads, n_encoder_layers)
        self.actors = nn.ModuleList(
            [PerDepotActor(embed_dim, n_heads) for _ in range(n_depots)]
        )
        self.critic = CentralisedAttentionCritic(embed_dim, n_heads)
        self.n_depots = n_depots

    def actor_logits(
        self, state, vehicle_idx: int, depot_idx: int, node_emb: Tensor,
    ) -> Tensor:
        """Compute logits for the next location of a specific vehicle."""
        b = node_emb.size(0)
        depot_emb = node_emb[:, depot_idx]
        cur_loc = state.vehicle_loc[:, vehicle_idx]
        last_node_emb = node_emb.gather(
            1, cur_loc.view(b, 1, 1).expand(b, 1, node_emb.size(-1))
        ).squeeze(1)
        rc = state.vehicle_rc[:, vehicle_idx]
        # Mask: visited customers + over-capacity
        n_dep = state.n_depots()
        n_cust = state.loc.size(1)
        visited_full = torch.cat([
            torch.zeros(b, n_dep, dtype=torch.bool, device=node_emb.device),
            state.visited_,
        ], dim=-1)
        demand_full = torch.cat([
            torch.zeros(b, n_dep, device=node_emb.device),
            state.demand,
        ], dim=-1)
        over_cap = demand_full > rc.unsqueeze(-1)
        mask = visited_full | over_cap
        return self.actors[depot_idx](node_emb, depot_emb, last_node_emb, rc, mask)

    def value(self, state, node_emb: Tensor) -> Tensor:
        """Centralised value over all vehicles' current contexts."""
        b = node_emb.size(0)
        v = state.vehicle_loc.size(1)
        # Gather each vehicle's current-location embedding
        loc_emb = node_emb.gather(
            1, state.vehicle_loc.unsqueeze(-1).expand(b, v, node_emb.size(-1))
        )
        return self.critic(loc_emb)
