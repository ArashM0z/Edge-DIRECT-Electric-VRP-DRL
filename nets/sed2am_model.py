"""SED2AM full agent: simultaneous encoder + vehicle selection + trip
construction decoders. Composes the three modules from §4.

Action at step t is a 2-tuple a^t = (i^t, v^t) — vehicle index, next location.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from nets.sed2am_encoder import SED2AMSimultaneousEncoder
from nets.trip_construction_decoder import TripConstructionDecoder
from nets.vehicle_selection_decoder import VehicleSelectionDecoder


class SED2AM(nn.Module):
    """Simultaneous Encoder Dual Decoder Attention Model."""

    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 3,
        n_intervals: int = 5,
        n_vehicles: int = 5,
        tanh_clipping: float = 10.0,
    ) -> None:
        super().__init__()
        self.encoder = SED2AMSimultaneousEncoder(
            embed_dim=embed_dim, n_heads=n_heads, n_layers=n_encoder_layers,
            n_intervals=n_intervals,
        )
        self.vehicle_decoder = VehicleSelectionDecoder(embed_dim, n_heads, tanh_clipping)
        self.trip_decoder = TripConstructionDecoder(embed_dim, n_heads, tanh_clipping)
        self.n_intervals = n_intervals
        self.n_vehicles = n_vehicles

    def forward(self, state, decode_mode: str = "sample") -> tuple[Tensor, Tensor, Tensor]:
        """Roll out a full episode. Returns (cost, log_prob, actions).

        Actions are stored as a long tensor of shape (B, T, 2) where the
        last dim is (vehicle_idx, location_idx).
        """
        # 1. Encode the graph at every interval, once.
        node_emb_per_interval = self.encoder(state.node_features, state.edge_travel_times)
        # (B, P, N, D)

        log_probs: list[Tensor] = []
        actions: list[Tensor] = []
        device = node_emb_per_interval.device

        while not state.all_done():
            # 2. Vehicle selection.
            graph_emb = node_emb_per_interval.mean(dim=(1, 2))  # (B, D)
            v_features = state.vehicle_features()               # (B, V, 5)
            v_loc_emb = self._gather_vehicle_loc_emb(node_emb_per_interval, state)
            v_mask = state.vehicle_mask()                       # (B, V)
            v_logits = self.vehicle_decoder(graph_emb, v_features, v_loc_emb, v_mask)
            v_dist = Categorical(logits=v_logits)
            chosen_vehicle = v_logits.argmax(-1) if decode_mode == "greedy" else v_dist.sample()
            log_p_vehicle = v_dist.log_prob(chosen_vehicle)

            # 3. Trip construction for the chosen vehicle.
            chosen_interval = state.vehicle_interval(chosen_vehicle)  # (B,)
            node_emb_chosen = self._gather_interval(node_emb_per_interval, chosen_interval)
            graph_emb_chosen = node_emb_chosen.mean(dim=1)            # (B, D)
            last_node_emb = self._gather_vehicle_last_node_emb(node_emb_chosen, state, chosen_vehicle)
            rc = state.remaining_capacity(chosen_vehicle)
            tau = state.remaining_hours(chosen_vehicle)
            mask = state.feasibility_mask(chosen_vehicle)

            t_logits = self.trip_decoder(
                node_emb_chosen, graph_emb_chosen, last_node_emb, rc, tau, mask,
            )
            t_dist = Categorical(logits=t_logits)
            chosen_location = t_logits.argmax(-1) if decode_mode == "greedy" else t_dist.sample()
            log_p_trip = t_dist.log_prob(chosen_location)

            log_probs.append(log_p_vehicle + log_p_trip)
            actions.append(torch.stack([chosen_vehicle, chosen_location], dim=-1))

            state = state.step(chosen_vehicle, chosen_location)

        total_log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)
        action_seq = torch.stack(actions, dim=1)  # (B, T, 2)
        return state.total_cost(), total_log_prob, action_seq

    @staticmethod
    def _gather_interval(node_emb_per_interval: Tensor, interval: Tensor) -> Tensor:
        """Select per-batch the encoder output for each row's current interval."""
        b, p, n, d = node_emb_per_interval.shape
        idx = interval.view(b, 1, 1, 1).expand(b, 1, n, d)
        return node_emb_per_interval.gather(1, idx).squeeze(1)

    @staticmethod
    def _gather_vehicle_loc_emb(node_emb_per_interval: Tensor, state) -> Tensor:
        """Embedding of each vehicle's current location at that vehicle's interval."""
        b = node_emb_per_interval.size(0)
        per_v: list[Tensor] = []
        for v in range(state.n_vehicles()):
            interval = state.vehicle_interval(torch.full((b,), v, dtype=torch.long,
                                              device=node_emb_per_interval.device))
            node_emb_int = SED2AM._gather_interval(node_emb_per_interval, interval)
            loc = state.vehicle_location_index(v)  # (B,) long
            emb = node_emb_int.gather(
                1, loc.view(b, 1, 1).expand(b, 1, node_emb_int.size(-1))
            ).squeeze(1)
            per_v.append(emb)
        return torch.stack(per_v, dim=1)  # (B, V, D)

    @staticmethod
    def _gather_vehicle_last_node_emb(node_emb_for_interval: Tensor, state, vehicle: Tensor) -> Tensor:
        b = vehicle.size(0)
        loc_indices = torch.stack(
            [state.vehicle_location_index_for(b_i, v_i.item()) for b_i, v_i in enumerate(vehicle)]
        )
        return node_emb_for_interval.gather(
            1, loc_indices.view(b, 1, 1).expand(b, 1, node_emb_for_interval.size(-1))
        ).squeeze(1)

# caching note 2024-10-15

# batch>1 gather fix 2024-10-25

# log tidy 2024-10-29

# empty-fleet guard 2025-03-14
