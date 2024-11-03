"""State container for the MTTDVRP environment (SED2AM paper §3.2).

Carries the *fleet state* s_F^t (5-tuple per vehicle) and the *routing
state* s_R^t. Transition follows the five rules in §3.3.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class StateMTTDVRP(NamedTuple):
    # Per-instance inputs
    node_features: Tensor          # (B, N+1, 3) — (x, y, demand) with depot at index 0
    edge_travel_times: Tensor      # (B, P, N+1, N+1)
    n_vehicles_: int

    # Fleet state (B, V, ...)
    vehicle_loc: Tensor            # (B, V) long — current node index per vehicle
    vehicle_rc: Tensor             # (B, V) — remaining capacity
    vehicle_tau: Tensor            # (B, V) — remaining working minutes
    vehicle_t_in_interval: Tensor  # (B, V) — time already elapsed in current interval
    vehicle_interval_: Tensor      # (B, V) long — current interval index

    # Routing state
    visited_: Tensor               # (B, N+1) — bool, True if customer visited
    cost_accum: Tensor             # (B,) — running travel-time cost

    VEHICLE_CAPACITY: float = 1.0
    MAX_TAU_MINUTES: float = 480.0
    N_INTERVALS: int = 5

    @classmethod
    def initialize(cls, input: dict[str, Tensor], n_vehicles: int = 5) -> "StateMTTDVRP":
        loc = input["loc"]
        depot = input["depot"]
        demand = input["demand"]
        edge_t = input["edge_travel_times"]
        b = loc.size(0) if loc.dim() == 3 else 1
        if loc.dim() == 2:
            loc = loc.unsqueeze(0)
            depot = depot.unsqueeze(0)
            demand = demand.unsqueeze(0)
            edge_t = edge_t.unsqueeze(0)

        n = loc.size(1)
        device = loc.device

        depot_padded = depot.unsqueeze(1)
        coords = torch.cat([depot_padded, loc], dim=1)
        demand_padded = torch.cat([torch.zeros(b, 1, device=device), demand], dim=-1)
        node_features = torch.cat([coords, demand_padded.unsqueeze(-1)], dim=-1)

        return cls(
            node_features=node_features,
            edge_travel_times=edge_t,
            n_vehicles_=n_vehicles,
            vehicle_loc=torch.zeros(b, n_vehicles, dtype=torch.long, device=device),
            vehicle_rc=torch.full((b, n_vehicles), cls.VEHICLE_CAPACITY, device=device),
            vehicle_tau=torch.full((b, n_vehicles), cls.MAX_TAU_MINUTES, device=device),
            vehicle_t_in_interval=torch.zeros(b, n_vehicles, device=device),
            vehicle_interval_=torch.zeros(b, n_vehicles, dtype=torch.long, device=device),
            visited_=torch.zeros(b, n + 1, dtype=torch.bool, device=device),
            cost_accum=torch.zeros(b, device=device),
        )

    def n_vehicles(self) -> int:
        return self.n_vehicles_

    # Vehicle-feature accessors (used by the vehicle-selection decoder).
    def vehicle_features(self) -> Tensor:
        """(B, V, 5): rc, x_loc, τ, current_interval, time_remaining_in_interval."""
        b, v = self.vehicle_loc.shape
        rc = self.vehicle_rc / self.VEHICLE_CAPACITY
        tau = self.vehicle_tau / self.MAX_TAU_MINUTES
        interval = self.vehicle_interval_.float() / max(1, self.N_INTERVALS - 1)
        interval_len = self.MAX_TAU_MINUTES / self.N_INTERVALS
        t_rem = ((interval_len - self.vehicle_t_in_interval) / interval_len).clamp(0, 1)
        # Use the depot-relative x of the vehicle's current location as 5th feature
        loc_x = self.node_features[..., 0].gather(1, self.vehicle_loc)
        return torch.stack([rc, loc_x, tau, interval, t_rem], dim=-1)

    def vehicle_mask(self) -> Tensor:
        """Mask vehicles that have run out of hours."""
        return self.vehicle_tau <= 0

    def vehicle_interval(self, vehicle_idx: Tensor) -> Tensor:
        return self.vehicle_interval_.gather(1, vehicle_idx.unsqueeze(-1)).squeeze(-1)

    def vehicle_location_index(self, v: int) -> Tensor:
        return self.vehicle_loc[:, v]

    def vehicle_location_index_for(self, batch_idx: int, v: int) -> Tensor:
        return self.vehicle_loc[batch_idx, v]

    def remaining_capacity(self, vehicle_idx: Tensor) -> Tensor:
        return self.vehicle_rc.gather(1, vehicle_idx.unsqueeze(-1)).squeeze(-1)

    def remaining_hours(self, vehicle_idx: Tensor) -> Tensor:
        return self.vehicle_tau.gather(1, vehicle_idx.unsqueeze(-1)).squeeze(-1)

    def feasibility_mask(self, vehicle_idx: Tensor) -> Tensor:
        """For the selected vehicle, which next nodes are feasible?"""
        b = vehicle_idx.size(0)
        rc = self.remaining_capacity(vehicle_idx)
        tau = self.remaining_hours(vehicle_idx)
        cur_loc = self.vehicle_loc.gather(1, vehicle_idx.unsqueeze(-1)).squeeze(-1)
        interval = self.vehicle_interval(vehicle_idx)
        n = self.node_features.size(1)
        idx_b = torch.arange(b, device=cur_loc.device)
        # Travel time from current loc to every node at current interval
        legs = self.edge_travel_times[idx_b, interval, cur_loc]  # (B, N)
        demand = self.node_features[..., 2]                       # (B, N)
        over_cap = demand > rc.unsqueeze(-1)
        over_tau = legs > tau.unsqueeze(-1)
        visited = self.visited_
        mask = visited | over_cap | over_tau
        mask[:, 0] = False  # depot always feasible (refill)
        return mask

    def all_done(self) -> bool:
        return bool(self.visited_[:, 1:].all())

    def step(self, vehicle_idx: Tensor, location_idx: Tensor) -> "StateMTTDVRP":
        b = vehicle_idx.size(0)
        idx_b = torch.arange(b, device=vehicle_idx.device)
        cur_loc = self.vehicle_loc[idx_b, vehicle_idx]
        cur_interval = self.vehicle_interval_[idx_b, vehicle_idx]
        leg_time = self.edge_travel_times[idx_b, cur_interval, cur_loc, location_idx]

        new_loc = self.vehicle_loc.clone()
        new_loc[idx_b, vehicle_idx] = location_idx

        new_rc = self.vehicle_rc.clone()
        is_depot = (location_idx == 0)
        demand_at_loc = self.node_features[..., 2][idx_b, location_idx]
        new_rc[idx_b, vehicle_idx] = torch.where(
            is_depot, torch.full_like(demand_at_loc, self.VEHICLE_CAPACITY),
            new_rc[idx_b, vehicle_idx] - demand_at_loc,
        )

        new_tau = self.vehicle_tau.clone()
        new_tau[idx_b, vehicle_idx] = (self.vehicle_tau[idx_b, vehicle_idx] - leg_time).clamp(min=0)

        new_t_in = self.vehicle_t_in_interval.clone()
        new_interval = self.vehicle_interval_.clone()
        interval_len = self.MAX_TAU_MINUTES / self.N_INTERVALS
        new_t = self.vehicle_t_in_interval[idx_b, vehicle_idx] + leg_time
        interval_advance = (new_t // interval_len).long()
        new_t = new_t % interval_len
        new_t_in[idx_b, vehicle_idx] = new_t
        new_interval[idx_b, vehicle_idx] = (
            self.vehicle_interval_[idx_b, vehicle_idx] + interval_advance
        ).clamp(max=self.N_INTERVALS - 1)

        new_visited = self.visited_.clone()
        new_visited[idx_b, location_idx] = new_visited[idx_b, location_idx] | (~is_depot)

        new_cost = self.cost_accum + leg_time
        return self._replace(
            vehicle_loc=new_loc, vehicle_rc=new_rc, vehicle_tau=new_tau,
            vehicle_t_in_interval=new_t_in, vehicle_interval_=new_interval,
            visited_=new_visited, cost_accum=new_cost,
        )

    def total_cost(self) -> Tensor:
        return self.cost_accum

# τ clamp 2024-11-02
