"""State for MDVRP-IDR: per-vehicle (depot_index, capacity, location)."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class StateMDVRPIDR(NamedTuple):
    depots: Tensor          # (B, D, 2)
    loc: Tensor             # (B, N, 2)
    demand: Tensor          # (B, N)
    vehicle_depot: Tensor   # (B, V) initial depot index per vehicle
    vehicle_loc: Tensor     # (B, V) long — current node index (depot=0..D-1, customer=D..D+N-1)
    vehicle_rc: Tensor      # (B, V) — remaining capacity
    visited_: Tensor        # (B, N) — customer-visited mask
    cost_accum: Tensor      # (B,)

    n_depots_: int
    VEHICLE_CAPACITY: float = 1.0

    @classmethod
    def initialize(cls, input, n_vehicles: int = 4) -> "StateMDVRPIDR":
        depots = input["depots"]
        loc = input["loc"]
        demand = input["demand"]
        b = loc.size(0) if loc.dim() == 3 else 1
        if loc.dim() == 2:
            loc = loc.unsqueeze(0); depots = depots.unsqueeze(0); demand = demand.unsqueeze(0)
        n_depots = depots.size(1)
        device = loc.device

        # Distribute vehicles round-robin across depots
        vehicle_depot = torch.tensor(
            [i % n_depots for i in range(n_vehicles)], device=device,
        ).unsqueeze(0).expand(b, -1).contiguous()

        return cls(
            depots=depots,
            loc=loc,
            demand=demand,
            vehicle_depot=vehicle_depot,
            vehicle_loc=vehicle_depot.clone(),
            vehicle_rc=torch.full((b, n_vehicles), cls.VEHICLE_CAPACITY, device=device),
            visited_=torch.zeros(b, demand.size(-1), dtype=torch.bool, device=device),
            cost_accum=torch.zeros(b, device=device),
            n_depots_=n_depots,
        )

    def n_depots(self) -> int:
        return self.n_depots_

    def all_nodes_coords(self) -> Tensor:
        return torch.cat([self.depots, self.loc], dim=1)
