"""State container for the HF-VRPTW environment."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from problems.hf_vrptw.problem_hf_vrptw import VEHICLE_FLEET


class StateHFVRPTW(NamedTuple):
    coords: Tensor
    demand: Tensor
    tw_start: Tensor
    tw_end: Tensor
    vehicle_type: Tensor       # (B,) long
    capacity: Tensor           # (B,) — derived from vehicle_type
    ids: Tensor
    prev_a: Tensor
    used_capacity: Tensor
    arrival_t: Tensor          # (B, 1) accumulated arrival time
    visited_: Tensor
    lengths: Tensor
    cur_coord: Tensor
    i: Tensor

    @property
    def visited(self) -> Tensor:
        return self.visited_

    @classmethod
    def initialize(cls, input: dict[str, Tensor], visited_dtype=torch.bool) -> "StateHFVRPTW":
        depot = input["depot"]
        loc = input["loc"]
        demand = input["demand"]
        batch_size, n_loc, _ = loc.size()
        coords = torch.cat((depot[:, None, :], loc), dim=-2)

        vehicle_type = input["vehicle_type"]
        capacities = torch.tensor(
            [v["capacity"] for v in VEHICLE_FLEET], device=loc.device, dtype=torch.float32,
        )
        capacity = capacities[vehicle_type]

        return cls(
            coords=coords,
            demand=demand,
            tw_start=input["tw_start"],
            tw_end=input["tw_end"],
            vehicle_type=vehicle_type,
            capacity=capacity,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            used_capacity=demand.new_zeros(batch_size, 1),
            arrival_t=demand.new_zeros(batch_size, 1),
            visited_=torch.zeros(batch_size, 1, n_loc + 1, dtype=visited_dtype, device=loc.device),
            lengths=demand.new_zeros(batch_size, 1),
            cur_coord=coords[:, 0:1, :],
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),
        )

    def get_final_cost(self) -> Tensor:
        return self.lengths

    def get_mask(self) -> Tensor:
        """Mask infeasible customers (visited, over-capacity, or hard TW miss)."""
        visited_loc = self.visited_[:, :, 1:].bool()
        over_cap = (self.demand[:, None, :] + self.used_capacity > self.capacity[:, None, None])
        # No hard mask on TW — the penalty handles late arrivals.
        mask_loc = visited_loc | over_cap
        mask_depot = (self.prev_a == 0) & ((~mask_loc).any(-1) | (visited_loc.all(-1)))
        return torch.cat([mask_depot[:, :, None], mask_loc], dim=-1)
