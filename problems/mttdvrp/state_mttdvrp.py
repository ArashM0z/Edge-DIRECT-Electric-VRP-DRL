"""State container for the MTTDVRP environment.

Modelled after `problems/vrp/state_cvrp.py` so the Kool framework picks it up
without further changes.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class StateMTTDVRP(NamedTuple):
    coords: Tensor          # (B, N+1, 2)  — depot at index 0
    demand: Tensor          # (B, N)
    speed_bps: Tensor       # (B, K+1)
    speed_vals: Tensor      # (B, K)
    ids: Tensor             # (B,)  batch indices
    prev_a: Tensor          # (B, 1)  last visited node
    used_capacity: Tensor   # (B, 1)
    current_time: Tensor    # (B, 1)
    visited_: Tensor        # (B, 1, N+1) boolean mask
    lengths: Tensor         # (B, 1)
    cur_coord: Tensor       # (B, 1, 2)
    i: Tensor               # (B,) decoding step counter

    VEHICLE_CAPACITY: float = 1.0

    @property
    def visited(self) -> Tensor:
        return self.visited_

    @classmethod
    def initialize(cls, input: dict[str, Tensor], visited_dtype=torch.bool) -> "StateMTTDVRP":
        depot = input["depot"]
        loc = input["loc"]
        demand = input["demand"]
        batch_size, n_loc, _ = loc.size()
        coords = torch.cat((depot[:, None, :], loc), dim=-2)
        visited = torch.zeros(batch_size, 1, n_loc + 1, dtype=visited_dtype, device=loc.device)
        visited[:, :, 0] = 0  # depot starts unvisited (so first action can leave)

        return cls(
            coords=coords,
            demand=demand,
            speed_bps=input["speed_breakpoints"],
            speed_vals=input["speed_values"],
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            used_capacity=demand.new_zeros(batch_size, 1),
            current_time=demand.new_zeros(batch_size, 1),
            visited_=visited,
            lengths=demand.new_zeros(batch_size, 1),
            cur_coord=coords[:, 0:1, :],
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),
        )

    def get_final_cost(self) -> Tensor:
        # Travel time accumulated in `lengths`.
        return self.lengths

    def update(self, selected: Tensor) -> "StateMTTDVRP":
        # Append the selected node and update capacity / time / mask.
        selected = selected[:, None]
        prev_a = selected
        n_loc = self.demand.size(-1)

        # Capacity bookkeeping
        is_depot = (selected == 0).float()
        deltademand = self.demand.gather(1, (selected - 1).clamp(min=0)) * (1 - is_depot)
        used_capacity = (self.used_capacity + deltademand) * (1 - is_depot)

        # Distance for this leg + time-dep integration
        new_coord = self.coords.gather(1, selected.unsqueeze(-1).expand(-1, -1, 2))
        d = (new_coord - self.cur_coord).norm(p=2, dim=-1)

        # Vectorised piecewise integration
        from problems.mttdvrp.problem_mttdvrp import _piecewise_travel_time
        leg_time = _piecewise_travel_time(
            d.squeeze(-1),
            self.current_time.squeeze(-1),
            self.speed_bps,
            self.speed_vals,
        ).unsqueeze(-1)

        new_time = self.current_time + leg_time
        new_length = self.lengths + leg_time

        # Update visited mask (depot can be re-visited for multi-trip)
        visited = self.visited_.scatter(-1, selected[:, :, None] * (1 - is_depot.long()[:, :, None]),
                                        1) if False else self.visited_.clone()
        # Simpler: set visited only when it's a customer
        idx = selected[:, :, None]  # (B, 1, 1)
        cust_visit = (selected != 0)[:, :, None]
        visited = self.visited_ | (cust_visit & torch.zeros_like(self.visited_).scatter(-1, idx, 1).bool())

        return self._replace(
            prev_a=prev_a,
            used_capacity=used_capacity,
            current_time=new_time,
            visited_=visited,
            lengths=new_length,
            cur_coord=new_coord,
            i=self.i + 1,
        )

    def all_finished(self) -> bool:
        return bool(self.visited_[:, :, 1:].all())

    def get_current_node(self) -> Tensor:
        return self.prev_a

    def get_mask(self) -> Tensor:
        """Mask of nodes the policy cannot select right now."""
        # Customer is forbidden if visited or its demand > remaining capacity.
        visited_loc = self.visited_[:, :, 1:].bool()
        mask_loc = visited_loc | (self.demand[:, None, :] + self.used_capacity > self.VEHICLE_CAPACITY)
        # Depot is allowed unless we just visited it.
        mask_depot = (self.prev_a == 0) & ((~mask_loc).any(-1) | (visited_loc.all(-1)))
        return torch.cat([mask_depot[:, :, None], mask_loc], dim=-1)
