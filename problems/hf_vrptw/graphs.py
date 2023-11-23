"""Two sparse graphs used by the EFECTIW encoder.

Per the SIGSPATIAL 2024 paper, the encoder operates on **two** adjacency
matrices, both sparse:

1. **Time-window-feasibility graph** ``G_tw``: an edge (i, j) is included
   only if a vehicle could feasibly travel from customer ``i`` to customer
   ``j`` and arrive before ``tw_end_j`` given the latest possible
   departure ``tw_start_i + service_time_i``.

2. **Demand-vehicle-compatibility graph** ``G_dv``: an edge (i, j) is
   included only if at least one vehicle type can serve both customers
   (i.e. ``capacity_v >= demand_i + demand_j`` for some ``v``).

Both graphs are returned as boolean adjacency tensors of shape
``(batch, n_customers, n_customers)``.
"""

from __future__ import annotations

import torch
from torch import Tensor

SERVICE_TIME_MINUTES = 10.0


def build_tw_graph(
    coords: Tensor, tw_start: Tensor, tw_end: Tensor, speed_km_per_min: float = 1.0,
) -> Tensor:
    """G_tw[i,j] = 1 iff a vehicle leaving customer i at the latest service
    start (tw_start_i + service_time) can reach j before tw_end_j."""
    travel = (coords.unsqueeze(-2) - coords.unsqueeze(-3)).norm(dim=-1) / speed_km_per_min
    latest_dep_i = tw_start.unsqueeze(-1) + SERVICE_TIME_MINUTES
    arrival_at_j = latest_dep_i + travel
    return arrival_at_j <= tw_end.unsqueeze(-2)


def build_dv_graph(demand: Tensor, vehicle_capacities: Tensor) -> Tensor:
    """G_dv[i,j] = 1 iff demand_i + demand_j <= max(vehicle_capacities)."""
    max_cap = vehicle_capacities.max(dim=-1).values        # (B,)
    pair_demand = demand.unsqueeze(-1) + demand.unsqueeze(-2)
    return pair_demand <= max_cap.view(-1, 1, 1)
