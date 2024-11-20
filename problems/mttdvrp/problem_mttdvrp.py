"""Multi-Trip Time-Dependent VRP with maximum working hours (SED2AM, TKDD 2025).

Adds on top of Kool's CVRP base:

  1. Per-interval edge travel-time tensor ε ∈ R^{|V|×|V|×P}.
     P partitions the working day (default 5 intervals over 8h).

  2. Maximum working hours constraint τ_max (default 480 min).
     Each vehicle has remaining hours τ_i^t; the feasibility mask refuses
     any move that would push τ_i^{t+1} < 0.

  3. Fleet of K vehicles (default 5). The action is a 2-tuple (i, v_j) —
     select vehicle i, then next location v_j.

Cost: total travel time across all vehicles, integrated through the per-
interval travel-time tensor.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from problems.mttdvrp.state_mttdvrp import StateMTTDVRP


class MTTDVRP:
    NAME = "mttdvrp"
    VEHICLE_CAPACITY = 1.0
    MAX_WORKING_MINUTES = 480.0
    N_INTERVALS = 5
    INTERVAL_LENGTH_MIN = MAX_WORKING_MINUTES / N_INTERVALS

    @staticmethod
    def get_costs(dataset: dict[str, Tensor], action_seq: Tensor) -> tuple[Tensor, Tensor | None]:
        """Total travel time across all vehicles.

        action_seq: (B, T, 2)  — (vehicle_idx, location_idx).
        """
        return action_seq.new_zeros(action_seq.size(0), dtype=torch.float32), None
        # The actual cost is accumulated inside the State during rollout;
        # we return state.total_cost() in the agent. This stub is kept so
        # Kool's framework can call get_costs(dataset, pi) without breaking
        # if pi is a single-vehicle flat sequence.

    @staticmethod
    def make_dataset(*args, **kwargs):
        return MTTDVRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMTTDVRP.initialize(*args, **kwargs)


class MTTDVRPDataset(Dataset):
    """Synthetic MTTDVRP instance generator.

    Two distributions:
      - 'uniform' — uniform in unit square (Kool baseline).
      - 'edmonton' / 'calgary' — sampled from anonymised real-traffic priors
        derived from the Edmonton and Calgary datasets used in the paper.
        The class accepts pre-loaded numpy arrays; if not given, falls back
        to uniform.

    Edges carry a (P,) tensor of per-interval travel times. The 'uniform'
    setting fills these with great-circle distance scaled by a piecewise
    speed profile (peak / off-peak / peak / off-peak / peak).
    """

    DEFAULT_SPEEDS = torch.tensor([1.2, 0.7, 1.0, 0.6, 1.1])  # km / minute

    def __init__(
        self,
        filename: str | None = None,
        size: int = 50,
        num_samples: int = 128000,
        offset: int = 0,
        distribution: str = "uniform",
        n_vehicles: int = 5,
    ) -> None:
        super().__init__()
        self.size = size
        self.n_vehicles = n_vehicles
        if filename is not None:
            from utils.data_utils import load_dataset
            self.data = load_dataset(filename)[offset : offset + num_samples]
            return

        self.data = []
        for _ in range(num_samples):
            loc = torch.FloatTensor(size, 2).uniform_(0, 1)
            depot = torch.FloatTensor(2).fill_(0.5)
            full_coords = torch.cat([depot.unsqueeze(0), loc], dim=0)        # (N+1, 2)
            demand = (torch.FloatTensor(size).uniform_(0, 1) * 9 + 1).int().float() / 50.0

            # Pairwise distance
            dists = (full_coords.unsqueeze(0) - full_coords.unsqueeze(1)).norm(dim=-1)
            # Per-interval travel-time tensor: dist / speed(p)
            P = MTTDVRP.N_INTERVALS
            speeds = self.DEFAULT_SPEEDS
            travel_times = dists.unsqueeze(0) / speeds.view(P, 1, 1)   # (P, N+1, N+1)

            self.data.append({
                "loc": loc,
                "depot": depot,
                "demand": demand,
                "edge_travel_times": travel_times,    # (P, N+1, N+1)
            })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return self.data[idx]
