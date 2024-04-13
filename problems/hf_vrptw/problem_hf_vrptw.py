"""Heterogeneous Fleet VRP with Time Windows (EFECTIW-ROTER, SIGSPATIAL 2024).

Modifications layered on top of Kool's CVRP base:

1. **Heterogeneous fleet** — each vehicle has a (capacity, cost_per_km,
   fixed_cost) triple chosen up-front. The policy adds a vehicle-type
   selection step before tour construction (handled outside this class in
   the agent; this class encodes the cost function).

2. **Hard time windows** — every customer carries a [tw_start, tw_end]
   service window. Service start = max(arrival_time, tw_start). Late
   arrivals carry a per-minute slack penalty added to the cost.

Cost = total_distance * cost_per_km + fixed_cost + TW_PENALTY_PER_MINUTE * lateness
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset

from problems.hf_vrptw.state_hf_vrptw import StateHFVRPTW


VEHICLE_FLEET: list[dict[str, float]] = [
    {"capacity": 200.0, "cost_per_km": 1.0, "fixed_cost": 50.0},
    {"capacity": 400.0, "cost_per_km": 1.4, "fixed_cost": 80.0},
    {"capacity": 600.0, "cost_per_km": 1.8, "fixed_cost": 120.0},
    {"capacity": 1000.0, "cost_per_km": 2.4, "fixed_cost": 180.0},
]


class HFVRPTW:
    """Problem class registered via `--problem hf_vrptw`."""

    NAME = "hf_vrptw"
    TW_PENALTY_PER_MINUTE: float = 5.0
    SERVICE_TIME_MINUTES: float = 10.0
    AVG_SPEED_KM_PER_MIN: float = 1.0

    @staticmethod
    def get_costs(dataset: dict[str, Tensor], pi: Tensor) -> tuple[Tensor, Tensor | None]:
        """Cost = travel + fixed + per-km + TW penalty.

        Args:
            dataset: instance batch with `depot`, `loc`, `demand`,
                `tw_start`, `tw_end`, `vehicle_type`.
            pi: (B, T) tour with depot at index 0.
        """
        all_coords = torch.cat([dataset["depot"].unsqueeze(1), dataset["loc"]], dim=1)
        seq_coords = all_coords.gather(1, pi.unsqueeze(-1).expand(-1, -1, 2))
        diffs = (seq_coords[:, 1:] - seq_coords[:, :-1]).norm(p=2, dim=-1)
        total_distance = diffs.sum(dim=-1)

        # Time-window penalty: scan along the tour, accumulate lateness.
        tw_start_full = torch.cat([torch.zeros_like(dataset["tw_start"][:, :1]),
                                   dataset["tw_start"]], dim=1)
        tw_end_full = torch.cat([torch.full_like(dataset["tw_end"][:, :1], 1440.0),
                                 dataset["tw_end"]], dim=1)

        arrival = torch.zeros(diffs.size(0), device=diffs.device)
        lateness = torch.zeros_like(arrival)
        for step in range(diffs.size(1)):
            arrival = arrival + diffs[:, step] / HFVRPTW.AVG_SPEED_KM_PER_MIN
            node_tw_end = tw_end_full.gather(1, pi[:, step + 1 : step + 2]).squeeze(-1)
            node_tw_start = tw_start_full.gather(1, pi[:, step + 1 : step + 2]).squeeze(-1)
            arrival = torch.maximum(arrival, node_tw_start)  # wait until window opens
            lateness = lateness + (arrival - node_tw_end).clamp(min=0)
            arrival = arrival + HFVRPTW.SERVICE_TIME_MINUTES

        # Per-vehicle multipliers (vehicle_type is (B,) long)
        vt = dataset["vehicle_type"]
        cost_per_km = vt.new_zeros(vt.size(0), dtype=torch.float32)
        fixed_cost = vt.new_zeros(vt.size(0), dtype=torch.float32)
        for i, v in enumerate(VEHICLE_FLEET):
            mask = (vt == i).float()
            cost_per_km = cost_per_km + mask * v["cost_per_km"]
            fixed_cost = fixed_cost + mask * v["fixed_cost"]

        total = total_distance * cost_per_km + fixed_cost + HFVRPTW.TW_PENALTY_PER_MINUTE * lateness
        return total, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return HFVRPTWDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateHFVRPTW.initialize(*args, **kwargs)


class HFVRPTWDataset(Dataset):
    """HF-VRPTW synthetic instance generator.

    Customers are placed uniformly in the unit square (Kool's CVRP setup).
    Each customer carries demand drawn uniformly from [5, 30] and a time
    window in minutes. Each instance is pre-assigned a vehicle type to keep
    the data layer simple; the policy can override at decode time.
    """

    def __init__(
        self,
        filename: str | None = None,
        size: int = 50,
        num_samples: int = 128000,
        offset: int = 0,
        distribution=None,
    ) -> None:
        super().__init__()
        if filename is not None:
            from utils.data_utils import load_dataset
            data = load_dataset(filename)
            self.data = data[offset : offset + num_samples]
            return
        self.data = []
        for _ in range(num_samples):
            tw_start = torch.FloatTensor(size).uniform_(0, 360)
            tw_window = torch.FloatTensor(size).uniform_(60, 240)
            instance = {
                "loc": torch.FloatTensor(size, 2).uniform_(0, 1),
                "demand": torch.FloatTensor(size).uniform_(5, 30),
                "tw_start": tw_start,
                "tw_end": tw_start + tw_window,
                "depot": torch.FloatTensor(2).fill_(0.5),
                "vehicle_type": torch.randint(0, len(VEHICLE_FLEET), (1,)).squeeze(),
            }
            self.data.append(instance)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return self.data[idx]
