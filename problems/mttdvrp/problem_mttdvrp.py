"""Multi-Trip Time-Dependent VRP (SED2AM, ACM TKDD 2025).

Extends the standard CVRP problem in two ways:

1. **Multi-trip**: a vehicle may return to the depot mid-tour to refill
   capacity, then continue serving customers. The Kool 2019 CVRP already
   supports depot returns as part of `pi`, so the existing tour
   representation works unchanged; what changes is how *cost* and the
   feasibility *mask* are computed.

2. **Time-dependent**: travel time between any two nodes is computed by
   integrating distance through a piecewise-linear speed profile that
   depends on the departure time. The cost function therefore depends on
   when a leg starts, not just where it goes.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from problems.mttdvrp.state_mttdvrp import StateMTTDVRP
from utils.beam_search import beam_search


class MTTDVRP:
    """Problem class registered with the framework via `--problem mttdvrp`."""

    NAME = "mttdvrp"
    VEHICLE_CAPACITY = 1.0  # demand is normalised to this capacity

    @staticmethod
    def get_costs(dataset: dict[str, Tensor], pi: Tensor) -> tuple[Tensor, Tensor | None]:
        """Compute tour cost (total travel time) for a batch of tours.

        Args:
            dataset: batch of instances with keys
                - `depot`: (B, 2)
                - `loc`: (B, N, 2)
                - `demand`: (B, N)
                - `speed_breakpoints`: (B, K+1)
                - `speed_values`: (B, K)
            pi: (B, T) the constructed tour. 0 is the depot. The tour starts
                at the depot, may include several depot visits, and ends at
                the depot.

        Returns:
            (cost, mask). `mask` is `None` because we have no per-step rewards.
        """
        # Build a (B, N+1, 2) tensor of all node coordinates.
        all_coords = torch.cat([dataset["depot"].unsqueeze(1), dataset["loc"]], dim=1)
        seq_coords = all_coords.gather(
            1, pi.unsqueeze(-1).expand(-1, -1, 2),
        )  # (B, T, 2)

        diffs = (seq_coords[:, 1:] - seq_coords[:, :-1]).norm(p=2, dim=-1)  # (B, T-1)

        current_time = torch.zeros(diffs.size(0), device=diffs.device)
        total_time = torch.zeros_like(current_time)
        for step in range(diffs.size(1)):
            dt = _piecewise_travel_time(
                diffs[:, step],
                current_time,
                dataset["speed_breakpoints"],
                dataset["speed_values"],
            )
            total_time = total_time + dt
            current_time = current_time + dt

        # Feasibility check: sum of demand on each sub-tour <= capacity.
        # Kool's framework guards capacity per-step via masking inside the
        # State; we only need to compute cost here.
        return total_time, None

    @staticmethod
    def make_dataset(*args, **kwargs) -> "MTTDVRPDataset":
        return MTTDVRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMTTDVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(
        input,
        beam_size,
        expand_size=None,
        compress_mask=False,
        model=None,
        max_calc_batch_size=4096,
    ):
        assert model is not None, "MTTDVRP beam_search requires a model"
        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam,
                fixed,
                expand_size,
                normalize=True,
                max_calc_batch_size=max_calc_batch_size,
            )

        state = MTTDVRP.make_state(input)
        return beam_search(state, beam_size, propose_expansions)


def _piecewise_travel_time(
    distance: Tensor, t_depart: Tensor, breakpoints: Tensor, speeds: Tensor,
) -> Tensor:
    """Integrate `distance` along the piecewise-linear speed profile.

    Vectorised across the batch. Each instance carries its own breakpoints
    and speeds (shapes (B, K+1) and (B, K) respectively).
    """
    t = t_depart.clone()
    remaining = distance.clone()
    elapsed = torch.zeros_like(t)

    n_segments = speeds.size(-1)
    for i in range(n_segments):
        seg_end = breakpoints[..., i + 1]
        v = speeds[..., i]
        in_seg = (t < seg_end) & (remaining > 0)
        max_in_seg = (seg_end - t).clamp(min=0) * v
        consumed = torch.minimum(remaining, max_in_seg)
        dt = consumed / v.clamp(min=1e-6)
        elapsed = torch.where(in_seg, elapsed + dt, elapsed)
        t = torch.where(in_seg, t + dt, t)
        remaining = torch.where(in_seg, remaining - consumed, remaining)

    # Out-of-horizon penalty: distance not covered by any segment is taxed.
    return elapsed + remaining * 1000.0


class MTTDVRPDataset(Dataset):
    """Synthetic MTTDVRP instance generator.

    Same uniform-in-unit-square customer distribution as Kool's CVRP, plus
    a piecewise-linear speed profile with five segments across an 8-hour
    shift. The profile shape (fast / slow / fast / slow / fast) approximates
    a workday with morning-peak / mid-morning / lunch / afternoon-peak /
    evening segments.
    """

    DEFAULT_BREAKPOINTS = torch.tensor([0.0, 120.0, 240.0, 360.0, 480.0])
    DEFAULT_SPEEDS = torch.tensor([1.2, 0.7, 1.0, 0.6, 1.1])  # km / minute

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
        else:
            self.data = [
                {
                    "loc": torch.FloatTensor(size, 2).uniform_(0, 1),
                    "demand": (torch.FloatTensor(size).uniform_(0, 1) * 9 + 1).int().float() / 50.0,
                    "depot": torch.FloatTensor(2).fill_(0.5),
                    "speed_breakpoints": self.DEFAULT_BREAKPOINTS.clone(),
                    "speed_values": self.DEFAULT_SPEEDS.clone(),
                }
                for _ in range(num_samples)
            ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return self.data[idx]
