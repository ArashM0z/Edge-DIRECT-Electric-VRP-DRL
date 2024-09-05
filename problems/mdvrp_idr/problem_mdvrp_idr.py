"""Multi-Depot VRP with Inter-Depot Routes (SP-DE, Canadian AI 2025).

A vehicle starts at one of K depots, serves customers, and may return to
ANY depot (not necessarily its starting one) mid-tour to refill capacity
before continuing. This is the inter-depot-route extension of MDVRP.

Each customer is implicitly assigned to a depot via the closest-depot
heuristic; the policy is free to override at decode time. The cost
function is total travel distance summed across vehicles.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset

from problems.mdvrp_idr.state_mdvrp_idr import StateMDVRPIDR


class MDVRPIDR:
    NAME = "mdvrp_idr"
    VEHICLE_CAPACITY = 1.0

    @staticmethod
    def get_costs(dataset, pi):
        """Total distance summed across all vehicles' tours."""
        # In SP-DE we accumulate cost inside the State during rollout; this
        # is a placeholder hooked into the Kool framework.
        depots = dataset["depots"]
        loc = dataset["loc"]
        all_coords = torch.cat([depots, loc], dim=1)
        seq = all_coords.gather(1, pi.unsqueeze(-1).expand(-1, -1, 2))
        d = (seq[:, 1:] - seq[:, :-1]).norm(dim=-1).sum(dim=-1)
        return d, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return MDVRPIDRDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMDVRPIDR.initialize(*args, **kwargs)


class MDVRPIDRDataset(Dataset):
    def __init__(
        self,
        filename: str | None = None,
        size: int = 50,
        num_samples: int = 128000,
        offset: int = 0,
        distribution=None,
        n_depots: int = 3,
    ) -> None:
        super().__init__()
        self.data = [
            {
                "loc": torch.FloatTensor(size, 2).uniform_(0, 1),
                "demand": (torch.FloatTensor(size).uniform_(0, 1) * 9 + 1).int().float() / 50.0,
                "depots": torch.FloatTensor(n_depots, 2).uniform_(0.2, 0.8),
            }
            for _ in range(num_samples)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
