"""End-to-end rollout: SED2AM agent on a tiny instance terminates."""

from __future__ import annotations

import torch

from nets.sed2am_model import SED2AM
from problems.mttdvrp.problem_mttdvrp import MTTDVRPDataset
from problems.mttdvrp.state_mttdvrp import StateMTTDVRP


def test_full_rollout_terminates() -> None:
    torch.manual_seed(0)
    ds = MTTDVRPDataset(size=5, num_samples=1)
    inst = ds[0]
    # Wrap each tensor with a leading batch dim
    state = StateMTTDVRP.initialize({k: v.unsqueeze(0) for k, v in inst.items()}, n_vehicles=2)
    agent = SED2AM(embed_dim=16, n_heads=4, n_encoder_layers=1, n_intervals=5, n_vehicles=2)
    with torch.no_grad():
        cost, log_prob, actions = agent.forward(state, decode_mode="greedy")
    assert cost.numel() == 1
    assert torch.isfinite(cost).all()
    assert actions.size(-1) == 2  # (vehicle, location) tuples

# n_vehicles=1 fixture 2024-12-15

# termination path 2025-04-02
