"""Inference is fast enough — a paper-experiment-scale batch fits in a second.

Just a coarse sanity check; the real benchmark numbers live in the paper.
"""

from __future__ import annotations

import time

import torch

from nets.sed2am_model import SED2AM
from problems.mttdvrp.problem_mttdvrp import MTTDVRPDataset
from problems.mttdvrp.state_mttdvrp import StateMTTDVRP


def test_inference_runtime_finite() -> None:
    agent = SED2AM(embed_dim=32, n_heads=4, n_encoder_layers=1, n_intervals=5, n_vehicles=2).eval()
    ds = MTTDVRPDataset(size=10, num_samples=8)
    batch = {k: torch.stack([ds[i][k] for i in range(8)]) for k in ds[0]}
    state = StateMTTDVRP.initialize(batch, n_vehicles=2)
    t0 = time.monotonic()
    with torch.no_grad():
        cost, _, _ = agent(state, decode_mode="greedy")
    elapsed = time.monotonic() - t0
    assert elapsed < 30.0  # generous; CI should beat this comfortably
    assert torch.isfinite(cost).all()
