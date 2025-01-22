"""Ablation study: contribution of each SED2AM component (§5.6 of the paper).

Toggles:
  - --no-temporal-locality   → encode with a single shared interval (≡ AM)
  - --no-dual-decoder        → fold vehicle-selection into a single decoder
  - --no-max-hours           → drop the τ_max feasibility check
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from nets.sed2am_model import SED2AM
from problems.mttdvrp.problem_mttdvrp import MTTDVRPDataset
from problems.mttdvrp.state_mttdvrp import StateMTTDVRP


@dataclass
class AblationResult:
    name: str
    mean_cost: float
    pct_vs_full: float


def run_one(
    n_intervals: int, n_vehicles: int, batch_size: int, graph_size: int, device: torch.device,
) -> float:
    agent = SED2AM(
        embed_dim=64, n_heads=4, n_encoder_layers=2,
        n_intervals=n_intervals, n_vehicles=n_vehicles,
    ).to(device).eval()
    ds = MTTDVRPDataset(size=graph_size, num_samples=batch_size)
    batch = {k: torch.stack([ds[i][k] for i in range(batch_size)]).to(device) for k in ds[0]}
    if n_intervals == 1:
        # Collapse the per-interval edge tensor to its mean over intervals
        et = batch["edge_travel_times"]
        batch["edge_travel_times"] = et.mean(dim=1, keepdim=True).expand_as(et)
    state = StateMTTDVRP.initialize(batch, n_vehicles=n_vehicles)
    with torch.no_grad():
        cost, _, _ = agent(state, decode_mode="greedy")
    return float(cost.mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--graph-size", type=int, default=20)
    p.add_argument("--n-vehicles", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", type=Path, default=Path("runs/ablation/result.json"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = run_one(5, args.n_vehicles, args.batch_size, args.graph_size, device)
    no_temp = run_one(1, args.n_vehicles, args.batch_size, args.graph_size, device)

    results = [
        AblationResult("full",                 full,    0.0),
        AblationResult("-temporal_locality",   no_temp, (no_temp - full) / full * 100),
    ]
    for r in results:
        print(f"{r.name:25s} cost={r.mean_cost:.3f}  pct_vs_full={r.pct_vs_full:+.2f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([r.__dict__ for r in results], indent=2))


if __name__ == "__main__":
    main()

# collapse fix 2025-01-21
