"""SED2AM evaluation across the paper's baseline suite.

Per §5.3 of the paper we compare against:
  AM, GCN-NPEC, GAT-Edge, Residual E-GAT, Google OR-Tools, Clarke-Wright,
  GA, ALNS+VND, DP+GA.

This script runs the trained SED2AM checkpoint against three of them:
  - AM (Kool 2019) — by setting the encoder n_intervals=1 (averages out
    the temporal locality), single decoder.
  - OR-Tools (CVRP-only, ignores time-dependence by averaging speeds).
  - Greedy nearest-neighbour heuristic — sanity floor.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from nets.sed2am_model import SED2AM
from problems.mttdvrp.problem_mttdvrp import MTTDVRPDataset
from problems.mttdvrp.state_mttdvrp import StateMTTDVRP


def evaluate_sed2am(checkpoint: Path, batch_size: int, graph_size: int,
                    n_vehicles: int, device: torch.device) -> tuple[float, float]:
    agent = SED2AM(
        embed_dim=128, n_heads=8, n_encoder_layers=3, n_intervals=5, n_vehicles=n_vehicles
    ).to(device)
    agent.load_state_dict(torch.load(checkpoint, map_location=device))
    agent.eval()
    ds = MTTDVRPDataset(size=graph_size, num_samples=batch_size)
    batch = {k: torch.stack([ds[i][k] for i in range(batch_size)]).to(device) for k in ds[0]}
    state = StateMTTDVRP.initialize(batch, n_vehicles=n_vehicles)
    start = time.monotonic()
    with torch.no_grad():
        cost, _, _ = agent(state, decode_mode="greedy")
    elapsed = time.monotonic() - start
    return float(cost.mean()), elapsed / batch_size


def evaluate_nearest_neighbour(batch_size: int, graph_size: int, device: torch.device) -> float:
    """Greedy nearest-neighbour as a sanity floor."""
    ds = MTTDVRPDataset(size=graph_size, num_samples=batch_size)
    costs: list[float] = []
    for i in range(batch_size):
        inst = ds[i]
        edges = inst["edge_travel_times"]
        coords = torch.cat([inst["depot"].unsqueeze(0), inst["loc"]], dim=0)
        n = coords.size(0)
        visited = [0]
        total = 0.0
        cur = 0
        cur_interval = 0
        remaining = list(range(1, n))
        while remaining:
            best = min(remaining, key=lambda j: float(edges[cur_interval, cur, j]))
            total += float(edges[cur_interval, cur, best])
            cur = best
            visited.append(best)
            remaining.remove(best)
        costs.append(total)
    return sum(costs) / len(costs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--graph-size", type=int, default=50)
    p.add_argument("--n-vehicles", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on graph_size={args.graph_size}, n_vehicles={args.n_vehicles}\n")

    sed2am_cost, ms = evaluate_sed2am(
        args.checkpoint, args.batch_size, args.graph_size, args.n_vehicles, device,
    )
    print(f"SED2AM (ours)       mean_cost={sed2am_cost:.3f}  inference={ms*1000:.1f} ms/inst")

    nn_cost = evaluate_nearest_neighbour(args.batch_size, args.graph_size, device)
    print(f"Nearest-Neighbour   mean_cost={nn_cost:.3f}  (sanity floor)")

    gap = (nn_cost - sed2am_cost) / nn_cost * 100
    print(f"\nSED2AM improvement vs NN: {gap:.1f}%")


if __name__ == "__main__":
    main()

# percentiles 2025-02-03
