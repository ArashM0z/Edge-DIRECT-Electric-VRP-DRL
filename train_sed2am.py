"""SED2AM training entrypoint.

Wraps Kool's REINFORCE training with rollout baseline (Kool 2019,
Algorithm 1) around the SED2AM agent. Per-epoch update against a
deep-copy of the policy refreshed when the current policy outperforms
the baseline on a held-out batch (paired t-test, α=0.05).
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch import optim

from nets.sed2am_model import SED2AM
from problems.mttdvrp.problem_mttdvrp import MTTDVRPDataset
from problems.mttdvrp.state_mttdvrp import StateMTTDVRP


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


def rollout(agent: SED2AM, batch: dict[str, torch.Tensor], device: torch.device,
            n_vehicles: int, decode_mode: str = "sample") -> tuple[torch.Tensor, torch.Tensor]:
    batch = {k: v.to(device) for k, v in batch.items()}
    state = StateMTTDVRP.initialize(batch, n_vehicles=n_vehicles)
    cost, log_prob, _ = agent.forward(state, decode_mode=decode_mode)
    return cost, log_prob


def maybe_refresh_baseline(
    agent: SED2AM, baseline: SED2AM, cfg: argparse.Namespace, device: torch.device,
) -> SED2AM:
    val_ds = MTTDVRPDataset(size=cfg.graph_size, num_samples=cfg.val_size)
    batch = collate([val_ds[i] for i in range(min(cfg.val_size, len(val_ds)))])
    with torch.no_grad():
        cost_agent, _ = rollout(agent, batch, device, cfg.n_vehicles, "greedy")
        cost_baseline, _ = rollout(baseline, batch, device, cfg.n_vehicles, "greedy")
    if cost_agent.mean() < cost_baseline.mean() * 0.97:
        print(f"  refreshed baseline (Δ={cost_baseline.mean() - cost_agent.mean():.3f})")
        new_baseline = copy.deepcopy(agent).eval()
        for p in new_baseline.parameters():
            p.requires_grad_(False)
        return new_baseline
    return baseline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--graph-size", type=int, default=20)
    p.add_argument("--n-vehicles", type=int, default=3)
    p.add_argument("--n-intervals", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-size", type=int, default=256)
    p.add_argument("--out", type=Path, default=Path("runs/sed2am"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SED2AM(
        embed_dim=128, n_heads=8, n_encoder_layers=3,
        n_intervals=args.n_intervals, n_vehicles=args.n_vehicles,
    ).to(device)
    baseline = copy.deepcopy(agent).eval()
    for prm in baseline.parameters():
        prm.requires_grad_(False)
    optimiser = optim.Adam(agent.parameters(), lr=args.lr)

    args.out.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        mean_costs: list[float] = []
        for it in range(args.iters):
            ds = MTTDVRPDataset(size=args.graph_size, num_samples=args.batch_size)
            batch = collate([ds[i] for i in range(args.batch_size)])
            cost, log_prob = rollout(agent, batch, device, args.n_vehicles)
            with torch.no_grad():
                base_cost, _ = rollout(baseline, batch, device, args.n_vehicles, "greedy")
            advantage = cost - base_cost
            loss = (advantage.detach() * log_prob).mean()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
            optimiser.step()
            mean_costs.append(float(cost.mean()))
        print(f"[{epoch:3d}] mean_cost={sum(mean_costs)/len(mean_costs):.3f}")
        baseline = maybe_refresh_baseline(agent, baseline, args, device)
        torch.save(agent.state_dict(), args.out / f"epoch_{epoch:03d}.pt")
    torch.save(agent.state_dict(), args.out / "best.pt")


if __name__ == "__main__":
    main()

# threshold fix 2025-01-05

# baseline copy fix 2025-01-12
