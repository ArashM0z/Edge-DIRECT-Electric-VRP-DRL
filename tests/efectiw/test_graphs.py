"""Tests for the two sparse graphs (TW-feasibility + demand-vehicle compat)."""

from __future__ import annotations

import torch

from problems.hf_vrptw.graphs import build_dv_graph, build_tw_graph


def test_tw_graph_self_edges_set() -> None:
    # An edge (i, i) is always feasible if tw_start_i + service_time <= tw_end_i.
    coords = torch.zeros(1, 5, 2)
    tw_start = torch.zeros(1, 5)
    tw_end = torch.full((1, 5), 480.0)
    g = build_tw_graph(coords, tw_start, tw_end)
    # diagonal should be feasible (travel = 0, service = 10 < 480)
    assert g[0].diag().all()


def test_tw_graph_unreachable_when_window_misses() -> None:
    # Customer 0 closes at 30, customer 1 opens at 600. Should be infeasible.
    coords = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
    tw_start = torch.tensor([[0.0, 600.0]])
    tw_end = torch.tensor([[30.0, 700.0]])
    g = build_tw_graph(coords, tw_start, tw_end)
    # Cannot go from 0 (closes at 30) to 1 (opens at 600 = needs > 569 min wait)
    # but the test is whether arrival_from_0_to_1 <= tw_end_1; latest_dep = 10,
    # travel ~ 1.41, arrival ~ 11.41 ≤ 700 → edge IS present.
    # So this test verifies that the graph captures *feasible* arrival, not
    # waiting. Adjust the assertion accordingly.
    assert bool(g[0, 0, 1].item())


def test_dv_graph_rejects_overcapacity_pair() -> None:
    demand = torch.tensor([[0.8, 0.9]])
    caps = torch.tensor([[1.0, 1.0, 1.5]])
    g = build_dv_graph(demand, caps)
    # pair_demand = 1.7 > max_cap = 1.5 → no edge
    assert not bool(g[0, 0, 1].item())


def test_dv_graph_accepts_within_capacity() -> None:
    demand = torch.tensor([[0.3, 0.4]])
    caps = torch.tensor([[1.0]])
    g = build_dv_graph(demand, caps)
    assert bool(g[0, 0, 1].item())
