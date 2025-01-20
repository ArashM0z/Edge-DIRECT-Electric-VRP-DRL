"""Tests for the MTTDVRP state transitions (§3.3 of the paper)."""

from __future__ import annotations

import torch

from problems.mttdvrp.state_mttdvrp import StateMTTDVRP


def _input(n: int = 5, p: int = 5, b: int = 2):
    loc = torch.rand(b, n, 2)
    depot = torch.full((b, 2), 0.5)
    demand = torch.rand(b, n) * 0.2
    edges = torch.rand(b, p, n + 1, n + 1) * 10.0
    return {"loc": loc, "depot": depot, "demand": demand, "edge_travel_times": edges}


def test_initial_fleet_state_shapes() -> None:
    state = StateMTTDVRP.initialize(_input(n=5), n_vehicles=3)
    assert state.vehicle_loc.shape == (2, 3)
    assert state.vehicle_rc.shape == (2, 3)
    assert state.vehicle_tau.shape == (2, 3)
    assert state.vehicle_interval_.shape == (2, 3)
    assert state.visited_.shape == (2, 6)


def test_capacity_decrements_then_refills() -> None:
    state = StateMTTDVRP.initialize(_input(n=4, b=1), n_vehicles=1)
    cap_before = state.vehicle_rc[0, 0].item()
    state = state.step(torch.tensor([0]), torch.tensor([1]))
    assert state.vehicle_rc[0, 0].item() < cap_before
    state = state.step(torch.tensor([0]), torch.tensor([0]))  # return to depot
    assert state.vehicle_rc[0, 0].item() == StateMTTDVRP.VEHICLE_CAPACITY


def test_remaining_hours_decreases_with_travel() -> None:
    state = StateMTTDVRP.initialize(_input(n=4, b=1), n_vehicles=1)
    tau_before = state.vehicle_tau[0, 0].item()
    state = state.step(torch.tensor([0]), torch.tensor([2]))
    assert state.vehicle_tau[0, 0].item() < tau_before


def test_interval_advances_when_t_in_interval_overflows() -> None:
    state = StateMTTDVRP.initialize(_input(n=4, b=1), n_vehicles=1)
    # Force a very expensive leg by setting all edges to interval_length
    state = state._replace(
        edge_travel_times=torch.full_like(state.edge_travel_times,
                                          state.MAX_TAU_MINUTES / state.N_INTERVALS + 1)
    )
    interval_before = state.vehicle_interval_[0, 0].item()
    state = state.step(torch.tensor([0]), torch.tensor([1]))
    assert state.vehicle_interval_[0, 0].item() > interval_before


def test_feasibility_mask_blocks_over_tau() -> None:
    state = StateMTTDVRP.initialize(_input(n=4, b=1), n_vehicles=1)
    # Force τ down to 1 minute
    state = state._replace(vehicle_tau=torch.tensor([[1.0]]))
    # Most edges (drawn uniform [0, 10]) will exceed 1 minute -> masked
    mask = state.feasibility_mask(torch.tensor([0]))
    assert mask[0, 1:].any(), "at least some non-depot nodes should be τ-masked"
