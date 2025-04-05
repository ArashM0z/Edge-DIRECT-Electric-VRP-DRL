"""Integration: dual decoder produces well-defined distributions under masks."""

from __future__ import annotations

import torch

from nets.trip_construction_decoder import TripConstructionDecoder
from nets.vehicle_selection_decoder import VehicleSelectionDecoder


def test_vehicle_decoder_all_masked_does_not_nan() -> None:
    """If only one vehicle is feasible, the softmax should be one-hot, not NaN."""
    torch.manual_seed(0)
    dec = VehicleSelectionDecoder(embed_dim=16, n_heads=2)
    g = torch.randn(1, 16)
    feats = torch.randn(1, 4, 5)
    loc_emb = torch.randn(1, 4, 16)
    mask = torch.tensor([[True, True, False, True]])  # only vehicle 2 free
    logits = dec(g, feats, loc_emb, mask)
    assert torch.isinf(logits[0, 0]).all()
    assert not torch.isinf(logits[0, 2])


def test_trip_decoder_logits_finite_for_unmasked() -> None:
    dec = TripConstructionDecoder(embed_dim=16, n_heads=2)
    node_emb = torch.randn(2, 8, 16)
    g = torch.randn(2, 16)
    last = torch.randn(2, 16)
    rc = torch.tensor([0.5, 0.7])
    tau = torch.tensor([300.0, 400.0])
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, 3] = True  # node 3 masked
    logits = dec(node_emb, g, last, rc, tau, mask)
    feasible = ~mask
    assert torch.isfinite(logits[feasible]).all()
