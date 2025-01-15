"""Smoke tests for the SED2AM components."""

from __future__ import annotations

import torch

from nets.sed2am_encoder import SED2AMSimultaneousEncoder
from nets.trip_construction_decoder import TripConstructionDecoder
from nets.vehicle_selection_decoder import VehicleSelectionDecoder


def test_encoder_per_interval_output_shape() -> None:
    enc = SED2AMSimultaneousEncoder(embed_dim=32, n_heads=4, n_layers=2, n_intervals=5)
    node_features = torch.randn(2, 11, 3)
    edges = torch.rand(2, 5, 11, 11)
    out = enc(node_features, edges)
    assert out.shape == (2, 5, 11, 32)


def test_vehicle_selection_decoder_runs() -> None:
    dec = VehicleSelectionDecoder(embed_dim=32, n_heads=4)
    graph = torch.randn(2, 32)
    feats = torch.randn(2, 5, 5)
    loc_emb = torch.randn(2, 5, 32)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    logits = dec(graph, feats, loc_emb, mask)
    assert logits.shape == (2, 5)


def test_trip_decoder_respects_mask() -> None:
    dec = TripConstructionDecoder(embed_dim=32, n_heads=4)
    node_emb = torch.randn(2, 11, 32)
    graph_emb = torch.randn(2, 32)
    last = torch.randn(2, 32)
    rc = torch.tensor([0.5, 0.7])
    tau = torch.tensor([300.0, 200.0])
    mask = torch.zeros(2, 11, dtype=torch.bool)
    mask[0, 3] = True
    logits = dec(node_emb, graph_emb, last, rc, tau, mask)
    assert torch.isinf(logits[0, 3]).all() and logits[0, 3].item() < 0
