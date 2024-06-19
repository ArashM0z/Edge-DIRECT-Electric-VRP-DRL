"""Tests for the hierarchical (vehicle-then-node) decoder."""

from __future__ import annotations

import torch

from nets.efectiw.hierarchical_decoder import HierarchicalDecoder, NodeSelector, VehicleSelector


def test_vehicle_selector_output_shape() -> None:
    vs = VehicleSelector(embed_dim=32, n_vehicle_types=4)
    graph = torch.randn(2, 32)
    fleet = torch.randn(2, 12)  # 3 features × 4 vehicle types
    out = vs(graph, fleet)
    assert out.shape == (2, 4)


def test_node_selector_masks_correctly() -> None:
    ns = NodeSelector(embed_dim=32, n_heads=4)
    node = torch.randn(2, 8, 32)
    graph = torch.randn(2, 32)
    rc = torch.rand(2)
    ct = torch.rand(2) * 100
    last = torch.randn(2, 32)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, 3] = True
    out = ns(node, graph, rc, ct, last, mask)
    assert out.shape == (2, 8)
    # masked positions should be -inf
    assert torch.isinf(out[:, 3]).all()
    assert (out[:, 3] < 0).all()


def test_hierarchical_decoder_constructs() -> None:
    hd = HierarchicalDecoder(embed_dim=16, n_vehicle_types=3, n_heads=4)
    assert isinstance(hd.vehicle_selector, VehicleSelector)
    assert isinstance(hd.node_selector, NodeSelector)
