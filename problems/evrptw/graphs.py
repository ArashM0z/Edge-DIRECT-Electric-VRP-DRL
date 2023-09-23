"""Time-window-overlap graph used by Edge-DIRECT's extra graph representation.

An edge (i, j) is included iff the time windows [tw_s_i, tw_e_i] and
[tw_s_j, tw_e_j] overlap *or* one is followed by the other with enough
travel-time slack to be feasible. This is the "extra graph representation"
contribution from the paper — it gives the encoder a structural signal
about which customer pairs are temporally compatible.
"""

from __future__ import annotations

import torch
from torch import Tensor


def build_tw_overlap_graph(tw_start: Tensor, tw_end: Tensor) -> Tensor:
    """Symmetric TW-overlap adjacency."""
    overlap = ~((tw_end.unsqueeze(-1) < tw_start.unsqueeze(-2))
                | (tw_end.unsqueeze(-2) < tw_start.unsqueeze(-1)))
    return overlap
