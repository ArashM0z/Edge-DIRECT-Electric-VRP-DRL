"""Non-linear CC+CV charging time model used in Edge-DIRECT."""

from __future__ import annotations

import torch
from torch import Tensor


def charging_time_minutes(
    soc_start: Tensor, soc_target: Tensor, station_kw: Tensor, battery_kwh: float,
) -> Tensor:
    """Two-piece linear CC+CV approximation of an EV charging curve."""
    cc_end = 0.8
    cc_part = (cc_end - soc_start).clamp(min=0) * battery_kwh * 60.0 / station_kw
    cv_part = (soc_target - cc_end).clamp(min=0) * battery_kwh * 60.0 / (station_kw * 0.4)
    direct_full = (soc_target - soc_start).clamp(min=0) * battery_kwh * 60.0 / station_kw
    return torch.where(soc_target <= cc_end, direct_full, cc_part + cv_part)


def energy_consumption_kwh(
    distance_km: Tensor, vehicle_kwh_per_km: Tensor, slope_factor: Tensor | None = None,
) -> Tensor:
    """Energy consumption with optional slope-based factor."""
    factor = slope_factor if slope_factor is not None else torch.ones_like(distance_km)
    return distance_km * vehicle_kwh_per_km * factor
