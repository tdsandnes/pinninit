"""Helpers shared by all test folders. Shared fixtures are in tests/conftest.py."""

from __future__ import annotations

import torch

from pinninit.data_structure.coords_dataclass import Coords, make_coords_dataclass
from pinninit.geometry.coordinate_system import CoordSystem

N_POINTS = 10


def make_coords(coord_system: CoordSystem, **coords: torch.Tensor) -> Coords:
    """Make a Coords instance with the given coordinate tensors,
    e.g. make_coords(cartesian_2d, x=x, y=y)."""
    device = next(iter(coords.values())).device
    coords_class = make_coords_dataclass(set(coords), coord_system, device)
    return coords_class(**coords)
