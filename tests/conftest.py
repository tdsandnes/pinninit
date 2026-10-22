"""Fixtures shared by all test folders. Shared helpers are in tests/helpers.py."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from pinninit.data_structure.batching import Batches
from pinninit.data_structure.coords import Coords
from pinninit.data_structure.data_container import DataContainer
from pinninit.geometry.coordinate_system import (
    CartesianCoords,
    CoordSystem,
    CylindricalCoords,
    SphericalCoords,
)
from pinninit.loss.evaluation_context import EvaluationContext
from pinninit.loss.loss_function import weighted_mse_loss

from .helpers import N_POINTS, make_coords


@pytest.fixture(autouse=True)
def torch_state() -> Iterator[None]:
    """Start each test from the same random seed."""
    dtype = torch.get_default_dtype()
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(dtype)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


# Coordinate systems
@pytest.fixture
def cartesian_2d(device: torch.device) -> CoordSystem:
    return CartesianCoords(coord_axes={"x", "y"}, device=device)


@pytest.fixture
def cartesian_3d(device: torch.device) -> CoordSystem:
    return CartesianCoords(coord_axes={"x", "y", "z"}, device=device)


@pytest.fixture
def cylindrical_2d(device: torch.device) -> CoordSystem:
    return CylindricalCoords(coord_axes={"r", "theta"}, device=device)


@pytest.fixture
def cylindrical_3d(device: torch.device) -> CoordSystem:
    return CylindricalCoords(coord_axes={"r", "theta", "z"}, device=device)


@pytest.fixture
def spherical_3d(device: torch.device) -> CoordSystem:
    return SphericalCoords(coord_axes={"r", "theta", "phi"}, device=device)


# Coords
@pytest.fixture
def coords_2d(cartesian_2d: CoordSystem, device: torch.device) -> Coords:
    """Cartesian (x, y) coords."""
    x = torch.linspace(0.1, 0.9, N_POINTS, device=device, requires_grad=True)
    y = torch.linspace(0.2, 0.8, N_POINTS, device=device, requires_grad=True)
    return make_coords(cartesian_2d, x=x, y=y)


@pytest.fixture
def coords_cylindrical(cylindrical_2d: CoordSystem, device: torch.device) -> Coords:
    """Cylindrical (r, theta) coords."""
    r = torch.linspace(0.5, 2.0, N_POINTS, device=device, requires_grad=True)
    theta = torch.linspace(0.1, 1.5, N_POINTS, device=device, requires_grad=True)
    return make_coords(cylindrical_2d, r=r, theta=theta)


@pytest.fixture
def coords_spherical(spherical_3d: CoordSystem, device: torch.device) -> Coords:
    """Spherical (r, theta, phi) coords."""
    r = torch.linspace(0.5, 2.0, N_POINTS, device=device, requires_grad=True)
    theta = torch.linspace(0.3, 2.8, N_POINTS, device=device, requires_grad=True)
    phi = torch.linspace(0.1, 5.9, N_POINTS, device=device, requires_grad=True)
    return make_coords(spherical_3d, r=r, theta=theta, phi=phi)


# Data containers
@pytest.fixture
def data(coords_2d: Coords) -> DataContainer:
    """Container on coords_2d with scalar and vector primary and derived
    fields, and a const."""
    return DataContainer(
        coords=coords_2d,
        primary_field_label_to_rank={"u": 0, "v": 1},
        derived_field_label_to_rank={"w": 0, "q": 1},
        const_labels={"c"},
    )


# Evaluation contexts
@pytest.fixture
def eval_context(data: DataContainer, coords_2d: Coords) -> EvaluationContext:
    """Evaluation context for the data container on coords_2d, with a weighted
    loss function and 3 batches."""
    return EvaluationContext(
        output_data=data,
        loss_function=weighted_mse_loss,
        batches=Batches(coords_2d, n_batches=3),
    )
