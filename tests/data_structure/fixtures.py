from __future__ import annotations

import torch

from pinninit.data_structure.coords_dataclass import (
    Coords,
    make_coords_dataclass,
)
from pinninit.data_structure.field_dataclass import Field, make_field_dataclass
from pinninit.geometry.coordinate_system import (
    CartesianCoords,
    CoordSystem,
    CylindricalCoords,
    SphericalCoords,
)
from pinninit.geometry.domain import Domain

# Constants
N_POINTS = 10
BATCH_SIZE = 4


# Device
def make_device() -> torch.device:
    return torch.device("cpu")


# Coordinate systems
def make_cartesian_1d(device: torch.device) -> CoordSystem:
    return CartesianCoords(
        dim_geometric=1,
        coord_axes={
            "x",
        },
        fixed_axes=None,
        device=device,
    )


def make_cartesian_2d(device: torch.device) -> CoordSystem:
    return CartesianCoords(
        dim_geometric=2,
        coord_axes={"x", "y"},
        fixed_axes=None,
        device=device,
    )


def make_cartesian_3d(device: torch.device) -> CoordSystem:
    return CartesianCoords(
        dim_geometric=3,
        coord_axes={"x", "y", "z"},
        fixed_axes=None,
        device=device,
    )


def make_cylindrical_2d(device: torch.device) -> CoordSystem:
    return CylindricalCoords(
        dim_geometric=2,
        coord_axes={"r", "theta"},
        fixed_axes=None,
        device=device,
    )


def make_spherical_3d(device: torch.device) -> CoordSystem:
    return SphericalCoords(
        dim_geometric=3,
        coord_axes={"r", "theta", "phi"},
        fixed_axes=None,
        device=device,
    )


# Domains
def make_domain_1d_cartesian(coord_system: CoordSystem, device: torch.device) -> Domain:
    return Domain(
        coord_system=coord_system,
        geometric_bounds={"x": (0.0, 1.0)},
        periodic_coords=None,
        device=device,
    )


def make_domain_2d_cartesian(coord_system: CoordSystem, device: torch.device) -> Domain:
    return Domain(
        coord_system=coord_system,
        geometric_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        periodic_coords=None,
        device=device,
    )


# Raw coordinate tensors
def make_x_1d(device: torch.device, n: int = N_POINTS) -> torch.Tensor:
    return torch.linspace(0.1, 0.9, n, device=device, requires_grad=True)


def make_xy_2d(
    device: torch.device, n: int = N_POINTS
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(0.1, 0.9, n, device=device, requires_grad=True)
    y = torch.linspace(0.2, 0.8, n, device=device, requires_grad=True)
    return x, y


def make_xyt_2d(
    device: torch.device, n: int = N_POINTS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.linspace(0.1, 0.9, n, device=device, requires_grad=True)
    y = torch.linspace(0.2, 0.8, n, device=device, requires_grad=True)
    t = torch.linspace(0.3, 0.7, n, device=device, requires_grad=True)
    return x, y, t


def make_xyz_3d(
    device: torch.device, n: int = N_POINTS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.linspace(0.1, 0.9, n, device=device, requires_grad=True)
    y = torch.linspace(0.2, 0.8, n, device=device, requires_grad=True)
    z = torch.linspace(0.3, 0.7, n, device=device, requires_grad=True)
    return x, y, z


def make_r_theta_2d(
    device: torch.device, n: int = N_POINTS
) -> tuple[torch.Tensor, torch.Tensor]:
    r = torch.linspace(0.5, 2.0, n, device=device, requires_grad=True)
    theta = torch.linspace(0.1, 1.5, n, device=device, requires_grad=True)
    return r, theta


def make_r_theta_phi_3d(
    device: torch.device, n: int = N_POINTS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r = torch.linspace(0.5, 2.0, n, device=device, requires_grad=True)
    theta = torch.linspace(
        0.3, 2.8, n, device=device, requires_grad=True
    )  # Must be within (0, pi)
    phi = torch.linspace(
        0.1, 5.9, n, device=device, requires_grad=True
    )  # Must be within (0, 2pi)
    return r, theta, phi


# Batch indices
def make_batch_slice() -> slice:
    return slice(0, BATCH_SIZE)


def make_batch_tensor_indices(device: torch.device) -> torch.Tensor:
    return torch.tensor([0, 2, 4, 6], device=device)


# PINN environment setup
def make_pinn_environment(
    device: torch.device,
) -> tuple[Coords, Field, torch.nn.Linear]:
    """
    Helper function to set up coordinate and field instances for NN integration.
    """
    # Create the coordinate dataclass class
    coord_system = make_cartesian_2d(device=device)
    coords_class = make_coords_dataclass(
        {"x", "y"}, coord_system=coord_system, device=device
    )
    x = torch.tensor([1.5, 2.5, 3.5, 4.5], device=device, requires_grad=True)
    y = torch.tensor([5.5, 6.5, 7.5, 8.5], device=device, requires_grad=True)
    coords = coords_class(x=x, y=y)

    # Set up field
    f_x = torch.tensor([10.0, 20.0, 30.0, 40.0], device=device, requires_grad=True)
    f_y = torch.tensor([50.0, 60.0, 70.0, 80.0], device=device, requires_grad=True)
    field_class = make_field_dataclass(rank=1, coords=coords, device=device)
    reference_field = field_class(label="f", x=f_x, y=f_y)

    # Define a predictable single-layer NN (Linear transformation)
    nn_layer = torch.nn.Linear(2, 2, device=device)
    with torch.no_grad():
        nn_layer.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0]], device=device))
        nn_layer.bias.zero_()

    return coords, reference_field, nn_layer
