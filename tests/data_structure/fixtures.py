from __future__ import annotations

import torch

from pinninit.geometry.coordinate_system import (
    CartesianCoords,
    CylindricalCoords,
    SphericalCoords,
)
from pinninit.geometry.domain import Domain
from pinninit.io.input_parameters import EquationsParams, FieldsParams

# Constants
N_POINTS = 10
BATCH_SIZE = 4


# Device
def make_device() -> torch.device:
    return torch.device("cpu")


# Coordinate systems
def make_cartesian_1d(device: torch.device) -> CartesianCoords:
    return CartesianCoords(
        dim_geometry=1,
        coord_axes=("x",),
        fixed_axes=(),
        fixed_axes_values=(),
        device=device,
    )


def make_cartesian_2d(device: torch.device) -> CartesianCoords:
    return CartesianCoords(
        dim_geometry=2,
        coord_axes=("x", "y"),
        fixed_axes=(),
        fixed_axes_values=(),
        device=device,
    )


def make_cartesian_3d(device: torch.device) -> CartesianCoords:
    return CartesianCoords(
        dim_geometry=3,
        coord_axes=("x", "y", "z"),
        fixed_axes=(),
        fixed_axes_values=(),
        device=device,
    )


def make_cylindrical_2d(device: torch.device) -> CylindricalCoords:
    return CylindricalCoords(
        dim_geometry=2,
        coord_axes=("r", "theta"),
        fixed_axes=(),
        fixed_axes_values=(),
        device=device,
    )


def make_spherical_3d(device: torch.device) -> SphericalCoords:
    return SphericalCoords(
        dim_geometry=3,
        coord_axes=("r", "theta", "phi"),
        fixed_axes=(),
        fixed_axes_values=(),
        device=device,
    )


# Domains
def make_domain_1d_cartesian(
    coord_system: CartesianCoords, device: torch.device
) -> Domain:
    return Domain(
        coord_system=coord_system,
        geometric_bounds={"x": (0.0, 1.0)},
        periodic_coords=None,
        device=device,
    )


def make_domain_2d_cartesian(
    coord_system: CartesianCoords, device: torch.device
) -> Domain:
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


# FieldsParams / EquationsParams (bypasses YAML loading)
def make_fields_params(
    input_fields: list[dict] | None = None,
    output_fields: list[dict] | None = None,
    derivable_fields: list[dict] | None = None,
) -> FieldsParams:
    params = object.__new__(FieldsParams)
    params.input_fields = input_fields or []
    params.output_fields = output_fields or []
    params.derivable_fields = derivable_fields or []
    return params


def make_equations_params(equations: list[dict] | None = None) -> EquationsParams:
    params = object.__new__(EquationsParams)
    params.equations = equations or []
    return params


# PINN environment setup
# Inside tests/data_structure/fixtures.py


def make_pinn_environment(
    make_coords_dataclass, make_field_dataclass, device: torch.device
) -> tuple[object, object, torch.nn.Linear]:
    """
    Helper function to set up true coordinate and field instances for NN integration.
    """
    # Generate the required 2D Cartesian system object
    coord_system = make_cartesian_2d(device=device)

    # Create the true coordinate dataclass class constructor
    Coords = make_coords_dataclass(("x", "y"), device=device)

    # Supply coordinate tensors
    x = torch.tensor([1.5, 2.5, 3.5, 4.5], device=device, requires_grad=True)
    y = torch.tensor([5.5, 6.5, 7.5, 8.5], device=device, requires_grad=True)
    coords = Coords(coord_system=coord_system, x=x, y=y)

    # Setup a reference discrete field matching the parent coordinates
    f_x = torch.tensor([10.0, 20.0, 30.0, 40.0], device=device, requires_grad=True)
    f_y = torch.tensor([50.0, 60.0, 70.0, 80.0], device=device, requires_grad=True)
    Field = make_field_dataclass(rank=1, coords=coords, device=device)
    reference_field = Field(x=f_x, y=f_y)

    # Define a predictable single-layer NN (Linear transformation)
    nn_layer = torch.nn.Linear(2, 2, device=device)
    with torch.no_grad():
        nn_layer.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0]], device=device))
        nn_layer.bias.zero_()

    return coords, reference_field, nn_layer
