"""
Unit tests for differential_equations/operators.py.
"""

from __future__ import annotations

import pytest
import torch

from pinninit.data_structure.coords_dataclass import (
    CoordsProtocol,
    make_coords_dataclass,
)
from pinninit.differential_equations.operators import Operators
from pinninit.geometry.coordinate_system import CoordSystem
from tests.data_structure.fixtures import (
    N_POINTS,
    make_cartesian_1d,
    make_cartesian_2d,
    make_cylindrical_2d,
    make_device,
    make_r_theta_2d,
    make_x_1d,
    make_xy_2d,
)


@pytest.fixture  # type: ignore[misc]
def device() -> torch.device:
    return make_device()


@pytest.fixture  # type: ignore[misc]
def n_points() -> int:
    return N_POINTS


@pytest.fixture  # type: ignore[misc]
def cartesian_1d(device: torch.device) -> CoordSystem:
    return make_cartesian_1d(device)


@pytest.fixture  # type: ignore[misc]
def cartesian_2d(device: torch.device) -> CoordSystem:
    return make_cartesian_2d(device)


@pytest.fixture  # type: ignore[misc]
def cylindrical_2d(device: torch.device) -> CoordSystem:
    return make_cylindrical_2d(device)


@pytest.fixture  # type: ignore[misc]
def x_1d(device: torch.device) -> torch.Tensor:
    return make_x_1d(device)


@pytest.fixture  # type: ignore[misc]
def xy_2d(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return make_xy_2d(device)


@pytest.fixture  # type: ignore[misc]
def r_theta_2d(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return make_r_theta_2d(device)


# Helpers
def make_coords_cartesian_1d(
    coord_system: CoordSystem, x: torch.Tensor
) -> CoordsProtocol:
    Coords = make_coords_dataclass(
        coord_labels={"x"}, coord_system=coord_system, device=x.device
    )
    return Coords(x=x)


def make_coords_cartesian_2d(
    coord_system: CoordSystem, x: torch.Tensor, y: torch.Tensor
) -> CoordsProtocol:
    Coords = make_coords_dataclass(
        coord_labels={"x", "y"},
        coord_system=coord_system,
        device=x.device,
    )
    return Coords(x=x, y=y)


def make_coords_cylindrical_2d(
    coord_system: CoordSystem, r: torch.Tensor, theta: torch.Tensor
) -> CoordsProtocol:
    Coords = make_coords_dataclass(
        coord_labels={"r", "theta"},
        coord_system=coord_system,
        device=r.device,
    )
    return Coords(r=r, theta=theta)


class TestOperatorsConstruction:
    def test_initialises_metric_tensor_from_coords(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)
        operators = Operators(coords)

        assert operators.coords is coords
        assert operators.coord_system is cartesian_2d
        assert operators.geometric_labels == ("x", "y")
        assert operators.dim_geometric == 2
        assert operators.device == device
        assert operators.G.sqrt_det.shape == (coords.n_points,)


class TestCartesianDerivatives:
    def test_d_dcoord_returns_elementwise_derivative(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)
        operators = Operators(coords)

        x = coords.get_coord("x")
        y = coords.get_coord("y")
        f = x**2 + x * y

        dfdx = operators.d_dcoord(f, "x")
        dfdy = operators.d_dcoord(f, "y")

        assert torch.allclose(dfdx, 2.0 * x + y)
        assert torch.allclose(dfdy, x)

    def test_contravariant_grad_and_grad_match_cartesian_gradient(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)
        operators = Operators(coords)

        geometric = coords.geometric
        x = geometric[:, 0]
        y = geometric[:, 1]
        f = x**2 + 3.0 * y

        expected = torch.stack([2.0 * x, torch.full_like(y, 3.0)], dim=1)

        assert torch.allclose(operators.contravariant_grad(f), expected)
        assert torch.allclose(operators.grad(f), expected)

    def test_divergence_of_cartesian_vector_field(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)
        operators = Operators(coords)

        x = coords.get_coord("x")
        y = coords.get_coord("y")
        vector = torch.stack([x**2, y**3], dim=1)

        expected = 2.0 * x + 3.0 * y**2

        assert torch.allclose(operators.contravariant_div(vector), expected)
        assert torch.allclose(operators.div(vector), expected)

    def test_laplacian_of_quadratic_cartesian_scalar(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)
        operators = Operators(coords)

        geometric = coords.geometric
        x = geometric[:, 0]
        y = geometric[:, 1]
        f = x**2 + y**2

        expected = torch.full((coords.n_points,), 4.0, device=device)

        assert torch.allclose(operators.laplacian(f), expected)

    def test_vector_laplacian_applies_componentwise(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)
        operators = Operators(coords)

        geometric = coords.geometric
        x = geometric[:, 0]
        y = geometric[:, 1]
        vector = torch.stack([x**2 + y**2, x**3 + y**2], dim=1)

        expected = torch.stack(
            [
                torch.full_like(x, 4.0),
                6.0 * x + 2.0,
            ],
            dim=1,
        )

        assert torch.allclose(operators.vector_laplacian(vector), expected)

    def test_v_dot_grad_cartesian_vector_field(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)
        operators = Operators(coords)

        geometric = coords.geometric
        x = geometric[:, 0]
        y = geometric[:, 1]

        v1 = torch.stack([x, y], dim=1)
        v2 = torch.stack([x**2, y**2], dim=1)

        expected = torch.stack([2.0 * x**2, 2.0 * y**2], dim=1)

        assert torch.allclose(operators.v_dot_grad(v1, v2), expected)


class TestCylindricalDerivatives:
    def test_cylindrical_contravariant_and_physical_gradient(
        self,
        cylindrical_2d: CoordSystem,
        r_theta_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        r, theta = r_theta_2d
        coords = make_coords_cylindrical_2d(cylindrical_2d, r, theta)
        operators = Operators(coords)

        geometric = coords.geometric
        r = geometric[:, 0]
        theta = geometric[:, 1]
        f = r**2 + theta

        expected_contravariant = torch.stack(
            [
                2.0 * r,
                1.0 / r**2,
            ],
            dim=1,
        )
        expected_physical = torch.stack(
            [
                2.0 * r,
                1.0 / r,
            ],
            dim=1,
        )

        assert torch.allclose(operators.contravariant_grad(f), expected_contravariant)
        assert torch.allclose(operators.grad(f), expected_physical)
