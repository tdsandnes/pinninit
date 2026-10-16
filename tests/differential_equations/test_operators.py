"""Unit tests for differential_equations/operators.py."""

from __future__ import annotations

import pytest
import torch

from pinninit.data_structure.coords import Coords
from pinninit.differential_equations.operators import Operators
from pinninit.geometry.coordinate_system import CoordSystem

from ..helpers import N_POINTS, make_coords


class TestCartesian:
    def test_d_dcoord(self, coords_2d: Coords) -> None:
        operators = Operators(coords_2d)
        x, y = coords_2d.get_coord("x"), coords_2d.get_coord("y")
        f = x**2 + x * y

        assert torch.allclose(operators.d_dcoord(f, "x"), 2 * x + y)
        assert torch.allclose(operators.d_dcoord(f, "y"), x)

    def test_d_dcoord_of_a_function_independent_of_the_coord_is_zero(
        self, coords_2d: Coords
    ) -> None:
        operators = Operators(coords_2d)
        x = coords_2d.get_coord("x")

        assert torch.equal(operators.d_dcoord(x**2, "y"), torch.zeros_like(x))

    def test_d_dcoord_of_a_rank_2_tensor(self, coords_2d: Coords) -> None:
        operators = Operators(coords_2d)
        x, y = coords_2d.get_coord("x"), coords_2d.get_coord("y")
        tensor = torch.stack(
            [torch.stack([x**2, x * y], dim=1), torch.stack([y**2, x + y], dim=1)],
            dim=1,
        )

        zeros, ones = torch.zeros_like(x), torch.ones_like(x)
        expected = torch.stack(
            [torch.stack([2 * x, y], dim=1), torch.stack([zeros, ones], dim=1)],
            dim=1,
        )
        assert torch.allclose(operators.d_dcoord(tensor, "x"), expected)

    def test_grad(self, coords_2d: Coords) -> None:
        operators = Operators(coords_2d)
        x, y = coords_2d.get_coord("x"), coords_2d.get_coord("y")
        f = x**2 + 3 * y

        expected = torch.stack([2 * x, torch.full_like(y, 3.0)], dim=1)
        assert torch.allclose(operators.contravariant_grad(f), expected)
        assert torch.allclose(operators.grad(f), expected)

    def test_grad_of_a_vector_raises(self, coords_2d: Coords) -> None:
        operators = Operators(coords_2d)
        with pytest.raises(ValueError, match="expects a scalar field"):
            operators.grad(coords_2d.geometric)

    def test_tensor_not_computed_from_the_coords_raises(
        self, coords_2d: Coords, device: torch.device
    ) -> None:
        idx = torch.tensor([1, 4, 6], device=device)
        operators = Operators(coords_2d.get_batch(idx))
        # Computed from the full coords, so not from the batch coords
        f = coords_2d.get_coord("x")[idx] ** 2

        with pytest.raises(ValueError, match="computed from the operators' coords"):
            operators.d_dcoord(f, "x")
        with pytest.raises(ValueError, match="computed from the operators' coords"):
            operators.grad(f)
        with pytest.raises(ValueError, match="computed from the operators' coords"):
            operators.grad(f.detach())

    def test_div(self, coords_2d: Coords) -> None:
        operators = Operators(coords_2d)
        x, y = coords_2d.get_coord("x"), coords_2d.get_coord("y")
        v = torch.stack([x**2, y**3], dim=1)

        expected = 2 * x + 3 * y**2
        assert torch.allclose(operators.contravariant_div(v), expected)
        assert torch.allclose(operators.div(v), expected)

    def test_laplacian(self, coords_2d: Coords) -> None:
        operators = Operators(coords_2d)
        x, y = coords_2d.get_coord("x"), coords_2d.get_coord("y")

        laplacian = operators.laplacian(x**3 + y**2)

        assert torch.allclose(laplacian, 6 * x + 2)

    def test_vector_laplacian(self, coords_2d: Coords) -> None:
        operators = Operators(coords_2d)
        x, y = coords_2d.get_coord("x"), coords_2d.get_coord("y")
        v = torch.stack([x**2 + y**2, x**3 + y**2], dim=1)

        expected = torch.stack([torch.full_like(x, 4.0), 6 * x + 2], dim=1)
        assert torch.allclose(operators.vector_laplacian(v), expected)

    def test_v_dot_grad(self, coords_2d: Coords) -> None:
        operators = Operators(coords_2d)
        x, y = coords_2d.get_coord("x"), coords_2d.get_coord("y")
        v1 = torch.stack([x, y], dim=1)
        v2 = torch.stack([x**2, y**2], dim=1)

        expected = torch.stack([2 * x**2, 2 * y**2], dim=1)
        assert torch.allclose(operators.v_dot_grad(v1, v2), expected)


class TestCylindrical:
    def test_grad(self, coords_cylindrical: Coords) -> None:
        operators = Operators(coords_cylindrical)
        r = coords_cylindrical.get_coord("r")
        theta = coords_cylindrical.get_coord("theta")
        f = r**2 + theta

        contravariant = torch.stack([2 * r, 1 / r**2], dim=1)
        physical = torch.stack([2 * r, 1 / r], dim=1)
        assert torch.allclose(operators.contravariant_grad(f), contravariant)
        assert torch.allclose(operators.grad(f), physical)

    def test_laplacian_of_log_r_is_zero(self, coords_cylindrical: Coords) -> None:
        operators = Operators(coords_cylindrical)
        r = coords_cylindrical.get_coord("r")

        laplacian = operators.laplacian(torch.log(r))

        assert torch.allclose(laplacian, torch.zeros_like(r), atol=1e-6)

    def test_laplacian_is_div_grad(self, coords_cylindrical: Coords) -> None:
        operators = Operators(coords_cylindrical)
        r = coords_cylindrical.get_coord("r")
        theta = coords_cylindrical.get_coord("theta")
        f = r**2 * torch.sin(theta)

        laplacian = operators.laplacian(f)

        assert torch.allclose(laplacian, operators.div(operators.grad(f)))

    def test_v_dot_grad_gives_centripetal_acceleration(
        self, coords_cylindrical: Coords
    ) -> None:
        # Unit speed in theta: (v.grad)v = -e_r / r
        operators = Operators(coords_cylindrical)
        r = coords_cylindrical.get_coord("r")
        zeros, ones = torch.zeros_like(r), torch.ones_like(r)
        v = torch.stack([zeros, ones], dim=1)

        expected = torch.stack([-1 / r, zeros], dim=1)
        assert torch.allclose(operators.v_dot_grad(v, v), expected, atol=1e-6)

    def test_vector_laplacian_of_radial_unit_vector(
        self, coords_cylindrical: Coords
    ) -> None:
        # lap(e_r) = -e_r / r^2
        operators = Operators(coords_cylindrical)
        r = coords_cylindrical.get_coord("r")
        zeros, ones = torch.zeros_like(r), torch.ones_like(r)
        e_r = torch.stack([ones, zeros], dim=1)

        expected = torch.stack([-1 / r**2, zeros], dim=1)
        assert torch.allclose(operators.vector_laplacian(e_r), expected, atol=1e-6)

    def test_derivatives_of_internal_coords(self, coords_cylindrical: Coords) -> None:
        # Internal coords are x = r cos(theta), y = r sin(theta)
        operators = Operators(coords_cylindrical)
        r = coords_cylindrical.get_coord("r")
        theta = coords_cylindrical.get_coord("theta")
        x = coords_cylindrical.internal[:, 0]

        assert torch.allclose(operators.d_dcoord(x, "r"), torch.cos(theta))
        assert torch.allclose(operators.d_dcoord(x, "theta"), -r * torch.sin(theta))
        expected_grad = torch.stack([torch.cos(theta), -torch.sin(theta)], dim=1)
        assert torch.allclose(operators.grad(x), expected_grad)


class TestSpherical:
    def test_laplacian_of_1_over_r_is_zero(self, coords_spherical: Coords) -> None:
        operators = Operators(coords_spherical)
        r = coords_spherical.get_coord("r")

        laplacian = operators.laplacian(1 / r)

        assert torch.allclose(laplacian, torch.zeros_like(r), atol=1e-5)

    def test_div_of_radial_unit_vector(self, coords_spherical: Coords) -> None:
        # div(e_r) = 2 / r
        operators = Operators(coords_spherical)
        r = coords_spherical.get_coord("r")
        zeros, ones = torch.zeros_like(r), torch.ones_like(r)
        e_r = torch.stack([ones, zeros, zeros], dim=1)

        assert torch.allclose(operators.div(e_r), 2 / r)

    def test_vector_laplacian_of_radial_unit_vector(
        self, coords_spherical: Coords
    ) -> None:
        # lap(e_r) = -2 e_r / r^2
        operators = Operators(coords_spherical)
        r = coords_spherical.get_coord("r")
        zeros, ones = torch.zeros_like(r), torch.ones_like(r)
        e_r = torch.stack([ones, zeros, zeros], dim=1)

        expected = torch.stack([-2 / r**2, zeros, zeros], dim=1)
        assert torch.allclose(operators.vector_laplacian(e_r), expected, atol=1e-5)


class TestPrecision:
    def test_float64_coords_keep_float64_precision(
        self, cylindrical_2d: CoordSystem, device: torch.device
    ) -> None:
        # The default dtype is float32, so this catches anything computed at the
        # default dtype rather than the coords' dtype
        r = torch.linspace(0.5, 2.0, N_POINTS, dtype=torch.float64, device=device)
        theta = torch.linspace(0.1, 1.5, N_POINTS, dtype=torch.float64, device=device)
        coords = make_coords(
            cylindrical_2d, r=r.requires_grad_(), theta=theta.requires_grad_()
        )
        operators = Operators(coords)
        r = coords.get_coord("r")
        zeros, ones = torch.zeros_like(r), torch.ones_like(r)
        e_r = torch.stack([ones, zeros], dim=1)

        laplacian = operators.laplacian(torch.log(r))
        vector_laplacian = operators.vector_laplacian(e_r)

        assert torch.allclose(laplacian, zeros, atol=1e-12)
        expected = torch.stack([-1 / r**2, zeros], dim=1)
        assert torch.allclose(vector_laplacian, expected, rtol=0, atol=1e-12)

    def test_float64_cartesian_vector_operators(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        x = torch.linspace(0.1, 0.9, N_POINTS, dtype=torch.float64, device=device)
        y = torch.linspace(0.2, 0.8, N_POINTS, dtype=torch.float64, device=device)
        coords = make_coords(cartesian_2d, x=x.requires_grad_(), y=y.requires_grad_())
        operators = Operators(coords)
        x, y = coords.get_coord("x"), coords.get_coord("y")
        v = torch.stack([x**2, y**2], dim=1)

        result = operators.v_dot_grad(v, v)

        assert result.dtype == torch.float64
        assert torch.allclose(result, torch.stack([2 * x**3, 2 * y**3], dim=1))


class TestBatches:
    def test_results_on_a_batch_match_the_full_coords(
        self, coords_2d: Coords, device: torch.device
    ) -> None:
        def f(coords: Coords) -> torch.Tensor:
            return coords.get_coord("x") ** 2 + 3 * coords.get_coord("y") ** 3

        idx = torch.tensor([0, 2, 5, 7, 9], device=device)
        batch = coords_2d.get_batch(idx)
        full_operators = Operators(coords_2d)
        batch_operators = Operators(batch)

        full_grad = full_operators.grad(f(coords_2d))
        full_laplacian = full_operators.laplacian(f(coords_2d))
        assert torch.allclose(batch_operators.grad(f(batch)), full_grad[idx])
        assert torch.allclose(batch_operators.laplacian(f(batch)), full_laplacian[idx])
