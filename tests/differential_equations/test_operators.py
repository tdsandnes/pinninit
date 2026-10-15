"""
Unit tests for differential_equations/operators.py.
"""

from __future__ import annotations

import torch

from pinninit.data_structure.coords_dataclass import Coords
from pinninit.differential_equations.operators import Operators
from pinninit.geometry.coordinate_system import CoordSystem


class TestOperatorsConstruction:
    def test_initialises_metric_tensor_from_coords(
        self,
        cartesian_2d: CoordSystem,
        coords_2d: Coords,
        device: torch.device,
    ) -> None:
        coords = coords_2d
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
        coords_2d: Coords,
    ) -> None:
        coords = coords_2d
        operators = Operators(coords)

        x = coords.get_coord("x")
        y = coords.get_coord("y")
        f = x**2 + x * y

        dfdx = operators.d_dcoord(f, "x")
        dfdy = operators.d_dcoord(f, "y")

        assert torch.allclose(dfdx, 2.0 * x + y)
        assert torch.allclose(dfdy, x)

    def test_d_dcoord_vector_field(
        self,
        coords_2d: Coords,
    ) -> None:
        coords = coords_2d
        operators = Operators(coords)

        x = coords.get_coord("x")
        y = coords.get_coord("y")

        vector = torch.stack(
            [
                x**2,
                x * y,
            ],
            dim=1,
        )

        expected = torch.stack(
            [
                2.0 * x,
                y,
            ],
            dim=1,
        )

        assert torch.allclose(
            operators.d_dcoord(vector, "x"),
            expected,
        )

    def test_d_dcoord_rank_two_tensor(
        self,
        coords_2d: Coords,
    ) -> None:
        coords = coords_2d
        operators = Operators(coords)

        x = coords.get_coord("x")
        y = coords.get_coord("y")

        tensor = torch.stack(
            [
                torch.stack([x**2, x * y], dim=1),
                torch.stack([y**2, x + y], dim=1),
            ],
            dim=1,
        )

        expected = torch.stack(
            [
                torch.stack([2.0 * x, y], dim=1),
                torch.stack([torch.zeros_like(x), torch.ones_like(x)], dim=1),
            ],
            dim=1,
        )

        assert torch.allclose(
            operators.d_dcoord(tensor, "x"),
            expected,
        )

    def test_contravariant_grad_and_grad_match_cartesian_gradient(
        self,
        coords_2d: Coords,
    ) -> None:
        coords = coords_2d
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
        coords_2d: Coords,
    ) -> None:
        coords = coords_2d
        operators = Operators(coords)

        x = coords.get_coord("x")
        y = coords.get_coord("y")
        vector = torch.stack([x**2, y**3], dim=1)

        expected = 2.0 * x + 3.0 * y**2

        assert torch.allclose(operators.contravariant_div(vector), expected)
        assert torch.allclose(operators.div(vector), expected)

    def test_laplacian_of_quadratic_cartesian_scalar(
        self,
        coords_2d: Coords,
        device: torch.device,
    ) -> None:
        coords = coords_2d
        operators = Operators(coords)

        geometric = coords.geometric
        x = geometric[:, 0]
        y = geometric[:, 1]
        f = x**2 + y**2

        expected = torch.full((coords.n_points,), 4.0, device=device)

        assert torch.allclose(operators.laplacian(f), expected)

    def test_laplacian_matches_div_grad(
        self,
        coords_2d: Coords,
    ) -> None:
        coords = coords_2d
        operators = Operators(coords)

        x = coords.get_coord("x")
        y = coords.get_coord("y")

        f = x**3 + x * y + y**2

        actual = operators.laplacian(f)
        expected = operators.div(operators.grad(f))

        assert torch.allclose(actual, expected)

    def test_vector_laplacian_applies_componentwise(
        self,
        coords_2d: Coords,
    ) -> None:
        coords = coords_2d
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
        coords_2d: Coords,
    ) -> None:
        coords = coords_2d
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
        coords_cylindrical: Coords,
    ) -> None:
        coords = coords_cylindrical
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

    def test_cylindrical_laplacian_matches_div_grad(
        self,
        coords_cylindrical: Coords,
    ) -> None:
        coords = coords_cylindrical
        operators = Operators(coords)

        r = coords.get_coord("r")
        theta = coords.get_coord("theta")

        f = r**2 * torch.sin(theta)

        actual = operators.laplacian(f)
        expected = operators.div(operators.grad(f))

        assert torch.allclose(actual, expected)

    def test_laplacian_log_r_is_zero(
        self,
        coords_cylindrical: Coords,
    ) -> None:
        coords = coords_cylindrical
        operators = Operators(coords)

        r = coords.get_coord("r")

        f = torch.log(r)

        expected = torch.zeros_like(r)

        assert torch.allclose(operators.laplacian(f), expected)

    def test_v_dot_grad_recovers_centripetal_acceleration(
        self,
        coords_cylindrical: Coords,
    ) -> None:
        coords = coords_cylindrical
        operators = Operators(coords)

        r = coords.get_coord("r")

        v = torch.stack(
            [
                torch.zeros_like(r),  # v_r
                torch.ones_like(r),  # v_theta
            ],
            dim=1,
        )

        expected = torch.stack(
            [
                -1.0 / r,
                torch.zeros_like(r),
            ],
            dim=1,
        )

        actual = operators.v_dot_grad(v, v)

        assert torch.allclose(
            actual,
            expected,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_vector_laplacian_recovers_basis_vector_curvature(
        self,
        coords_cylindrical: Coords,
    ) -> None:
        coords = coords_cylindrical
        operators = Operators(coords)

        r = coords.get_coord("r")

        v = torch.stack(
            [
                torch.ones_like(r),  # e_r
                torch.zeros_like(r),
            ],
            dim=1,
        )

        expected = torch.stack(
            [
                -1.0 / r**2,
                torch.zeros_like(r),
            ],
            dim=1,
        )

        actual = operators.vector_laplacian(v)

        assert torch.allclose(
            actual,
            expected,
            atol=1e-6,
            rtol=1e-6,
        )


class TestBatchedCoordinates:
    def test_grad_matches_root_result_on_batch(
        self,
        coords_2d: Coords,
    ) -> None:
        coords = coords_2d
        root_ops = Operators(coords)
        root_x = coords.get_coord("x")
        root_y = coords.get_coord("y")
        f = root_x**2 + 3.0 * root_y
        expected = root_ops.grad(f)

        # Batch
        idx = torch.tensor([0, 2, 5, 7, 9], device=coords.device)
        batch_coords = coords.get_batch(idx)
        batch_x = batch_coords.get_coord("x")
        batch_y = batch_coords.get_coord("y")
        batch_f = batch_x**2 + 3.0 * batch_y
        batch_ops = Operators(batch_coords)

        # Evaluate grad
        actual = batch_ops.grad(batch_f)

        assert torch.allclose(actual, expected[idx])

    def test_divergence_matches_root_result_on_batch(
        self,
        coords_2d: Coords,
    ) -> None:
        coords = coords_2d
        root_ops = Operators(coords)
        root_x = coords.get_coord("x")
        root_y = coords.get_coord("y")
        vector = torch.stack([root_x**2, root_y**3], dim=1)
        expected = root_ops.div(vector)

        # Batch
        idx = torch.tensor([1, 4, 6, 8], device=coords.device)
        batch_coords = coords.get_batch(idx)
        batch_x = batch_coords.get_coord("x")
        batch_y = batch_coords.get_coord("y")
        batch_vector = torch.stack(
            [batch_x**2, batch_y**3],
            dim=1,
        )
        batch_ops = Operators(batch_coords)

        # Evaluate div
        actual = batch_ops.div(batch_vector)

        assert torch.allclose(actual, expected[idx])

    def test_laplacian_matches_root_result_on_batch(
        self,
        coords_2d: Coords,
    ) -> None:
        coords = coords_2d
        root_ops = Operators(coords)
        root_x = coords.get_coord("x")
        root_y = coords.get_coord("y")
        f = root_x**2 + root_y**2
        expected = root_ops.laplacian(f)

        # Batch
        idx = torch.tensor([0, 3, 6, 9], device=coords.device)
        batch_coords = coords.get_batch(idx)
        batch_x = batch_coords.get_coord("x")
        batch_y = batch_coords.get_coord("y")
        batch_f = batch_x**2 + batch_y**2
        batch_ops = Operators(batch_coords)

        # Evaluate laplacian
        actual = batch_ops.laplacian(batch_f)

        assert torch.allclose(actual, expected[idx])

    def test_nested_batch_grad_matches_root_result(
        self,
        coords_2d: Coords,
    ) -> None:
        coords = coords_2d
        root_ops = Operators(coords)
        root_x = coords.get_coord("x")
        root_y = coords.get_coord("y")
        f = root_x**2 + 3.0 * root_y
        expected = root_ops.grad(f)

        # batch 1
        idx1 = torch.tensor([0, 2, 4, 6, 8], device=coords.device)
        batch1 = coords.get_batch(idx1)

        # batch 2
        idx2 = torch.tensor([1, 3], device=coords.device)
        batch2 = batch1.get_batch(idx2)
        batch2_x = batch2.get_coord("x")
        batch2_y = batch2.get_coord("y")
        batch2_f = batch2_x**2 + 3.0 * batch2_y
        batch2_ops = Operators(batch2)

        # Evaluate grad on batch 2 using batch2-coord operators
        actual = batch2_ops.grad(batch2_f)

        # Compare with sliced root
        expected_indices = idx1[idx2]
        assert torch.allclose(actual, expected[expected_indices])


class TestCoordinateGraphStructure:
    def test_d_dcoord_of_internal_coordinate(
        self,
        coords_cylindrical: Coords,
    ) -> None:
        coords = coords_cylindrical
        r, theta = coords.get_coord("r"), coords.get_coord("theta")
        operators = Operators(coords)

        # Make linear combination of internal coords
        a = torch.tensor([1.0, 2.0], device=coords.device)
        f = (coords.internal * a).sum(dim=1)
        dx_dr = operators.d_dcoord(f, "r")
        dx_dtheta = operators.d_dcoord(f, "theta")

        # Analytic cylindrical mapping
        expected_dr = a[0] * torch.cos(theta) + a[1] * torch.sin(theta)
        expected_dtheta = a[0] * (-r * torch.sin(theta)) + a[1] * (r * torch.cos(theta))

        assert torch.allclose(dx_dr, expected_dr)
        assert torch.allclose(dx_dtheta, expected_dtheta)

    def test_grad_of_internal_coordinate(
        self,
        coords_cylindrical: Coords,
    ) -> None:
        coords = coords_cylindrical
        theta = coords.get_coord("theta")
        operators = Operators(coords)

        # Make linear combination of internal coords
        a = torch.tensor([1.0, 2.0], device=coords.device)
        f = (coords.internal * a).sum(dim=1)
        grad_f = operators.grad(f)
        expected = torch.stack(
            [
                a[0] * torch.cos(theta) + a[1] * torch.sin(theta),
                -a[0] * torch.sin(theta) + a[1] * torch.cos(theta),
            ],
            dim=1,
        )

        assert torch.allclose(grad_f, expected)

    def test_d_dcoord_of_geometric_coordinate(
        self,
        coords_cylindrical: Coords,
    ) -> None:
        coords = coords_cylindrical
        operators = Operators(coords)

        r_geom = coords.get_coord("r")

        derivative = operators.d_dcoord(r_geom, "r")

        assert torch.allclose(
            derivative,
            torch.ones_like(r_geom),
        )
