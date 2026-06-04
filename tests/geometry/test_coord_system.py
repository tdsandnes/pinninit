"""
Unit tests for coordinate_system.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pinninit.geometry.coordinate_system import (
    COORD_SYSTEM_REGISTRY,
    CartesianCoords,
    CoordSystem,
    CoordSystemLabels,
    CylindricalCoords,
    SphericalCoords,
)
from pinninit.geometry.metric_tensor import DiagonalMetricTensor, IdentityMetricTensor


# Device
def make_device() -> torch.device:
    return torch.device("cpu")


def get_axis_index(coord_system: CoordSystem, axis: str) -> int:
    return coord_system.axes_labels_selected.index(axis)


@pytest.fixture  # type: ignore[misc]
def device() -> torch.device:
    return make_device()


def make_bounds(d: dict[str, float], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.tensor(v, device=device) for k, v in d.items()}


# Construction
class TestCartesianConstruction:
    def test_1d(self, device: torch.device) -> None:
        coord_system = CartesianCoords(dim_geometric=1, coord_axes={"x"}, device=device)
        assert coord_system.axes_labels_selected == ("x",)
        assert coord_system.dim_geometric == 1

    def test_2d(self, device: torch.device) -> None:
        coord_system = CartesianCoords(
            dim_geometric=2, coord_axes={"x", "y"}, device=device
        )
        assert coord_system.axes_labels_selected == ("x", "y")

    def test_3d(self, device: torch.device) -> None:
        coord_system = CartesianCoords(
            dim_geometric=3, coord_axes={"x", "y", "z"}, device=device
        )
        assert coord_system.axes_labels_selected == ("x", "y", "z")

    def test_axes_sorted_by_canonical_order(self, device: torch.device) -> None:
        coord_system = CartesianCoords(
            dim_geometric=2, coord_axes={"y", "x"}, device=device
        )
        assert coord_system.axes_labels_selected == ("x", "y")

    def test_invalid_dim_geometric_raises(self, device: torch.device) -> None:
        with pytest.raises(ValueError, match="dim_geometric"):
            CartesianCoords(dim_geometric=0, coord_axes={"x"}, device=device)

    def test_dim_geometric_too_large_raises(self, device: torch.device) -> None:
        with pytest.raises(ValueError, match="dim_geometric"):
            CartesianCoords(dim_geometric=4, coord_axes={"x", "y", "z"}, device=device)

    def test_invalid_axis_raises(self, device: torch.device) -> None:
        with pytest.raises(ValueError, match="Invalid axes"):
            CartesianCoords(dim_geometric=1, coord_axes={"r"}, device=device)

    def test_redirection_raises(self, device: torch.device) -> None:
        with pytest.raises(ValueError, match="consistent coordinate system"):
            CartesianCoords(dim_geometric=1, coord_axes={"z"}, device=device)

    def test_fixed_axes(self, device: torch.device) -> None:
        coord_system = CartesianCoords(
            dim_geometric=1,
            coord_axes={"x"},
            device=device,
            fixed_axes={"y": 0.0},
        )
        assert coord_system.fixed_axes == {"y": 0.0}

    def test_fixed_axes_overlap_raises(self, device: torch.device) -> None:
        with pytest.raises(ValueError, match="cannot appear in both"):
            CartesianCoords(
                dim_geometric=1,
                coord_axes={"x"},
                device=device,
                fixed_axes={"x": 0.0},
            )

    def test_fixed_axes_invalid_label_raises(self, device: torch.device) -> None:
        with pytest.raises(ValueError, match="Invalid fixed axes"):
            CartesianCoords(
                dim_geometric=1,
                coord_axes={"x"},
                device=device,
                fixed_axes={"r": 0.0},
            )


class TestCylindricalConstruction:
    def test_2d_r_theta(self, device: torch.device) -> None:
        coord_system = CylindricalCoords(
            dim_geometric=2, coord_axes={"r", "theta"}, device=device
        )
        assert coord_system.axes_labels_selected == ("r", "theta")

    def test_3d(self, device: torch.device) -> None:
        coord_system = CylindricalCoords(
            dim_geometric=3, coord_axes={"r", "theta", "z"}, device=device
        )
        assert coord_system.axes_labels_selected == ("r", "theta", "z")

    def test_missing_required_axis_raises(self, device: torch.device) -> None:
        with pytest.raises(ValueError, match="Missing required axis"):
            CylindricalCoords(dim_geometric=2, coord_axes={"theta", "z"}, device=device)

    def test_redirection_r_z_raises(self, device: torch.device) -> None:
        with pytest.raises(ValueError, match="consistent coordinate system"):
            CylindricalCoords(dim_geometric=2, coord_axes={"r", "z"}, device=device)

    def test_required_axis_as_fixed_is_ok(self, device: torch.device) -> None:
        coord_system = CylindricalCoords(
            dim_geometric=2,
            coord_axes={"theta", "z"},
            device=device,
            fixed_axes={"r": 1.0},
        )
        assert coord_system.fixed_axes == {"r": 1.0}


class TestSphericalConstruction:
    def test_2d_r_theta(self, device: torch.device) -> None:
        coord_system = SphericalCoords(
            dim_geometric=2, coord_axes={"r", "theta"}, device=device
        )
        assert coord_system.axes_labels_selected == ("r", "theta")

    def test_3d(self, device: torch.device) -> None:
        coord_system = SphericalCoords(
            dim_geometric=3, coord_axes={"r", "theta", "phi"}, device=device
        )
        assert coord_system.axes_labels_selected == ("r", "theta", "phi")

    def test_missing_required_axes_raises(self, device: torch.device) -> None:
        with pytest.raises(ValueError, match="Missing required axis"):
            SphericalCoords(dim_geometric=2, coord_axes={"theta", "phi"}, device=device)

    def test_redirection_r_phi_raises(self, device: torch.device) -> None:
        with pytest.raises(ValueError, match="consistent coordinate system"):
            SphericalCoords(dim_geometric=2, coord_axes={"r", "phi"}, device=device)

    def test_required_axis_as_fixed_is_ok(self, device: torch.device) -> None:
        coord_system = SphericalCoords(
            dim_geometric=2,
            coord_axes={"theta", "phi"},
            device=device,
            fixed_axes={"r": 1.0},
        )
        assert coord_system.fixed_axes == {"r": 1.0}


# Registry
class TestCoordSystemRegistry:
    @pytest.mark.parametrize("label", list(CoordSystemLabels))  # type: ignore[misc]
    def test_registry_contains_all_labels(self, label: CoordSystemLabels) -> None:
        assert label in COORD_SYSTEM_REGISTRY

    @pytest.mark.parametrize("label", list(CoordSystemLabels))  # type: ignore[misc]
    def test_registry_values_are_coord_system_subclasses(
        self, label: CoordSystemLabels
    ) -> None:
        assert issubclass(COORD_SYSTEM_REGISTRY[label], CoordSystem)


# Internal geometric labels
class TestInternalGeometricLabels:
    def test_cartesian_internal_labels_match_geometric(
        self, device: torch.device
    ) -> None:
        coord_system = CartesianCoords(
            dim_geometric=2, coord_axes={"x", "y"}, device=device
        )
        assert coord_system.internal_geometric_labels == ("x", "y")

    def test_cylindrical_2d_internal_labels(self, device: torch.device) -> None:
        coord_system = CylindricalCoords(
            dim_geometric=2, coord_axes={"r", "theta"}, device=device
        )
        assert coord_system.internal_geometric_labels == ("x", "y")

    def test_cylindrical_3d_internal_labels(self, device: torch.device) -> None:
        coord_system = CylindricalCoords(
            dim_geometric=3, coord_axes={"r", "theta", "z"}, device=device
        )
        assert coord_system.internal_geometric_labels == ("x", "y", "z")

    def test_spherical_2d_internal_labels(self, device: torch.device) -> None:
        coord_system = SphericalCoords(
            dim_geometric=2, coord_axes={"r", "theta"}, device=device
        )
        assert coord_system.internal_geometric_labels == ("x", "z")

    def test_spherical_3d_internal_labels(self, device: torch.device) -> None:
        coord_system = SphericalCoords(
            dim_geometric=3, coord_axes={"r", "theta", "phi"}, device=device
        )
        assert coord_system.internal_geometric_labels == ("x", "y", "z")


# map_geometric_to_internal_geometric
class TestMapGeometricToInternalGeometric:
    def test_cartesian_is_identity(self, device: torch.device) -> None:
        coord_system = CartesianCoords(
            dim_geometric=2, coord_axes={"x", "y"}, device=device
        )
        x = torch.tensor([1.0, 2.0], device=device)
        y = torch.tensor([3.0, 4.0], device=device)
        result = coord_system.map_geometric_to_internal_geometric({"x": x, "y": y})
        assert torch.allclose(result["x"], x)
        assert torch.allclose(result["y"], y)

    def test_cylindrical_2d_values(self, device: torch.device) -> None:
        coord_system = CylindricalCoords(
            dim_geometric=2, coord_axes={"r", "theta"}, device=device
        )
        r = torch.tensor([1.0, 2.0], device=device)
        theta = torch.tensor([0.0, np.pi / 2], device=device)
        result = coord_system.map_geometric_to_internal_geometric(
            {"r": r, "theta": theta}
        )
        assert torch.allclose(result["x"], r * torch.cos(theta))
        assert torch.allclose(result["y"], r * torch.sin(theta))

    def test_cylindrical_3d_values(self, device: torch.device) -> None:
        coord_system = CylindricalCoords(
            dim_geometric=3, coord_axes={"r", "theta", "z"}, device=device
        )
        r = torch.tensor([1.0, 2.0], device=device)
        theta = torch.tensor([0.0, np.pi / 2], device=device)
        z = torch.tensor([1.0, -1.0], device=device)
        result = coord_system.map_geometric_to_internal_geometric(
            {"r": r, "theta": theta, "z": z}
        )
        assert torch.allclose(result["x"], r * torch.cos(theta))
        assert torch.allclose(result["y"], r * torch.sin(theta))
        assert torch.allclose(result["z"], z)

    def test_spherical_3d_values(self, device: torch.device) -> None:
        coord_system = SphericalCoords(
            dim_geometric=3, coord_axes={"r", "theta", "phi"}, device=device
        )
        r = torch.tensor([1.0, 2.0], device=device)
        theta = torch.tensor([np.pi / 4, np.pi / 3], device=device)
        phi = torch.tensor([0.0, np.pi / 2], device=device)
        result = coord_system.map_geometric_to_internal_geometric(
            {"r": r, "theta": theta, "phi": phi}
        )
        assert torch.allclose(result["x"], r * torch.sin(theta) * torch.cos(phi))
        assert torch.allclose(result["y"], r * torch.sin(theta) * torch.sin(phi))
        assert torch.allclose(result["z"], r * torch.cos(theta))

    def test_spherical_2d_no_phi(self, device: torch.device) -> None:
        coord_system = SphericalCoords(
            dim_geometric=2, coord_axes={"r", "theta"}, device=device
        )
        r = torch.tensor([1.0, 2.0], device=device)
        theta = torch.tensor([np.pi / 4, np.pi / 3], device=device)
        result = coord_system.map_geometric_to_internal_geometric(
            {"r": r, "theta": theta}
        )
        assert torch.allclose(result["x"], r * torch.sin(theta))
        assert torch.allclose(result["z"], r * torch.cos(theta))
        assert "y" not in result

    @pytest.mark.parametrize("label", list(CoordSystemLabels))  # type: ignore[misc]
    def test_output_keys_match_internal_geometric_labels(
        self, label: CoordSystemLabels, device: torch.device
    ) -> None:
        if label == CoordSystemLabels.CARTESIAN:
            coord_system = COORD_SYSTEM_REGISTRY[label](
                dim_geometric=2, coord_axes={"x", "y"}, device=device
            )
            coords = {
                "x": torch.tensor([1.0], device=device),
                "y": torch.tensor([1.0], device=device),
            }
        elif label == CoordSystemLabels.CYLINDRICAL:
            coord_system = COORD_SYSTEM_REGISTRY[label](
                dim_geometric=2, coord_axes={"r", "theta"}, device=device
            )
            coords = {
                "r": torch.tensor([1.0], device=device),
                "theta": torch.tensor([0.5], device=device),
            }
        elif label == CoordSystemLabels.SPHERICAL:
            coord_system = COORD_SYSTEM_REGISTRY[label](
                dim_geometric=3, coord_axes={"r", "theta", "phi"}, device=device
            )
            coords = {
                "r": torch.tensor([1.0], device=device),
                "theta": torch.tensor([0.5], device=device),
                "phi": torch.tensor([0.5], device=device),
            }
        result = coord_system.map_geometric_to_internal_geometric(coords)
        assert set(result.keys()) == set(coord_system.internal_geometric_labels)


class TestCartesianMetric:
    def test_cartesian_returns_identity_metric(self, device: torch.device) -> None:
        coord_system = CartesianCoords(
            dim_geometric=3, coord_axes={"x", "y", "z"}, device=device
        )
        n = 4
        coords = {
            "x": torch.ones(n, device=device),
            "y": torch.ones(n, device=device),
            "z": torch.ones(n, device=device),
        }
        metric = coord_system.metric_tensor(n, coords)
        assert isinstance(metric, IdentityMetricTensor)

    def test_inverse_metric_operator_form(self, device: torch.device) -> None:
        coord_system = CartesianCoords(
            dim_geometric=3, coord_axes={"x", "y", "z"}, device=device
        )

        n = 5
        coords = {
            "x": torch.randn(n, device=device),
            "y": torch.randn(n, device=device),
            "z": torch.randn(n, device=device),
        }

        metric_tensor = coord_system.metric_tensor(n, coords)

        v = torch.randn(n, 3, device=device)
        out = metric_tensor.inv_metric_times_vec(v)

        assert torch.allclose(out, v)

    def test_sqrt_det(self, device: torch.device) -> None:
        coord_system = CartesianCoords(
            dim_geometric=3, coord_axes={"x", "y", "z"}, device=device
        )

        n = 4
        coords = {
            "x": 2.1 * torch.ones(n, device=device),
            "y": 3.0 * torch.ones(n, device=device),
            "z": 5.7 * torch.ones(n, device=device),
        }

        metric_tensor = coord_system.metric_tensor(n, coords)

        assert torch.allclose(metric_tensor.sqrt_det, torch.ones(n, device=device))


class TestCylindricalMetric:
    def test_cylindrical_returns_diagonal_metric(self, device: torch.device) -> None:
        coord_system = CylindricalCoords(
            dim_geometric=3, coord_axes={"r", "theta", "z"}, device=device
        )
        n = 4
        coords = {
            "r": torch.ones(n, device=device),
            "theta": torch.zeros(n, device=device),
            "z": torch.zeros(n, device=device),
        }
        metric = coord_system.metric_tensor(n, coords)
        assert isinstance(metric, DiagonalMetricTensor)

    def test_inverse_metric_operator_form(self, device: torch.device) -> None:
        coord_system = CylindricalCoords(
            dim_geometric=3,
            coord_axes={"r", "theta", "z"},
            device=device,
        )

        n = 6
        r = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device=device)
        theta = torch.zeros(n, device=device)
        z = torch.linspace(0, 1, n, device=device)

        metric_tensor = coord_system.metric_tensor(n, {"r": r, "theta": theta, "z": z})

        v = torch.stack(
            [
                torch.full_like(r, 3.4),
                torch.full_like(r, 2.1),
                torch.full_like(r, 101.9),
            ],
            dim=1,
        )

        out = metric_tensor.inv_metric_times_vec(v)

        i_r = coord_system.axes_labels_selected.index("r")
        i_t = coord_system.axes_labels_selected.index("theta")
        i_z = coord_system.axes_labels_selected.index("z")

        assert torch.allclose(out[:, i_r], v[:, i_r])
        assert torch.allclose(out[:, i_t], v[:, i_t] / (r**2))
        assert torch.allclose(out[:, i_z], v[:, i_z])

    def test_sqrt_det(self, device: torch.device) -> None:
        coord_system = CylindricalCoords(
            dim_geometric=3, coord_axes={"r", "theta", "z"}, device=device
        )
        n = 5
        r = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
        theta = torch.tensor([0.3, 0.7, 1.2, 1.8, 2.5], device=device)
        z = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)
        metric = coord_system.metric_tensor(n, {"r": r, "theta": theta, "z": z})
        assert torch.allclose(metric.sqrt_det, r)

    def test_cylindrical_metric_with_fixed_r(self, device: torch.device) -> None:
        coord_system = CylindricalCoords(
            dim_geometric=2,
            coord_axes={"theta", "z"},
            device=device,
            fixed_axes={"r": 3.0},
        )
        n = 4
        theta = torch.tensor([0.3, 0.7, 1.2, 1.8], device=device)
        z = torch.tensor([-1.0, 0.0, 1.0, 2.0], device=device)
        metric = coord_system.metric_tensor(n, {"theta": theta, "z": z})
        expected_r = torch.full((n,), 3.0, device=device)
        assert torch.allclose(metric.sqrt_det, expected_r)


class TestSphericalMetric:
    def test_spherical_returns_diagonal_metric(self, device: torch.device) -> None:
        coord_system = SphericalCoords(
            dim_geometric=3, coord_axes={"r", "theta", "phi"}, device=device
        )
        n = 4
        coords = {
            "r": torch.ones(n, device=device),
            "theta": torch.full((n,), 0.5, device=device),
            "phi": torch.zeros(n, device=device),
        }
        metric = coord_system.metric_tensor(n, coords)
        assert isinstance(metric, DiagonalMetricTensor)

    def test_inverse_metric_operator_form(self, device: torch.device) -> None:
        coord_system = SphericalCoords(
            dim_geometric=3,
            coord_axes={"r", "theta", "phi"},
            device=device,
        )

        n = 6
        r = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device=device)
        theta = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0, 1.2], device=device)
        phi = torch.linspace(0, 1, n, device=device)

        metric_tensor = coord_system.metric_tensor(
            n, {"r": r, "theta": theta, "phi": phi}
        )

        v = torch.stack(
            [
                torch.full_like(r, 3.4),
                torch.full_like(r, 2.1),
                torch.full_like(r, 101.9),
            ],
            dim=1,
        )

        out = metric_tensor.inv_metric_times_vec(v)

        i_r = coord_system.axes_labels_selected.index("r")
        i_t = coord_system.axes_labels_selected.index("theta")
        i_p = coord_system.axes_labels_selected.index("phi")

        sin2 = torch.sin(theta) ** 2

        assert torch.allclose(out[:, i_r], v[:, i_r])
        assert torch.allclose(out[:, i_t], v[:, i_t] / (r**2))
        assert torch.allclose(out[:, i_p], v[:, i_p] / (r**2 * sin2))

    def test_sqrt_det(self, device: torch.device) -> None:
        coord_system = SphericalCoords(
            dim_geometric=3, coord_axes={"r", "theta", "phi"}, device=device
        )
        n = 5
        r = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
        theta = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0], device=device)
        phi = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5], device=device)
        metric = coord_system.metric_tensor(n, {"r": r, "theta": theta, "phi": phi})
        assert torch.allclose(metric.sqrt_det, r**2 * torch.sin(theta))


class TestCartesianInternalBounds:
    def test_passthrough(self, device: torch.device) -> None:
        coord_system = CartesianCoords(
            dim_geometric=2, coord_axes={"x", "y"}, device=device
        )
        lb = make_bounds({"x": -1.0, "y": -2.0}, device)
        ub = make_bounds({"x": 3.0, "y": 4.0}, device)
        out_lb, out_ub = coord_system.internal_bounds_from_geometric(lb, ub)
        assert torch.allclose(out_lb["x"], lb["x"])
        assert torch.allclose(out_lb["y"], lb["y"])
        assert torch.allclose(out_ub["x"], ub["x"])
        assert torch.allclose(out_ub["y"], ub["y"])

    def test_3d_passthrough(self, device: torch.device) -> None:
        coord_system = CartesianCoords(
            dim_geometric=3, coord_axes={"x", "y", "z"}, device=device
        )
        lb = make_bounds({"x": 0.0, "y": -1.0, "z": -5.0}, device)
        ub = make_bounds({"x": 1.0, "y": 1.0, "z": 5.0}, device)
        out_lb, out_ub = coord_system.internal_bounds_from_geometric(lb, ub)
        for axis in ["x", "y", "z"]:
            assert torch.allclose(out_lb[axis], lb[axis])
            assert torch.allclose(out_ub[axis], ub[axis])


class TestCylindricalInternalBounds:
    def test_full_circle_tight_r(self, device: torch.device) -> None:
        # theta spans full [0, 2pi]: x, y bounds should be [-r_ub, r_ub]
        coord_system = CylindricalCoords(
            dim_geometric=2, coord_axes={"r", "theta"}, device=device
        )
        lb = make_bounds({"r": 1.0, "theta": 0.0}, device)
        ub = make_bounds({"r": 2.0, "theta": 2 * torch.pi}, device)
        out_lb, out_ub = coord_system.internal_bounds_from_geometric(lb, ub)
        assert torch.allclose(out_lb["x"], torch.tensor(-2.0), atol=1e-5)
        assert torch.allclose(out_ub["x"], torch.tensor(2.0), atol=1e-5)
        assert torch.allclose(out_lb["y"], torch.tensor(-2.0), atol=1e-5)
        assert torch.allclose(out_ub["y"], torch.tensor(2.0), atol=1e-5)

    def test_first_quadrant_only(self, device: torch.device) -> None:
        # theta in [0, pi/2]: x = r*cos(theta) in [0, r_ub],
        # y = r*sin(theta) in [0, r_ub]
        coord_system = CylindricalCoords(
            dim_geometric=2, coord_axes={"r", "theta"}, device=device
        )
        lb = make_bounds({"r": 0.0, "theta": 0.0}, device)
        ub = make_bounds({"r": 3.0, "theta": torch.pi / 2}, device)
        out_lb, out_ub = coord_system.internal_bounds_from_geometric(lb, ub)
        assert torch.allclose(out_lb["x"], torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(out_ub["x"], torch.tensor(3.0), atol=1e-5)
        assert torch.allclose(out_lb["y"], torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(out_ub["y"], torch.tensor(3.0), atol=1e-5)

    def test_theta_crosses_pi(self, device: torch.device) -> None:
        # theta in [pi/2, 3pi/2]: cos hits -1 at pi, so x_lb = -r_ub
        coord_system = CylindricalCoords(
            dim_geometric=2, coord_axes={"r", "theta"}, device=device
        )
        lb = make_bounds({"r": 1.0, "theta": torch.pi / 2}, device)
        ub = make_bounds({"r": 2.0, "theta": 3 * torch.pi / 2}, device)
        out_lb, out_ub = coord_system.internal_bounds_from_geometric(lb, ub)
        assert torch.allclose(out_lb["x"], torch.tensor(-2.0), atol=1e-5)
        # cos(pi/2) = cos(3pi/2) = 0, so x_ub = 0
        assert torch.allclose(out_ub["x"], torch.tensor(0.0), atol=1e-5)
        # sin hits 1 at pi/2 and -1 at 3pi/2
        assert torch.allclose(out_lb["y"], torch.tensor(-2.0), atol=1e-5)
        assert torch.allclose(out_ub["y"], torch.tensor(2.0), atol=1e-5)

    def test_nonzero_r_lb_first_quadrant(self, device: torch.device) -> None:
        # r in [1, 2], theta in [0, pi/2]: x in [0, 2], y in [0, 2]
        # but minimum x is r_lb * cos(pi/2) = 0, minimum y is r_lb * sin(0) = 0
        coord_system = CylindricalCoords(
            dim_geometric=2, coord_axes={"r", "theta"}, device=device
        )
        lb = make_bounds({"r": 1.0, "theta": 0.0}, device)
        ub = make_bounds({"r": 2.0, "theta": torch.pi / 2}, device)
        out_lb, out_ub = coord_system.internal_bounds_from_geometric(lb, ub)
        assert torch.allclose(out_lb["x"], torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(out_ub["x"], torch.tensor(2.0), atol=1e-5)
        assert torch.allclose(out_lb["y"], torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(out_ub["y"], torch.tensor(2.0), atol=1e-5)

    def test_includes_z_when_selected(self, device: torch.device) -> None:
        coord_system = CylindricalCoords(
            dim_geometric=3, coord_axes={"r", "theta", "z"}, device=device
        )
        lb = make_bounds({"r": 0.0, "theta": 0.0, "z": -3.0}, device)
        ub = make_bounds({"r": 1.0, "theta": 2 * torch.pi, "z": 5.0}, device)
        out_lb, out_ub = coord_system.internal_bounds_from_geometric(lb, ub)
        assert "z" in out_lb and "z" in out_ub
        assert torch.allclose(out_lb["z"], torch.tensor(-3.0))
        assert torch.allclose(out_ub["z"], torch.tensor(5.0))

    def test_z_not_in_output_when_not_selected(self, device: torch.device) -> None:
        coord_system = CylindricalCoords(
            dim_geometric=2, coord_axes={"r", "theta"}, device=device
        )
        lb = make_bounds({"r": 0.0, "theta": 0.0}, device)
        ub = make_bounds({"r": 1.0, "theta": 2 * torch.pi}, device)
        out_lb, out_ub = coord_system.internal_bounds_from_geometric(lb, ub)
        assert "z" not in out_lb
        assert "z" not in out_ub


class TestSphericalInternalBounds:
    def test_no_phi_theta_spans_0_to_pi(self, device: torch.device) -> None:
        # theta in [0, pi]: sin hits max of 1 at pi/2, x_ub = r_ub
        # cos goes from 1 to -1, z in [-r_ub, r_ub]
        coord_system = SphericalCoords(
            dim_geometric=2, coord_axes={"r", "theta"}, device=device
        )
        lb = make_bounds({"r": 0.0, "theta": 0.0}, device)
        ub = make_bounds({"r": 2.0, "theta": torch.pi}, device)
        out_lb, out_ub = coord_system.internal_bounds_from_geometric(lb, ub)
        assert "y" not in out_lb
        assert torch.allclose(out_lb["x"], torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(out_ub["x"], torch.tensor(2.0), atol=1e-5)
        assert torch.allclose(out_lb["z"], torch.tensor(-2.0), atol=1e-5)
        assert torch.allclose(out_ub["z"], torch.tensor(2.0), atol=1e-5)

    def test_no_phi_theta_upper_hemisphere_only(self, device: torch.device) -> None:
        # theta in [0, pi/2]: sin in [0, 1], cos in [0, 1]
        coord_system = SphericalCoords(
            dim_geometric=2, coord_axes={"r", "theta"}, device=device
        )
        lb = make_bounds({"r": 1.0, "theta": 0.0}, device)
        ub = make_bounds({"r": 3.0, "theta": torch.pi / 2}, device)
        out_lb, out_ub = coord_system.internal_bounds_from_geometric(lb, ub)
        assert torch.allclose(out_lb["x"], torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(out_ub["x"], torch.tensor(3.0), atol=1e-5)
        assert torch.allclose(out_lb["z"], torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(out_ub["z"], torch.tensor(3.0), atol=1e-5)

    def test_with_phi_full_sphere(self, device: torch.device) -> None:
        # Full sphere: x, y, z all in [-r_ub, r_ub]
        coord_system = SphericalCoords(
            dim_geometric=3, coord_axes={"r", "theta", "phi"}, device=device
        )
        lb = make_bounds({"r": 0.0, "theta": 0.0, "phi": 0.0}, device)
        ub = make_bounds({"r": 2.0, "theta": torch.pi, "phi": 2 * torch.pi}, device)
        out_lb, out_ub = coord_system.internal_bounds_from_geometric(lb, ub)
        for axis in ["x", "y", "z"]:
            assert torch.allclose(out_lb[axis], torch.tensor(-2.0), atol=1e-5)
            assert torch.allclose(out_ub[axis], torch.tensor(2.0), atol=1e-5)

    def test_with_phi_first_octant(self, device: torch.device) -> None:
        # theta in [0, pi/2], phi in [0, pi/2]: all components >= 0
        coord_system = SphericalCoords(
            dim_geometric=3, coord_axes={"r", "theta", "phi"}, device=device
        )
        lb = make_bounds({"r": 0.0, "theta": 0.0, "phi": 0.0}, device)
        ub = make_bounds({"r": 1.0, "theta": torch.pi / 2, "phi": torch.pi / 2}, device)
        out_lb, out_ub = coord_system.internal_bounds_from_geometric(lb, ub)
        assert torch.allclose(out_lb["x"], torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(out_lb["y"], torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(out_lb["z"], torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(out_ub["x"], torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(out_ub["y"], torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(out_ub["z"], torch.tensor(1.0), atol=1e-5)

    def test_y_not_in_output_when_phi_not_selected(self, device: torch.device) -> None:
        coord_system = SphericalCoords(
            dim_geometric=2, coord_axes={"r", "theta"}, device=device
        )
        lb = make_bounds({"r": 0.0, "theta": 0.0}, device)
        ub = make_bounds({"r": 1.0, "theta": torch.pi}, device)
        out_lb, out_ub = coord_system.internal_bounds_from_geometric(lb, ub)
        assert "y" not in out_lb
        assert "y" not in out_ub

    def test_fixed_r_cylindrical(self, device: torch.device) -> None:
        # r is fixed, so bounds should use fixed value for r
        coord_system = SphericalCoords(
            dim_geometric=2,
            coord_axes={"theta"},
            device=device,
            fixed_axes={"r": 2.0},
        )
        lb = make_bounds({"theta": 0.0}, device)
        ub = make_bounds({"theta": torch.pi / 2}, device)
        out_lb, out_ub = coord_system.internal_bounds_from_geometric(lb, ub)
        # sin(theta) in [0, 1], r fixed at 2: x in [0, 2]
        assert torch.allclose(out_ub["x"], torch.tensor(2.0), atol=1e-5)
        # cos(theta) in [0, 1]: z in [0, 2]
        assert torch.allclose(out_ub["z"], torch.tensor(2.0), atol=1e-5)
