"""Unit tests for geometry/coordinate_system.py."""

from __future__ import annotations

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

PI = torch.pi


def internal_bounds(
    coord_system: CoordSystem, lb: dict[str, float], ub: dict[str, float]
) -> tuple[dict[str, float], dict[str, float]]:
    """Internal bounds from geometric bounds, as plain floats."""
    device = coord_system.device
    internal_lb, internal_ub = coord_system.internal_bounds_from_geometric(
        {label: torch.tensor(value, device=device) for label, value in lb.items()},
        {label: torch.tensor(value, device=device) for label, value in ub.items()},
    )
    return (
        {label: value.item() for label, value in internal_lb.items()},
        {label: value.item() for label, value in internal_ub.items()},
    )


class TestConstruction:
    def test_axes_are_sorted_into_standard_order(self, device: torch.device) -> None:
        cartesian = CartesianCoords(coord_axes={"y", "x"}, device=device)
        spherical = SphericalCoords(coord_axes={"phi", "theta", "r"}, device=device)

        assert cartesian.axes_labels_selected == ("x", "y")
        assert spherical.axes_labels_selected == ("r", "theta", "phi")

    def test_no_axes_raises(self, device: torch.device) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            CartesianCoords(coord_axes=set(), device=device)

    def test_unknown_axis_raises(self, device: torch.device) -> None:
        with pytest.raises(ValueError, match="Invalid axes"):
            CartesianCoords(coord_axes={"r"}, device=device)

    def test_axes_with_a_simpler_equivalent_raise(self, device: torch.device) -> None:
        # e.g. cylindrical (r, z) is the same as Cartesian (x, y)
        with pytest.raises(ValueError, match="consistent coordinate system"):
            CartesianCoords(coord_axes={"z"}, device=device)
        with pytest.raises(ValueError, match="consistent coordinate system"):
            CylindricalCoords(coord_axes={"r", "z"}, device=device)
        with pytest.raises(ValueError, match="consistent coordinate system"):
            SphericalCoords(coord_axes={"r", "phi"}, device=device)

    def test_missing_required_axis_raises(self, device: torch.device) -> None:
        with pytest.raises(ValueError, match="Missing required axis"):
            CylindricalCoords(coord_axes={"theta", "z"}, device=device)

    def test_required_axis_can_be_fixed(self, device: torch.device) -> None:
        coord_system = CylindricalCoords(
            coord_axes={"theta", "z"},
            device=device,
            fixed_axes={"r": 1.0},
        )
        assert coord_system.axes_labels_selected == ("theta", "z")
        assert coord_system.fixed_axes == {"r": 1.0}

    def test_axis_both_fixed_and_selected_raises(self, device: torch.device) -> None:
        with pytest.raises(ValueError, match="cannot appear in both"):
            CartesianCoords(coord_axes={"x"}, device=device, fixed_axes={"x": 0.0})

    def test_unknown_fixed_axis_raises(self, device: torch.device) -> None:
        with pytest.raises(ValueError, match="Invalid fixed axes"):
            CartesianCoords(coord_axes={"x"}, device=device, fixed_axes={"r": 0.0})

    def test_fixed_axis_outside_its_bounds_raises(self, device: torch.device) -> None:
        with pytest.raises(ValueError, match="must be in"):
            CylindricalCoords(
                coord_axes={"theta"}, device=device, fixed_axes={"r": -1.0}
            )

    def test_registry_has_every_coord_system(self) -> None:
        for label in CoordSystemLabels:
            assert issubclass(COORD_SYSTEM_REGISTRY[label], CoordSystem)


class TestInternalCoords:
    def test_internal_labels(
        self,
        cartesian_2d: CoordSystem,
        cylindrical_2d: CoordSystem,
        cylindrical_3d: CoordSystem,
        spherical_3d: CoordSystem,
        device: torch.device,
    ) -> None:
        spherical_2d = SphericalCoords(coord_axes={"r", "theta"}, device=device)

        assert cartesian_2d.internal_geometric_labels == ("x", "y")
        assert cylindrical_2d.internal_geometric_labels == ("x", "y")
        assert cylindrical_3d.internal_geometric_labels == ("x", "y", "z")
        assert spherical_2d.internal_geometric_labels == ("x", "z")
        assert spherical_3d.internal_geometric_labels == ("x", "y", "z")

    def test_cartesian_mapping_is_unchanged(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        x = torch.tensor([1.0, 2.0], device=device)
        y = torch.tensor([3.0, 4.0], device=device)

        internal = cartesian_2d.map_geometric_to_internal_geometric({"x": x, "y": y})

        assert torch.equal(internal["x"], x)
        assert torch.equal(internal["y"], y)

    def test_cylindrical_mapping(
        self, cylindrical_3d: CoordSystem, device: torch.device
    ) -> None:
        r = torch.tensor([1.0, 2.0], device=device)
        theta = torch.tensor([0.0, PI / 2], device=device)
        z = torch.tensor([1.0, -1.0], device=device)

        internal = cylindrical_3d.map_geometric_to_internal_geometric(
            {"r": r, "theta": theta, "z": z}
        )

        assert torch.allclose(
            internal["x"], torch.tensor([1.0, 0.0], device=device), atol=1e-6
        )
        assert torch.allclose(
            internal["y"], torch.tensor([0.0, 2.0], device=device), atol=1e-6
        )
        assert torch.equal(internal["z"], z)

    def test_spherical_mapping(
        self, spherical_3d: CoordSystem, device: torch.device
    ) -> None:
        r = torch.tensor([1.0, 2.0], device=device)
        theta = torch.tensor([PI / 4, PI / 3], device=device)
        phi = torch.tensor([0.0, PI / 2], device=device)

        internal = spherical_3d.map_geometric_to_internal_geometric(
            {"r": r, "theta": theta, "phi": phi}
        )

        assert torch.allclose(internal["x"], r * torch.sin(theta) * torch.cos(phi))
        assert torch.allclose(internal["y"], r * torch.sin(theta) * torch.sin(phi))
        assert torch.allclose(internal["z"], r * torch.cos(theta))

    def test_spherical_mapping_without_phi_is_in_the_x_z_plane(
        self, device: torch.device
    ) -> None:
        coord_system = SphericalCoords(coord_axes={"r", "theta"}, device=device)
        r = torch.tensor([1.0, 2.0], device=device)
        theta = torch.tensor([PI / 4, PI / 3], device=device)

        internal = coord_system.map_geometric_to_internal_geometric(
            {"r": r, "theta": theta}
        )

        assert set(internal) == {"x", "z"}
        assert torch.allclose(internal["x"], r * torch.sin(theta))
        assert torch.allclose(internal["z"], r * torch.cos(theta))


class TestMetricTensor:
    def test_cartesian(self, cartesian_2d: CoordSystem, device: torch.device) -> None:
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        y = torch.tensor([4.0, 5.0, 6.0], device=device)
        v = torch.randn(3, 2, device=device)

        metric = cartesian_2d.metric_tensor(3, {"x": x, "y": y})

        assert torch.equal(metric.sqrt_det, torch.ones(3, device=device))
        assert torch.equal(metric.inv_metric_times_vec(v), v)

    def test_cylindrical(
        self, cylindrical_3d: CoordSystem, device: torch.device
    ) -> None:
        r = torch.tensor([1.0, 2.0, 3.0], device=device)
        theta = torch.tensor([0.3, 0.7, 1.2], device=device)
        z = torch.tensor([-1.0, 0.0, 1.0], device=device)
        v = torch.ones(3, 3, device=device)

        metric = cylindrical_3d.metric_tensor(3, {"r": r, "theta": theta, "z": z})

        # g = diag(1, r^2, 1)
        assert torch.allclose(metric.sqrt_det, r)
        expected = torch.stack([torch.ones_like(r), 1 / r**2, torch.ones_like(r)], 1)
        assert torch.allclose(metric.inv_metric_times_vec(v), expected)

    def test_cylindrical_with_fixed_r(self, device: torch.device) -> None:
        coord_system = CylindricalCoords(
            coord_axes={"theta", "z"},
            device=device,
            fixed_axes={"r": 3.0},
        )
        theta = torch.tensor([0.3, 0.7], device=device)
        z = torch.tensor([-1.0, 1.0], device=device)

        metric = coord_system.metric_tensor(2, {"theta": theta, "z": z})

        assert torch.allclose(metric.sqrt_det, torch.full((2,), 3.0, device=device))

    def test_spherical(self, spherical_3d: CoordSystem, device: torch.device) -> None:
        r = torch.tensor([1.0, 2.0, 3.0], device=device)
        theta = torch.tensor([0.2, 0.6, 1.0], device=device)
        phi = torch.tensor([0.5, 1.5, 2.5], device=device)
        v = torch.ones(3, 3, device=device)

        metric = spherical_3d.metric_tensor(3, {"r": r, "theta": theta, "phi": phi})

        # g = diag(1, r^2, r^2 sin^2(theta))
        assert torch.allclose(metric.sqrt_det, r**2 * torch.sin(theta))
        expected = torch.stack(
            [torch.ones_like(r), 1 / r**2, 1 / (r * torch.sin(theta)) ** 2], dim=1
        )
        assert torch.allclose(metric.inv_metric_times_vec(v), expected)


class TestVectorTransforms:
    def test_cylindrical_radial_vector_points_away_from_the_axis(
        self, cylindrical_2d: CoordSystem, device: torch.device
    ) -> None:
        geometric = {
            "r": torch.tensor([1.0, 1.0], device=device),
            "theta": torch.tensor([0.0, PI / 2], device=device),
        }
        e_r = torch.tensor([[1.0, 0.0], [1.0, 0.0]], device=device)

        cartesian = cylindrical_2d.transform_geometric_to_cartesian(e_r, geometric)

        expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device)
        assert torch.allclose(cartesian, expected, atol=1e-6)

    def test_spherical_round_trip(
        self, spherical_3d: CoordSystem, device: torch.device
    ) -> None:
        geometric = {
            "r": torch.tensor([1.0, 2.0], device=device),
            "theta": torch.tensor([0.4, 1.2], device=device),
            "phi": torch.tensor([0.5, 4.0], device=device),
        }
        v = torch.randn(2, 3, device=device)

        cartesian = spherical_3d.transform_geometric_to_cartesian(v, geometric)
        back = spherical_3d.transform_cartesian_to_geometric(cartesian, geometric)

        assert torch.allclose(back, v, atol=1e-6)


class TestInternalBounds:
    def test_cartesian_bounds_are_unchanged(self, cartesian_2d: CoordSystem) -> None:
        lb, ub = internal_bounds(
            cartesian_2d, {"x": -1.0, "y": -2.0}, {"x": 3.0, "y": 4.0}
        )
        assert lb == {"x": -1.0, "y": -2.0}
        assert ub == {"x": 3.0, "y": 4.0}

    def test_cylindrical_full_circle(self, cylindrical_2d: CoordSystem) -> None:
        lb, ub = internal_bounds(
            cylindrical_2d, {"r": 1.0, "theta": 0.0}, {"r": 2.0, "theta": 2 * PI}
        )
        assert lb == pytest.approx({"x": -2.0, "y": -2.0}, abs=1e-5)
        assert ub == pytest.approx({"x": 2.0, "y": 2.0}, abs=1e-5)

    def test_cylindrical_quarter_circle(self, cylindrical_2d: CoordSystem) -> None:
        lb, ub = internal_bounds(
            cylindrical_2d, {"r": 1.0, "theta": 0.0}, {"r": 2.0, "theta": PI / 2}
        )
        assert lb == pytest.approx({"x": 0.0, "y": 0.0}, abs=1e-5)
        assert ub == pytest.approx({"x": 2.0, "y": 2.0}, abs=1e-5)

    def test_cylindrical_theta_range_crossing_pi(
        self, cylindrical_2d: CoordSystem
    ) -> None:
        # cos(theta) reaches -1 at theta = pi, inside the range
        lb, ub = internal_bounds(
            cylindrical_2d,
            {"r": 1.0, "theta": PI / 2},
            {"r": 2.0, "theta": 3 * PI / 2},
        )
        assert lb == pytest.approx({"x": -2.0, "y": -2.0}, abs=1e-5)
        assert ub == pytest.approx({"x": 0.0, "y": 2.0}, abs=1e-5)

    def test_cylindrical_z_bounds_are_unchanged(
        self, cylindrical_3d: CoordSystem
    ) -> None:
        lb, ub = internal_bounds(
            cylindrical_3d,
            {"r": 0.0, "theta": 0.0, "z": -3.0},
            {"r": 1.0, "theta": 2 * PI, "z": 5.0},
        )
        assert lb == pytest.approx({"x": -1.0, "y": -1.0, "z": -3.0}, abs=1e-5)
        assert ub == pytest.approx({"x": 1.0, "y": 1.0, "z": 5.0}, abs=1e-5)

    def test_spherical_without_phi(self, device: torch.device) -> None:
        coord_system = SphericalCoords(coord_axes={"r", "theta"}, device=device)
        lb, ub = internal_bounds(
            coord_system, {"r": 0.0, "theta": 0.0}, {"r": 2.0, "theta": PI}
        )
        assert lb == pytest.approx({"x": 0.0, "z": -2.0}, abs=1e-5)
        assert ub == pytest.approx({"x": 2.0, "z": 2.0}, abs=1e-5)

    def test_spherical_full_sphere(self, spherical_3d: CoordSystem) -> None:
        lb, ub = internal_bounds(
            spherical_3d,
            {"r": 0.0, "theta": 0.0, "phi": 0.0},
            {"r": 2.0, "theta": PI, "phi": 2 * PI},
        )
        assert lb == pytest.approx({"x": -2.0, "y": -2.0, "z": -2.0}, abs=1e-5)
        assert ub == pytest.approx({"x": 2.0, "y": 2.0, "z": 2.0}, abs=1e-5)

    def test_spherical_first_octant(self, spherical_3d: CoordSystem) -> None:
        lb, ub = internal_bounds(
            spherical_3d,
            {"r": 0.0, "theta": 0.0, "phi": 0.0},
            {"r": 1.0, "theta": PI / 2, "phi": PI / 2},
        )
        assert lb == pytest.approx({"x": 0.0, "y": 0.0, "z": 0.0}, abs=1e-5)
        assert ub == pytest.approx({"x": 1.0, "y": 1.0, "z": 1.0}, abs=1e-5)

    def test_fixed_r_is_used(self, device: torch.device) -> None:
        coord_system = SphericalCoords(
            coord_axes={"theta"}, device=device, fixed_axes={"r": 2.0}
        )
        lb, ub = internal_bounds(coord_system, {"theta": 0.0}, {"theta": PI / 2})
        assert lb == pytest.approx({"x": 0.0, "z": 0.0}, abs=1e-5)
        assert ub == pytest.approx({"x": 2.0, "z": 2.0}, abs=1e-5)
