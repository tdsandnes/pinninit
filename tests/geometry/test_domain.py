"""
Unit tests for domain.py
"""

from __future__ import annotations

import pytest
import torch

from pinninit.geometry.coordinate_system import (
    CoordSystem,
)
from pinninit.geometry.domain import Domain


class TestDomainGeometricBounds:
    def test_defaults_to_metadata_bounds_when_none(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds=None,
            periodic_coords=None,
            device=device,
        )
        assert torch.isinf(domain.geometric_lb["x"])
        assert torch.isinf(domain.geometric_ub["x"])

    def test_user_bounds_stored_correctly(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (0.0, 1.0), "y": (-2.0, 2.0)},
            periodic_coords=None,
            device=device,
        )
        assert torch.allclose(domain.geometric_lb["x"], torch.tensor(0.0))
        assert torch.allclose(domain.geometric_ub["x"], torch.tensor(1.0))
        assert torch.allclose(domain.geometric_lb["y"], torch.tensor(-2.0))
        assert torch.allclose(domain.geometric_ub["y"], torch.tensor(2.0))

    def test_none_lb_falls_back_to_metadata(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (None, 1.0), "y": (0.0, 1.0)},
            periodic_coords=None,
            device=device,
        )
        assert torch.isinf(domain.geometric_lb["x"])
        assert torch.allclose(domain.geometric_ub["x"], torch.tensor(1.0))

    def test_none_ub_falls_back_to_metadata(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (0.0, None), "y": (0.0, 1.0)},
            periodic_coords=None,
            device=device,
        )
        assert torch.allclose(domain.geometric_lb["x"], torch.tensor(0.0))
        assert torch.isinf(domain.geometric_ub["x"])

    def test_lb_exceeds_metadata_raises(
        self, cylindrical_2d: CoordSystem, device: torch.device
    ) -> None:
        # r metadata lb is 0, so -1 should raise
        with pytest.raises(ValueError, match="Lower bound"):
            Domain(
                coord_system=cylindrical_2d,
                geometric_bounds={"r": (-1.0, 2.0), "theta": (0.0, 1.0)},
                periodic_coords=None,
                device=device,
            )

    def test_ub_exceeds_metadata_raises(
        self, cylindrical_2d: CoordSystem, device: torch.device
    ) -> None:
        # theta metadata ub is 2*pi
        with pytest.raises(ValueError, match="Upper bound"):
            Domain(
                coord_system=cylindrical_2d,
                geometric_bounds={"r": (0.0, 1.0), "theta": (0.0, 10.0)},
                periodic_coords=None,
                device=device,
            )

    def test_lb_equals_ub_does_not_raise(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (1.0, 1.0), "y": (0.0, 1.0)},
            periodic_coords=None,
            device=device,
        )

        torch.allclose(domain.geometric_lb["x"], domain.geometric_ub["x"])

    def test_lb_greater_than_ub_raises(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        with pytest.raises(ValueError, match="Lower bound must be <= upper bound"):
            Domain(
                coord_system=cartesian_2d,
                geometric_bounds={"x": (2.0, 1.0), "y": (0.0, 1.0)},
                periodic_coords=None,
                device=device,
            )

    def test_bounds_are_tensors(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
            periodic_coords=None,
            device=device,
        )
        for label in ["x", "y"]:
            assert isinstance(domain.geometric_lb[label], torch.Tensor)
            assert isinstance(domain.geometric_ub[label], torch.Tensor)

    def test_partial_bounds_override(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        # Only x provided, y falls back to metadata
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (0.0, 1.0)},
            periodic_coords=None,
            device=device,
        )
        assert torch.allclose(domain.geometric_lb["x"], torch.tensor(0.0))
        assert torch.isinf(domain.geometric_lb["y"])


class TestDomainInternalBounds:
    def test_cartesian_internal_matches_geometric(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (-1.0, 1.0), "y": (-2.0, 3.0)},
            periodic_coords=None,
            device=device,
        )
        assert torch.allclose(domain.internal_lb["x"], domain.geometric_lb["x"])
        assert torch.allclose(domain.internal_ub["x"], domain.geometric_ub["x"])
        assert torch.allclose(domain.internal_lb["y"], domain.geometric_lb["y"])
        assert torch.allclose(domain.internal_ub["y"], domain.geometric_ub["y"])

    def test_cylindrical_internal_bounds_full_circle(
        self, cylindrical_2d: CoordSystem, device: torch.device
    ) -> None:
        domain = Domain(
            coord_system=cylindrical_2d,
            geometric_bounds={"r": (0.0, 2.0), "theta": (0.0, 2 * torch.pi)},
            periodic_coords=None,
            device=device,
        )
        assert torch.allclose(domain.internal_lb["x"], torch.tensor(-2.0), atol=1e-5)
        assert torch.allclose(domain.internal_ub["x"], torch.tensor(2.0), atol=1e-5)
        assert torch.allclose(domain.internal_lb["y"], torch.tensor(-2.0), atol=1e-5)
        assert torch.allclose(domain.internal_ub["y"], torch.tensor(2.0), atol=1e-5)

    def test_spherical_internal_bounds_full_sphere(
        self, spherical_3d: CoordSystem, device: torch.device
    ) -> None:
        domain = Domain(
            coord_system=spherical_3d,
            geometric_bounds={
                "r": (0.0, 3.0),
                "theta": (0.0, torch.pi),
                "phi": (0.0, 2 * torch.pi),
            },
            periodic_coords=None,
            device=device,
        )
        for axis in ["x", "y", "z"]:
            assert torch.allclose(
                domain.internal_lb[axis], torch.tensor(-3.0), atol=1e-5
            )
            assert torch.allclose(
                domain.internal_ub[axis], torch.tensor(3.0), atol=1e-5
            )

    def test_internal_bounds_keys_match_internal_geometric_labels(
        self, cylindrical_3d: CoordSystem, device: torch.device
    ) -> None:
        domain = Domain(
            coord_system=cylindrical_3d,
            geometric_bounds={
                "r": (0.0, 1.0),
                "theta": (0.0, torch.pi),
                "z": (-1.0, 1.0),
            },
            periodic_coords=None,
            device=device,
        )
        expected_keys = set(cylindrical_3d.internal_geometric_labels)
        assert set(domain.internal_lb.keys()) == expected_keys
        assert set(domain.internal_ub.keys()) == expected_keys


class TestDomainContainsGeometric:
    def test_points_inside_return_true(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
            periodic_coords=None,
            device=device,
        )
        points = torch.tensor([[0.5, 0.5], [0.1, 0.9], [0.9, 0.1]], device=device)
        mask = domain.contains_geometric(points)
        assert mask.all()

    def test_points_outside_return_false(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
            periodic_coords=None,
            device=device,
        )
        points = torch.tensor([[1.5, 0.5], [-0.1, 0.5]], device=device)
        mask = domain.contains_geometric(points)
        assert not mask.any()

    def test_mixed_inside_outside(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
            periodic_coords=None,
            device=device,
        )
        points = torch.tensor([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5]], device=device)
        mask = domain.contains_geometric(points)
        assert mask[0].item() is True
        assert mask[1].item() is False
        assert mask[2].item() is False

    def test_boundary_points_included(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
            periodic_coords=None,
            device=device,
        )
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]], device=device)
        mask = domain.contains_geometric(points)
        assert mask.all()

    def test_returns_1d_bool_tensor(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
            periodic_coords=None,
            device=device,
        )
        points = torch.rand(10, 2, device=device)
        mask = domain.contains_geometric(points)
        assert mask.shape == (10,)
        assert mask.dtype == torch.bool


class TestDomainPeriodicCoords:
    def test_no_periodic_coords_empty_dict(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
            periodic_coords=None,
            device=device,
        )
        assert domain.periodic_coords == {}

    def test_period_inferred_from_bounds(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (0.0, 2.0), "y": (0.0, 1.0)},
            periodic_coords={"x"},
            device=device,
        )
        assert domain.periodic_coords["x"] == pytest.approx(2.0)

    def test_multiple_periodic_coords(
        self, cartesian_3d: CoordSystem, device: torch.device
    ) -> None:
        domain = Domain(
            coord_system=cartesian_3d,
            geometric_bounds={"x": (0.0, 2.0), "y": (0.0, 4.0), "z": (0.0, 1.0)},
            periodic_coords={"x", "y"},
            device=device,
        )
        assert domain.periodic_coords["x"] == pytest.approx(2.0)
        assert domain.periodic_coords["y"] == pytest.approx(4.0)
        assert "z" not in domain.periodic_coords

    def test_periodic_coords_non_cartesian_internal_raises(
        self, cylindrical_2d: CoordSystem, device: torch.device
    ) -> None:
        with pytest.raises(ValueError, match="Cartesian"):
            Domain(
                coord_system=cylindrical_2d,
                geometric_bounds={"r": (0.0, 1.0), "theta": (0.0, torch.pi)},
                periodic_coords={"r"},
                device=device,
            )

    def test_periodic_coord_not_in_geometric_labels_raises(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        with pytest.raises(ValueError, match="not a geometric axis"):
            Domain(
                coord_system=cartesian_2d,
                geometric_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
                periodic_coords={"z"},
                device=device,
            )

    def test_periodic_coords_stored_as_floats(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
            periodic_coords={"x"},
            device=device,
        )
        assert isinstance(domain.periodic_coords["x"], float)
