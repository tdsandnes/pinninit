"""Unit tests for geometry/domain.py."""

from __future__ import annotations

import math

import pytest
import torch

from pinninit.geometry.coordinate_system import CoordSystem
from pinninit.geometry.domain import Domain


class TestGeometricBounds:
    def test_bounds_default_to_the_coord_system_bounds(
        self, cylindrical_2d: CoordSystem
    ) -> None:
        domain = Domain(
            coord_system=cylindrical_2d,
        )

        assert domain.geometric_lb["r"].item() == 0.0
        assert domain.geometric_ub["r"].item() == math.inf
        assert domain.geometric_lb["theta"].item() == 0.0
        assert domain.geometric_ub["theta"].item() == pytest.approx(2 * math.pi)

    def test_internal_bounds_with_the_default_angle_bounds(
        self, cylindrical_2d: CoordSystem
    ) -> None:
        domain = Domain(coord_system=cylindrical_2d, geometric_bounds={"r": (1.0, 2.0)})

        for axis in ("x", "y"):
            assert domain.internal_lb[axis].item() == pytest.approx(-2.0)
            assert domain.internal_ub[axis].item() == pytest.approx(2.0)

    def test_given_bounds_are_used(self, cartesian_2d: CoordSystem) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (0.0, 1.0), "y": (-2.0, 2.0)},
        )

        assert domain.geometric_lb["x"].item() == 0.0
        assert domain.geometric_ub["x"].item() == 1.0
        assert domain.geometric_lb["y"].item() == -2.0
        assert domain.geometric_ub["y"].item() == 2.0

    def test_missing_bounds_fall_back_to_the_coord_system_bounds(
        self, cartesian_2d: CoordSystem
    ) -> None:
        # x has no lower bound and y has no bounds at all
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (None, 1.0)},
        )

        assert domain.geometric_lb["x"].item() == -math.inf
        assert domain.geometric_ub["x"].item() == 1.0
        assert domain.geometric_lb["y"].item() == -math.inf
        assert domain.geometric_ub["y"].item() == math.inf

    def test_equal_lower_and_upper_bounds_are_allowed(
        self, cartesian_2d: CoordSystem
    ) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (1.0, 1.0)},
        )

        assert domain.geometric_lb["x"].item() == domain.geometric_ub["x"].item()

    def test_bound_below_the_coord_system_bound_raises(
        self, cylindrical_2d: CoordSystem
    ) -> None:
        with pytest.raises(ValueError, match="Lower bound"):
            Domain(
                coord_system=cylindrical_2d,
                geometric_bounds={"r": (-1.0, 2.0)},
            )

    def test_bound_above_the_coord_system_bound_raises(
        self, cylindrical_2d: CoordSystem
    ) -> None:
        with pytest.raises(ValueError, match="Upper bound"):
            Domain(
                coord_system=cylindrical_2d,
                geometric_bounds={"theta": (0.0, 10.0)},
            )

    def test_lower_bound_above_upper_bound_raises(
        self, cartesian_2d: CoordSystem
    ) -> None:
        with pytest.raises(ValueError, match="Lower bound must be <= upper bound"):
            Domain(
                coord_system=cartesian_2d,
                geometric_bounds={"x": (2.0, 1.0)},
            )


class TestNonGeometricBounds:
    def test_time_bounds_are_included_in_the_internal_bounds(
        self, cartesian_2d: CoordSystem
    ) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (0.0, 1.0), "t": (0.0, 5.0)},
        )

        assert domain.non_geometric_lb["t"].item() == 0.0
        assert domain.non_geometric_ub["t"].item() == 5.0
        assert domain.internal_lb["t"].item() == 0.0
        assert domain.internal_ub["t"].item() == 5.0

    def test_time_without_both_bounds_raises(self, cartesian_2d: CoordSystem) -> None:
        with pytest.raises(ValueError, match="requires explicit min and max"):
            Domain(
                coord_system=cartesian_2d,
                geometric_bounds={"t": (0.0, None)},
            )


class TestInternalBounds:
    def test_internal_bounds_come_from_the_coord_system(
        self, cylindrical_3d: CoordSystem
    ) -> None:
        domain = Domain(
            coord_system=cylindrical_3d,
            geometric_bounds={
                "r": (0.0, 2.0),
                "theta": (0.0, 2 * math.pi),
                "z": (-1.0, 1.0),
            },
        )

        lb = {label: value.item() for label, value in domain.internal_lb.items()}
        ub = {label: value.item() for label, value in domain.internal_ub.items()}
        assert lb == pytest.approx({"x": -2.0, "y": -2.0, "z": -1.0}, abs=1e-5)
        assert ub == pytest.approx({"x": 2.0, "y": 2.0, "z": 1.0}, abs=1e-5)


class TestContainsGeometric:
    def test_points_inside_and_on_the_boundary_are_contained(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        )
        points = torch.tensor(
            [
                [0.5, 0.5],  # inside
                [0.0, 1.0],  # on the boundary
                [1.5, 0.5],  # outside in x
                [0.5, -0.1],  # outside in y
            ],
            device=device,
        )

        contained = domain.contains_geometric(points)

        expected = torch.tensor([True, True, False, False], device=device)
        assert torch.equal(contained, expected)


class TestPeriodicCoords:
    def test_no_periodic_coords(self, cartesian_2d: CoordSystem) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (0.0, 1.0)},
        )
        assert domain.periodic_coords == {}

    def test_period_is_the_length_of_the_domain(
        self, cartesian_3d: CoordSystem
    ) -> None:
        domain = Domain(
            coord_system=cartesian_3d,
            geometric_bounds={"x": (0.0, 2.0), "y": (-1.0, 3.0), "z": (0.0, 1.0)},
            periodic_coords={"x", "y"},
        )

        assert domain.periodic_coords == {"x": 2.0, "y": 4.0}

    def test_periodic_coords_need_a_cartesian_coord_system(
        self, cylindrical_2d: CoordSystem
    ) -> None:
        with pytest.raises(ValueError, match="only supported for Cartesian"):
            Domain(
                coord_system=cylindrical_2d,
                geometric_bounds={"r": (0.0, 1.0)},
                periodic_coords={"r"},
            )

    @pytest.mark.parametrize("x_bounds", [(None, None), (1.0, 1.0)])
    def test_periodic_coord_needs_a_finite_nonzero_period(
        self,
        cartesian_2d: CoordSystem,
        x_bounds: tuple[float | None, float | None],
    ) -> None:
        with pytest.raises(ValueError, match="needs finite bounds"):
            Domain(
                coord_system=cartesian_2d,
                geometric_bounds={"x": x_bounds},
                periodic_coords={"x"},
            )

    def test_unknown_periodic_coord_raises(self, cartesian_2d: CoordSystem) -> None:
        with pytest.raises(ValueError, match="not a geometric axis"):
            Domain(
                coord_system=cartesian_2d,
                geometric_bounds={"x": (0.0, 1.0)},
                periodic_coords={"z"},
            )
