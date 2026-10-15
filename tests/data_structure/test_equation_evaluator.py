"""
Unit tests for data_structure/equation_evaluator.py.
"""

from __future__ import annotations

import pytest
import torch

from pinninit.data_structure.coords_dataclass import (
    Coords,
    make_coords_dataclass,
)
from pinninit.data_structure.data_container import DataContainer
from pinninit.data_structure.equation import Equation
from pinninit.data_structure.equation_evaluator import EquationEvaluator
from pinninit.differential_equations.operators import Operators
from pinninit.geometry.coordinate_system import CoordSystem

from .fixtures import make_cartesian_2d, make_device, make_xy_2d


@pytest.fixture
def device() -> torch.device:
    return make_device()


@pytest.fixture
def cartesian_2d(device: torch.device) -> CoordSystem:
    return make_cartesian_2d(device)


@pytest.fixture
def xy_2d(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return make_xy_2d(device)


def make_coords_cartesian_2d(
    coord_system: CoordSystem, x: torch.Tensor, y: torch.Tensor
) -> Coords:
    coords_class = make_coords_dataclass(
        coord_labels={"x", "y"},
        coord_system=coord_system,
        device=x.device,
    )
    return coords_class(x=x, y=y)


class TestEquationEvaluatorSorting:
    def test_topological_sort_places_dependencies_first(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        def make_u(x: torch.Tensor, operators: Operators) -> torch.Tensor:
            return x + 1.0

        def make_v(u: torch.Tensor, operators: Operators) -> torch.Tensor:
            return u * 2.0

        eq_v = Equation("make_v", make_v, {"v": 0})
        eq_u = Equation("make_u", make_u, {"u": 0})

        data = DataContainer(
            coords=coords,
            primary_field_label_to_rank={},
            derived_field_label_to_rank={"u": 0, "v": 0},
            equations=[eq_v, eq_u],
        )

        sorted_labels = [
            equation.label for equation in data.equation_evaluator._sorted_equations
        ]

        assert sorted_labels == ["make_u", "make_v"]

    def test_disconnected_equations_are_all_included(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        def make_u(x: torch.Tensor, operators: Operators) -> torch.Tensor:
            return x + 1.0

        def make_v(y: torch.Tensor, operators: Operators) -> torch.Tensor:
            return y + 2.0

        eq_u = Equation("make_u", make_u, {"u": 0})
        eq_v = Equation("make_v", make_v, {"v": 0})

        data = DataContainer(
            coords=coords,
            primary_field_label_to_rank={},
            derived_field_label_to_rank={"u": 0, "v": 0},
            equations=[eq_u, eq_v],
        )

        sorted_labels = {
            equation.label for equation in data.equation_evaluator._sorted_equations
        }

        assert sorted_labels == {"make_u", "make_v"}

    def test_cycle_raises(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        def make_a(b: torch.Tensor, operators: Operators) -> torch.Tensor:
            return b + 1.0

        def make_b(a: torch.Tensor, operators: Operators) -> torch.Tensor:
            return a + 1.0

        eq_a = Equation("make_a", make_a, {"a": 0})
        eq_b = Equation("make_b", make_b, {"b": 0})

        with pytest.raises(ValueError, match="Cyclic equation dependency"):
            DataContainer(
                coords=coords,
                primary_field_label_to_rank={},
                derived_field_label_to_rank={"a": 0, "b": 0},
                equations=[eq_a, eq_b],
            )


class TestEquationEvaluatorEvaluate:
    def test_evaluate_writes_derived_fields_in_dependency_order(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        def make_u(
            x: torch.Tensor, y: torch.Tensor, operators: Operators
        ) -> torch.Tensor:
            return x + y

        def make_v(u: torch.Tensor, operators: Operators) -> torch.Tensor:
            return 3.0 * u

        eq_v = Equation("make_v", make_v, {"v": 0})
        eq_u = Equation("make_u", make_u, {"u": 0})

        data = DataContainer(
            coords=coords,
            primary_field_label_to_rank={},
            derived_field_label_to_rank={"u": 0, "v": 0},
            equations=[eq_v, eq_u],
        )

        data.evaluate_derived_fields()

        expected_u = coords.get_coord("x") + coords.get_coord("y")
        expected_v = 3.0 * expected_u

        assert torch.allclose(data.get_field_tensor("u"), expected_u)
        assert torch.allclose(data.get_field_tensor("v"), expected_v)

    def test_evaluate_on_batch_updates_only_batch_rows(
        self,
        cartesian_2d: CoordSystem,
        device: torch.device,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)
        batch_idx = torch.tensor([1, 3, 5], dtype=torch.long, device=device)
        batch_coords = coords.get_batch(batch_idx)

        def make_u(
            x: torch.Tensor, y: torch.Tensor, operators: Operators
        ) -> torch.Tensor:
            return x - y

        eq_u = Equation("make_u", make_u, {"u": 0})

        data = DataContainer(
            coords=coords,
            primary_field_label_to_rank={},
            derived_field_label_to_rank={"u": 0},
            equations=[eq_u],
        )

        data.evaluate_derived_fields(batch_coords)

        full_u = data.get_field_tensor("u", coords, enforce_ancestry=False)
        expected_batch = batch_coords.get_coord("x") - batch_coords.get_coord("y")

        assert torch.allclose(full_u[batch_idx], expected_batch)
        assert torch.allclose(
            full_u[torch.tensor([0, 2, 4, 6, 7], dtype=torch.long, device=device)],
            torch.zeros(5, device=device),
        )

    def test_uninitialised_primary_field_blocks_evaluation(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        def make_w(u: torch.Tensor, operators: Operators) -> torch.Tensor:
            return u + 1.0

        eq_w = Equation("make_w", make_w, {"w": 0})

        data = DataContainer(
            coords=coords,
            primary_field_label_to_rank={"u": 0},
            derived_field_label_to_rank={"w": 0},
            equations=[eq_w],
        )

        evaluator = EquationEvaluator(data)

        with pytest.raises(
            ValueError, match="Primary field 'u' has not been initialised"
        ):
            evaluator.evaluate(coords)
