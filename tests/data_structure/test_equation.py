"""
Unit tests for data_structure/equation.py.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch

from pinninit.data_structure.coords_dataclass import (
    Coords,
    make_coords_dataclass,
)
from pinninit.data_structure.data_container import DataContainer
from pinninit.data_structure.equation import Equation
from pinninit.differential_equations.operators import Operators
from pinninit.geometry.coordinate_system import CoordSystem

from .fixtures import (
    N_POINTS,
    make_cartesian_2d,
    make_device,
    make_xy_2d,
)


@pytest.fixture
def device() -> torch.device:
    return make_device()


@pytest.fixture
def n_points() -> int:
    return N_POINTS


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


def build_container(
    cartesian_2d: CoordSystem,
    xy_2d: tuple[torch.Tensor, torch.Tensor],
) -> DataContainer:
    x, y = xy_2d
    coords = make_coords_cartesian_2d(cartesian_2d, x, y)
    return DataContainer(
        coords=coords,
        primary_field_label_to_rank={"u": 0, "v": 1},
        derived_field_label_to_rank={"w": 0, "q": 1},
        const_labels={"c"},
    )


class TestEquationConstruction:
    def test_input_labels_are_derived_from_signature_excluding_operators(self) -> None:
        def residual(
            x: torch.Tensor, u: torch.Tensor, operators: Operators
        ) -> torch.Tensor:
            return x + u

        equation = Equation(
            label="residual",
            function=residual,
            output_label_to_rank={"w": 0},
        )

        assert equation.input_labels == ("x", "u")
        assert equation.output_labels == ("w",)

    def test_missing_operators_argument_raises(self) -> None:
        def residual(x: torch.Tensor) -> torch.Tensor:
            return x

        with pytest.raises(ValueError, match="must accept 'operators'"):
            Equation(
                label="bad_residual",
                function=residual,
                output_label_to_rank={"w": 0},
            )

    def test_negative_output_rank_raises(self) -> None:
        def residual(x: torch.Tensor, operators: Operators) -> torch.Tensor:
            return x

        with pytest.raises(ValueError, match="must be >= 0"):
            Equation(
                label="bad_rank",
                function=residual,
                output_label_to_rank={"w": -1},
            )

    def test_output_labels_preserve_output_label_to_rank_insertion_order(self) -> None:
        def residual(
            x: torch.Tensor,
            operators: Operators,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return x, x

        equation = Equation(
            label="ordered_outputs",
            function=residual,
            output_label_to_rank={
                "second": 0,
                "first": 0,
            },
        )

        assert equation.output_labels == ("second", "first")


class TestEquationGatherInputs:
    def test_gathers_coordinates_fields_and_consts(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)
        coords = data.coords

        u = coords.get_coord("x") + coords.get_coord("y")
        data.set_field_from_tensor("u", u)
        data.set_const("c", torch.tensor(2.0, device=device))

        def residual(
            x: torch.Tensor,
            u: torch.Tensor,
            c: torch.Tensor,
            operators: Operators,
        ) -> torch.Tensor:
            return x + u + c

        equation = Equation(
            label="residual",
            function=residual,
            output_label_to_rank={"w": 0},
        )

        operators = Operators(coords)
        result = equation.evaluate(data, coords, operators)

        expected = coords.get_coord("x") + u + torch.tensor(2.0, device=device)
        assert set(result) == {"w"}
        assert torch.allclose(result["w"], expected)

    def test_unknown_input_label_raises(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)

        def residual(missing: torch.Tensor, operators: Operators) -> torch.Tensor:
            return missing

        equation = Equation(
            label="unknown_input",
            function=residual,
            output_label_to_rank={"w": 0},
        )

        operators = Operators(data.coords)
        with pytest.raises(KeyError, match="not found"):
            equation.evaluate(data, data.coords, operators)

    def test_uninitialised_field_input_raises(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)

        def residual(u: torch.Tensor, operators: Operators) -> torch.Tensor:
            return u

        equation = Equation(
            label="uninitialised_field",
            function=residual,
            output_label_to_rank={"w": 0},
        )

        operators = Operators(data.coords)
        with pytest.raises(ValueError, match="Field 'u' has not been initialised"):
            equation.evaluate(data, data.coords, operators)

    def test_uninitialised_const_input_raises(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)

        def residual(c: torch.Tensor, operators: Operators) -> torch.Tensor:
            return c

        equation = Equation(
            label="uninitialised_const",
            function=residual,
            output_label_to_rank={"w": 0},
        )

        operators = Operators(data.coords)
        with pytest.raises(ValueError, match="Constant 'c' has not been initialised"):
            equation.evaluate(data, data.coords, operators)

    def test_duplicate_label_between_coord_and_field_raises(self) -> None:
        def residual(x: torch.Tensor, operators: Operators) -> torch.Tensor:
            return x

        class MockCoords:
            coord_labels = {"x"}

            def get_coord(self, label: str) -> torch.Tensor:
                return torch.ones(10)

        class MockField:
            label = "x"

            def tensor(self, coords: Coords) -> torch.Tensor:
                return torch.zeros(10)

        class MockContainer:
            coords = MockCoords()
            coord_labels = coords.coord_labels
            primary_fields = {"x": MockField()}
            fields = primary_fields
            field_labels = set(primary_fields.keys())
            fields_active_coords = {"x": coords}
            derived_fields: dict[str, MockField] = {}
            consts: dict[str, torch.Tensor] = {}

        class MockOperators:
            pass

        equation = Equation(
            label="duplicate_label",
            function=residual,
            output_label_to_rank={"w": 0},
        )

        data = MockContainer()

        with pytest.raises(ValueError, match="Duplicate labels"):
            equation.evaluate(
                cast(DataContainer, data),
                cast(Coords, data.coords),
                cast(Operators, MockOperators()),
            )


class TestEquationEvaluate:
    def test_scalar_output_is_wrapped_in_result_dict(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)
        coords = data.coords

        def residual(
            x: torch.Tensor, y: torch.Tensor, operators: Operators
        ) -> torch.Tensor:
            return x**2 + y

        equation = Equation(
            label="scalar_equation",
            function=residual,
            output_label_to_rank={"w": 0},
        )

        operators = Operators(coords)
        result = equation.evaluate(data, coords, operators)

        assert set(result) == {"w"}
        assert result["w"].shape == (coords.n_points,)
        assert torch.allclose(
            result["w"],
            coords.get_coord("x") ** 2 + coords.get_coord("y"),
        )

    def test_tuple_outputs_are_mapped_to_output_labels_in_order(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)
        coords = data.coords

        def residual(
            x: torch.Tensor,
            y: torch.Tensor,
            operators: Operators,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            vector = torch.stack([x, y], dim=1)
            scalar = x + y
            return scalar, vector

        equation = Equation(
            label="multi_output",
            function=residual,
            output_label_to_rank={"w": 0, "q": 1},
        )

        operators = Operators(coords)
        result = equation.evaluate(data, coords, operators)

        assert set(result) == {"w", "q"}
        assert result["w"].shape == (coords.n_points,)
        assert result["q"].shape == (coords.n_points, 2)
        assert torch.allclose(
            result["w"], coords.get_coord("x") + coords.get_coord("y")
        )
        assert torch.allclose(
            result["q"],
            torch.stack([coords.get_coord("x"), coords.get_coord("y")], dim=1),
        )

    def test_wrong_number_of_outputs_raises(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)

        def residual(x: torch.Tensor, operators: Operators) -> torch.Tensor:
            return x

        equation = Equation(
            label="wrong_count",
            function=residual,
            output_label_to_rank={"w": 0, "q": 1},
        )

        operators = Operators(data.coords)
        with pytest.raises(ValueError, match="returned 1 outputs"):
            equation.evaluate(data, data.coords, operators)

    def test_non_tensor_output_raises(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)

        def residual(x: torch.Tensor, operators: Operators) -> float:
            return 1.0

        equation = Equation(
            label="non_tensor",
            function=residual,  # type: ignore[arg-type]
            output_label_to_rank={"w": 0},
        )

        operators = Operators(data.coords)
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            equation.evaluate(data, data.coords, operators)

    def test_wrong_output_shape_raises(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)

        def residual(x: torch.Tensor, operators: Operators) -> torch.Tensor:
            return torch.stack([x, x], dim=1)

        equation = Equation(
            label="wrong_shape",
            function=residual,
            output_label_to_rank={"w": 0},
        )

        operators = Operators(data.coords)
        with pytest.raises(ValueError, match="expected"):
            equation.evaluate(data, data.coords, operators)

    def test_tuple_outputs_are_assigned_using_output_label_order(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)

        def residual(
            x: torch.Tensor,
            operators: Operators,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return x + 1, x + 2

        equation = Equation(
            label="ordered_outputs",
            function=residual,
            output_label_to_rank={
                "second": 0,
                "first": 0,
            },
        )

        operators = Operators(data.coords)
        result = equation.evaluate(data, data.coords, operators)

        x = data.coords.get_coord("x")

        assert torch.allclose(result["second"], x + 1)
        assert torch.allclose(result["first"], x + 2)
