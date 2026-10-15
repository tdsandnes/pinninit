"""Unit tests for data_structure/equation.py."""

from __future__ import annotations

import pytest
import torch

from pinninit.data_structure.data_container import DataContainer
from pinninit.data_structure.equation import Equation
from pinninit.differential_equations.operators import Operators


def evaluate(equation: Equation, data: DataContainer) -> dict[str, torch.Tensor]:
    return equation.evaluate(data, data.coords, Operators(data.coords))


class TestConstruction:
    def test_labels(self) -> None:
        def residual(
            x: torch.Tensor, u: torch.Tensor, operators: Operators
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return x, u

        equation = Equation("residual", residual, {"second": 0, "first": 0})

        assert equation.input_labels == ("x", "u")
        assert equation.output_labels == ("second", "first")

    def test_missing_operators_argument_raises(self) -> None:
        def residual(x: torch.Tensor) -> torch.Tensor:
            return x

        with pytest.raises(ValueError, match="must accept 'operators'"):
            Equation("residual", residual, {"w": 0})

    def test_negative_output_rank_raises(self) -> None:
        def residual(x: torch.Tensor, operators: Operators) -> torch.Tensor:
            return x

        with pytest.raises(ValueError, match="must be >= 0"):
            Equation("residual", residual, {"w": -1})


class TestEvaluate:
    def test_gathers_coords_fields_and_consts(self, data: DataContainer) -> None:
        x = data.coords.get_coord("x")
        u = 2.0 * x
        data.set_field_from_tensor("u", u)
        data.set_const("c", torch.tensor(3.0))

        def residual(
            x: torch.Tensor, u: torch.Tensor, c: torch.Tensor, operators: Operators
        ) -> torch.Tensor:
            return x + u + c

        result = evaluate(Equation("residual", residual, {"w": 0}), data)

        assert torch.allclose(result["w"], x + u + 3.0)

    def test_outputs_are_matched_to_labels_in_order(self, data: DataContainer) -> None:
        def residual(
            x: torch.Tensor, y: torch.Tensor, operators: Operators
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return x + y, torch.stack([x, y], dim=1)

        result = evaluate(Equation("residual", residual, {"w": 0, "q": 1}), data)

        x, y = data.coords.get_coord("x"), data.coords.get_coord("y")
        assert torch.allclose(result["w"], x + y)
        assert torch.allclose(result["q"], torch.stack([x, y], dim=1))

    def test_wrong_number_of_outputs_raises(self, data: DataContainer) -> None:
        def residual(x: torch.Tensor, operators: Operators) -> torch.Tensor:
            return x

        equation = Equation("residual", residual, {"w": 0, "q": 1})
        with pytest.raises(ValueError, match="returned 1 outputs"):
            evaluate(equation, data)

    def test_output_that_is_not_a_tensor_raises(self, data: DataContainer) -> None:
        def residual(x: torch.Tensor, operators: Operators) -> torch.Tensor:
            return 1.0  # type: ignore[return-value]

        equation = Equation("residual", residual, {"w": 0})
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            evaluate(equation, data)

    def test_output_with_wrong_shape_raises(self, data: DataContainer) -> None:
        def residual(x: torch.Tensor, operators: Operators) -> torch.Tensor:
            return torch.stack([x, x], dim=1)

        equation = Equation("residual", residual, {"w": 0})
        with pytest.raises(ValueError, match="expected"):
            evaluate(equation, data)

    def test_unknown_input_raises(self, data: DataContainer) -> None:
        def residual(missing: torch.Tensor, operators: Operators) -> torch.Tensor:
            return missing

        equation = Equation("residual", residual, {"w": 0})
        with pytest.raises(KeyError, match="'missing' not found in data container"):
            evaluate(equation, data)
