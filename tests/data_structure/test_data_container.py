"""Unit tests for data_structure/data_container.py."""

from __future__ import annotations

import pytest
import torch

from pinninit.data_structure.coords_dataclass import Coords
from pinninit.data_structure.data_container import DataContainer
from pinninit.data_structure.equation import Equation
from pinninit.differential_equations.operators import Operators
from pinninit.geometry.coordinate_system import CoordSystem

from ..helpers import N_POINTS, make_coords


class TestConstruction:
    def test_fields_and_consts_start_unset(self, data: DataContainer) -> None:
        assert data.field_labels == {"u", "v", "w", "q"}
        with pytest.raises(ValueError, match="'u' has not been initialised"):
            data.get_field_tensor("u")
        with pytest.raises(ValueError, match="'c' has not been initialised"):
            data.get_const("c")

    def test_unknown_field_raises(self, data: DataContainer) -> None:
        with pytest.raises(KeyError, match="'missing' not found"):
            data.get_field_tensor("missing")

    def test_unknown_const_raises(self, data: DataContainer) -> None:
        with pytest.raises(KeyError, match="'missing' not found"):
            data.set_const("missing", torch.tensor(1.0))


class TestFields:
    def test_set_field_from_tensor(self, data: DataContainer) -> None:
        values = 2.0 * data.coords.get_coord("x")

        data.set_field_from_tensor("u", values)

        assert torch.equal(data.get_field_tensor("u"), values)

    def test_set_field_from_component_tensor(self, data: DataContainer) -> None:
        x, y = data.coords.get_coord("x"), data.coords.get_coord("y")
        values = torch.stack([x, y], dim=1)

        data.set_field_from_component_tensor("v", values)

        assert torch.equal(data.get_field_component_tensor("v"), values)
        assert torch.equal(data.get_field_component("v", "y"), y)

    def test_set_field_from_component_dict(self, data: DataContainer) -> None:
        x, y = data.coords.get_coord("x"), data.coords.get_coord("y")

        data.set_field_from_component_dict("v", {"x": x, "y": y})

        assert torch.equal(data.get_field_tensor("v"), torch.stack([x, y], dim=1))

    def test_derived_field_can_only_be_set_with_allow_derived(
        self, data: DataContainer, device: torch.device
    ) -> None:
        values = torch.ones(N_POINTS, device=device)
        with pytest.raises(ValueError, match="derived field"):
            data.set_field_from_tensor("w", values)

        data.set_field_from_tensor("w", values, allow_derived=True)
        assert torch.equal(data.get_field_tensor("w"), values)

    def test_read_on_batch(self, data: DataContainer, device: torch.device) -> None:
        values = 2.0 * data.coords.get_coord("x")
        data.set_field_from_tensor("u", values)
        idx = torch.tensor([1, 4, 6], device=device)

        batch = data.coords.get_batch(idx)

        assert torch.equal(data.get_field_tensor("u", batch), values[idx])

    def test_write_on_batch_updates_only_batch_rows(
        self, data: DataContainer, device: torch.device
    ) -> None:
        data.set_field_from_tensor("u", torch.zeros(N_POINTS, device=device))
        idx = torch.tensor([1, 4, 6], device=device)
        batch_values = torch.tensor([1.0, 2.0, 3.0], device=device)

        data.set_field_from_tensor("u", batch_values, data.coords.get_batch(idx))

        expected = torch.zeros(N_POINTS, device=device)
        expected[idx] = batch_values
        assert torch.equal(data.get_field("u").tensor(data.coords), expected)

        # The other rows are now stale, so the full field can't be read
        with pytest.raises(ValueError, match="must be the active coords"):
            data.get_field_tensor("u")

    def test_setting_on_unrelated_coords_raises(
        self, data: DataContainer, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        unrelated = make_coords(
            cartesian_2d,
            x=torch.zeros(N_POINTS, device=device),
            y=torch.zeros(N_POINTS, device=device),
        )
        with pytest.raises(ValueError, match="must be the active coords"):
            data.set_field_from_tensor(
                "u", torch.zeros(N_POINTS, device=device), unrelated
            )


class TestConsts:
    def test_set_and_get_const(self, data: DataContainer) -> None:
        value = torch.tensor(3.0)
        data.set_const("c", value)
        assert data.get_const("c") is value

    def test_const_must_be_scalar(self, data: DataContainer) -> None:
        with pytest.raises(ValueError, match="must be a scalar tensor"):
            data.set_const("c", torch.ones(2))


class TestDerivedFieldEvaluation:
    def test_equations_evaluated_in_dependency_order(self, coords_2d: Coords) -> None:
        def make_b(a: torch.Tensor, operators: Operators) -> torch.Tensor:
            return 3.0 * a

        def make_a(
            x: torch.Tensor, y: torch.Tensor, operators: Operators
        ) -> torch.Tensor:
            return x + y

        # b is listed before a, which it depends on
        data = DataContainer(
            coords=coords_2d,
            primary_field_label_to_rank={},
            derived_field_label_to_rank={"a": 0, "b": 0},
            equations=[
                Equation("b", make_b, {"b": 0}),
                Equation("a", make_a, {"a": 0}),
            ],
        )
        data.evaluate_derived_fields()

        x, y = coords_2d.get_coord("x"), coords_2d.get_coord("y")
        assert torch.allclose(data.get_field_tensor("a"), x + y)
        assert torch.allclose(data.get_field_tensor("b"), 3.0 * (x + y))

    def test_cyclic_equations_raise(self, coords_2d: Coords) -> None:
        def make_a(b: torch.Tensor, operators: Operators) -> torch.Tensor:
            return b

        def make_b(a: torch.Tensor, operators: Operators) -> torch.Tensor:
            return a

        with pytest.raises(ValueError, match="Cyclic equation dependency"):
            DataContainer(
                coords=coords_2d,
                primary_field_label_to_rank={},
                derived_field_label_to_rank={"a": 0, "b": 0},
                equations=[
                    Equation("a", make_a, {"a": 0}),
                    Equation("b", make_b, {"b": 0}),
                ],
            )

    def test_evaluating_on_batch_updates_only_batch_rows(
        self, coords_2d: Coords, device: torch.device
    ) -> None:
        def make_a(x: torch.Tensor, operators: Operators) -> torch.Tensor:
            return x

        data = DataContainer(
            coords=coords_2d,
            primary_field_label_to_rank={},
            derived_field_label_to_rank={"a": 0},
            equations=[Equation("a", make_a, {"a": 0})],
        )
        idx = torch.tensor([1, 3, 5], device=device)

        data.evaluate_derived_fields(coords_2d.get_batch(idx))

        expected = torch.zeros(N_POINTS, device=device)
        expected[idx] = coords_2d.get_coord("x")[idx].detach()
        assert torch.equal(data.get_field("a").tensor(coords_2d).detach(), expected)

    def test_unset_primary_field_input_raises(self, coords_2d: Coords) -> None:
        def make_w(u: torch.Tensor, operators: Operators) -> torch.Tensor:
            return u

        data = DataContainer(
            coords=coords_2d,
            primary_field_label_to_rank={"u": 0},
            derived_field_label_to_rank={"w": 0},
            equations=[Equation("w", make_w, {"w": 0})],
        )
        with pytest.raises(ValueError, match="'u' has not been initialised"):
            data.evaluate_derived_fields()
