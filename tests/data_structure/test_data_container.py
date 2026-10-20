"""Unit tests for data_structure/data_container.py."""

from __future__ import annotations

import pytest
import torch

from pinninit.data_structure.coords import Coords
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

    def test_labels_must_be_unique_across_coords_fields_and_consts(
        self, coords_2d: Coords
    ) -> None:
        # They share one namespace as equation arguments
        with pytest.raises(ValueError, match=r"\['x'\] are used more than once"):
            DataContainer(coords=coords_2d, primary_field_label_to_rank={"x": 0})
        with pytest.raises(ValueError, match=r"\['u'\] are used more than once"):
            DataContainer(
                coords=coords_2d,
                primary_field_label_to_rank={"u": 0},
                const_labels={"u"},
            )
        with pytest.raises(ValueError, match=r"\['u'\] are used more than once"):
            DataContainer(
                coords=coords_2d,
                primary_field_label_to_rank={"u": 0},
                derived_field_label_to_rank={"u": 0},
            )


class TestFields:
    def test_set_and_get_fields(self, data: DataContainer) -> None:
        # The conversions between tensor layouts are tested in test_field.py
        x, y = data.coords.get_coord("x"), data.coords.get_coord("y")
        v = torch.stack([x, y], dim=1)

        data.set_field_from_tensor("u", 2.0 * x)
        data.set_field_from_component_tensor("v", v)

        assert torch.equal(data.get_field_tensor("u"), 2.0 * x)
        assert torch.equal(data.get_field_component_tensor("v"), v)

        data.set_field_from_component_dict("v", {"x": y, "y": x})

        assert torch.equal(data.get_field_tensor("v"), torch.stack([y, x], dim=1))

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

    def test_write_on_batch_stores_values_for_the_batch(
        self, data: DataContainer, device: torch.device
    ) -> None:
        data.set_field_from_tensor("u", torch.zeros(N_POINTS, device=device))
        batch = data.coords.get_batch(torch.tensor([1, 4, 6], device=device))
        batch_values = torch.tensor([1.0, 2.0, 3.0], device=device)

        data.set_field_from_tensor("u", batch_values, batch)

        assert data.get_field("u").coords is batch
        assert torch.equal(data.get_field_tensor("u", batch), batch_values)
        # The other points have no values now, so the full field can't be read
        with pytest.raises(ValueError, match="must be the container's coords"):
            data.get_field_tensor("u")

    def test_setting_on_unrelated_coords_raises(
        self, data: DataContainer, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        unrelated = make_coords(
            cartesian_2d,
            x=torch.zeros(N_POINTS, device=device),
            y=torch.zeros(N_POINTS, device=device),
        )
        with pytest.raises(ValueError, match="must be the container's coords"):
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

        with pytest.raises(ValueError, match="nodes are in a cycle"):
            DataContainer(
                coords=coords_2d,
                primary_field_label_to_rank={},
                derived_field_label_to_rank={"a": 0, "b": 0},
                equations=[
                    Equation("a", make_a, {"a": 0}),
                    Equation("b", make_b, {"b": 0}),
                ],
            )

    def test_evaluating_some_fields_only_evaluates_the_equations_they_need(
        self, coords_2d: Coords
    ) -> None:
        def make_a(x: torch.Tensor, operators: Operators) -> torch.Tensor:
            return x

        def make_b(a: torch.Tensor, operators: Operators) -> torch.Tensor:
            return 3.0 * a

        def make_c(y: torch.Tensor, operators: Operators) -> torch.Tensor:
            return y

        data = DataContainer(
            coords=coords_2d,
            primary_field_label_to_rank={},
            derived_field_label_to_rank={"a": 0, "b": 0, "c": 0},
            equations=[
                Equation("b", make_b, {"b": 0}),
                Equation("a", make_a, {"a": 0}),
                Equation("c", make_c, {"c": 0}),
            ],
        )
        data.evaluate_derived_fields(labels=["b"])

        x = coords_2d.get_coord("x")
        assert torch.allclose(data.get_field_tensor("b"), 3.0 * x)
        with pytest.raises(ValueError, match="'c' has not been initialised"):
            data.get_field_tensor("c")

    def test_evaluating_on_batch_stores_values_for_the_batch(
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
        batch = coords_2d.get_batch(torch.tensor([1, 3, 5], device=device))

        data.evaluate_derived_fields(batch)

        assert torch.equal(data.get_field_tensor("a", batch), batch.get_coord("x"))
        with pytest.raises(ValueError, match="must be the container's coords"):
            data.get_field_tensor("a")

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

    def test_equation_outputs_must_be_derived_fields(self, coords_2d: Coords) -> None:
        def make_u(x: torch.Tensor, operators: Operators) -> torch.Tensor:
            return x

        with pytest.raises(ValueError, match="not derived fields"):
            DataContainer(
                coords=coords_2d,
                primary_field_label_to_rank={"u": 0},
                equations=[Equation("u", make_u, {"u": 0})],
            )

    def test_equation_output_rank_must_match_the_field(self, coords_2d: Coords) -> None:
        def make_q(x: torch.Tensor, operators: Operators) -> torch.Tensor:
            return x

        with pytest.raises(ValueError, match="with rank 0, but the field has rank 1"):
            DataContainer(
                coords=coords_2d,
                primary_field_label_to_rank={},
                derived_field_label_to_rank={"q": 1},
                equations=[Equation("q", make_q, {"q": 0})],
            )

    def test_unknown_equation_input_raises(self, coords_2d: Coords) -> None:
        # Raised when the container is made, not when the equation is evaluated
        def make_w(missing: torch.Tensor, operators: Operators) -> torch.Tensor:
            return missing

        with pytest.raises(ValueError, match=r"takes \['missing'\]"):
            DataContainer(
                coords=coords_2d,
                primary_field_label_to_rank={},
                derived_field_label_to_rank={"w": 0},
                equations=[Equation("w", make_w, {"w": 0})],
            )

    def test_field_output_by_two_equations_raises(self, coords_2d: Coords) -> None:
        def make_a(x: torch.Tensor, operators: Operators) -> torch.Tensor:
            return x

        with pytest.raises(ValueError, match="output by more than one equation"):
            DataContainer(
                coords=coords_2d,
                primary_field_label_to_rank={},
                derived_field_label_to_rank={"a": 0},
                equations=[
                    Equation("first", make_a, {"a": 0}),
                    Equation("second", make_a, {"a": 0}),
                ],
            )

    def test_equation_labels_must_be_unique(self, coords_2d: Coords) -> None:
        def make_a(x: torch.Tensor, operators: Operators) -> torch.Tensor:
            return x

        with pytest.raises(ValueError, match="Equation labels must be unique"):
            DataContainer(
                coords=coords_2d,
                primary_field_label_to_rank={},
                derived_field_label_to_rank={"a": 0, "b": 0},
                equations=[
                    Equation("same", make_a, {"a": 0}),
                    Equation("same", make_a, {"b": 0}),
                ],
            )

    def test_equation_with_derivatives_evaluated_on_a_batch(
        self, coords_cylindrical: Coords, device: torch.device
    ) -> None:
        # As in training: u is computed from the batch coords, and an equation
        # then takes its derivatives on the same batch
        def make_w(u: torch.Tensor, operators: Operators) -> torch.Tensor:
            return operators.laplacian(u)

        data = DataContainer(
            coords=coords_cylindrical,
            primary_field_label_to_rank={"u": 0},
            derived_field_label_to_rank={"w": 0},
            equations=[Equation("w", make_w, {"w": 0})],
        )
        batch = coords_cylindrical.get_batch(torch.tensor([1, 4, 6], device=device))
        r, theta = batch.get_coord("r"), batch.get_coord("theta")

        data.set_field_from_tensor("u", r**2 * torch.sin(theta), batch)
        data.evaluate_derived_fields(batch)

        # Polar Laplacian of r^2 sin(theta) is 4 sin(theta) - sin(theta)
        assert torch.allclose(data.get_field_tensor("w", batch), 3 * torch.sin(theta))

    def test_derivatives_of_a_field_set_on_other_coords_raise(
        self, coords_2d: Coords, device: torch.device
    ) -> None:
        # u is set on the full coords, so it can be read on a batch, but its
        # derivatives with respect to the batch coords would all be zero
        def make_w(u: torch.Tensor, operators: Operators) -> torch.Tensor:
            return operators.d_dcoord(u, "x")

        data = DataContainer(
            coords=coords_2d,
            primary_field_label_to_rank={"u": 0},
            derived_field_label_to_rank={"w": 0},
            equations=[Equation("w", make_w, {"w": 0})],
        )
        data.set_field_from_tensor("u", coords_2d.get_coord("x") ** 2)
        batch = coords_2d.get_batch(torch.tensor([1, 4, 6], device=device))

        with pytest.raises(ValueError, match="computed from the operators' coords"):
            data.evaluate_derived_fields(batch)
