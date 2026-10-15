"""
Unit tests for data_structure/data_container.py.
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
from pinninit.differential_equations.operators import Operators
from pinninit.geometry.coordinate_system import CoordSystem

from .fixtures import N_POINTS, make_cartesian_2d, make_device, make_xy_2d


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


class TestDataContainerConstruction:
    def test_initialises_metadata_fields_consts(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)

        assert data.coords.coord_system is cartesian_2d
        assert data.coord_labels == ("x", "y")
        assert data.device == device

        assert data.primary_field_labels == {"u", "v"}
        assert data.derived_field_labels == {"w", "q"}
        assert data.field_labels == {"u", "v", "w", "q"}
        assert set(data.fields) == {"u", "v", "w", "q"}

        assert set(data.consts) == {"c"}
        assert data.consts["c"] is None
        assert data.primary_fields_active_coords == {"u": None, "v": None}
        assert data.derived_fields_active_coords == {"w": None, "q": None}
        assert data.consts_active_coords == {"c": None}

    def test_field_dataclass_is_cached_by_rank(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)

        assert type(data.get_field("u")) is type(data.get_field("w"))
        assert type(data.get_field("v")) is type(data.get_field("q"))
        assert type(data.get_field("u")) is not type(data.get_field("v"))

    def test_get_unknown_field_raises(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)

        with pytest.raises(KeyError, match="Field 'missing' not found"):
            data.get_field("missing")


class TestDataContainerFieldSettersAndGetters:
    def test_set_and_get_scalar_primary_field_from_tensor(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)
        coords = data.coords

        values = coords.get_coord("x") + coords.get_coord("y")

        data.set_field_from_tensor("u", values)

        assert torch.allclose(data.get_field_tensor("u"), values)
        assert torch.allclose(data.get_field_component_tensor("u"), values[:, None])
        assert data.primary_fields_active_coords["u"] is coords

    def test_set_and_get_vector_primary_field_from_component_tensor(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)
        coords = data.coords

        component_tensor = torch.stack(
            [
                coords.get_coord("x"),
                coords.get_coord("y"),
            ],
            dim=1,
        )

        data.set_field_from_component_tensor("v", component_tensor)

        assert torch.allclose(data.get_field_component_tensor("v"), component_tensor)
        assert torch.allclose(data.get_field_tensor("v"), component_tensor)
        assert data.primary_fields_active_coords["v"] is coords

    def test_set_and_get_vector_primary_field_from_component_dict(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)
        coords = data.coords

        components = {
            "x": coords.get_coord("x"),
            "y": coords.get_coord("y"),
        }

        data.set_field_from_component_dict("v", components)

        expected = torch.stack([components["x"], components["y"]], dim=1)

        assert torch.allclose(data.get_field_component_tensor("v"), expected)
        assert torch.allclose(data.get_field_tensor("v"), expected)

    def test_get_unset_field_raises(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)

        with pytest.raises(ValueError, match="'u' has not been initialised"):
            data.get_field_tensor("u")

    def test_setting_derived_field_without_allow_derived_raises(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)
        values = torch.ones(data.coords.n_points, device=device)

        with pytest.raises(
            ValueError, match="derived field and cannot be set directly"
        ):
            data.set_field_from_tensor("w", values)

    def test_setting_derived_field_with_allow_derived_succeeds(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)
        values = torch.ones(data.coords.n_points, device=device)

        data.set_field_from_tensor("w", values, allow_derived=True)

        assert torch.allclose(data.get_field_tensor("w"), values)
        assert data.derived_fields_active_coords["w"] is data.coords


class TestDataContainerBatching:
    def test_batch_getter_reads_subset_of_full_field(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)
        coords = data.coords
        full_values = coords.get_coord("x") + 2.0 * coords.get_coord("y")
        data.set_field_from_tensor("u", full_values)

        idx = torch.tensor([0, 3, 5], dtype=torch.long, device=device)
        batch_coords = coords.get_batch(idx)

        assert torch.allclose(
            data.get_field_tensor("u", batch_coords), full_values[idx]
        )
        assert torch.allclose(
            data.get_field_component_tensor("u", batch_coords),
            full_values[idx, None],
        )

    def test_batch_update_scatters_into_full_field(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)
        coords = data.coords

        initial = torch.zeros(coords.n_points, device=device)
        data.set_field_from_tensor("u", initial)

        idx = torch.tensor([1, 4, 6], dtype=torch.long, device=device)
        batch_coords = coords.get_batch(idx)
        batch_values = torch.tensor([10.0, 20.0, 30.0], device=device)

        data.set_field_from_tensor("u", batch_values, coords=batch_coords)

        expected = torch.zeros(coords.n_points, device=device)
        expected[idx] = batch_values

        assert torch.allclose(
            data.get_field_tensor("u", coords, enforce_ancestry=False), expected
        )
        assert data.primary_fields_active_coords["u"] is batch_coords

    def test_setting_field_with_unrelated_coords_raises(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)
        x, y = xy_2d
        unrelated_coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        values = torch.ones(unrelated_coords.n_points, device=device)

        with pytest.raises(ValueError, match="Evaluation coords"):
            data.set_field_from_tensor("u", values, coords=unrelated_coords)

    def test_get_field_component_full_coords(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)
        coords = data.coords
        values = coords.get_coord("x") + coords.get_coord("y")
        data.set_field_from_tensor("u", values)

        component = data.get_field_component("u", "value")
        assert torch.allclose(component, values)

    def test_get_field_component_batch_coords(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)
        coords = data.coords
        values = coords.get_coord("x") + coords.get_coord("y")
        data.set_field_from_tensor("u", values)

        idx = torch.tensor([0, 2, 4], dtype=torch.long, device=device)
        batch_coords = coords.get_batch(idx)

        component = data.get_field_component("u", "value", batch_coords)
        assert torch.allclose(component, values[idx])

    def test_get_field_component_vector_field(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)
        coords = data.coords
        component_tensor = torch.stack(
            [coords.get_coord("x"), coords.get_coord("y")], dim=1
        )
        data.set_field_from_component_tensor("v", component_tensor)

        x_component = data.get_field_component("v", "x")
        y_component = data.get_field_component("v", "y")

        assert torch.allclose(x_component, coords.get_coord("x"))
        assert torch.allclose(y_component, coords.get_coord("y"))

    def test_get_field_component_unset_raises(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)

        with pytest.raises(ValueError, match="has not been initialised"):
            data.get_field_component("u", "value")

    def test_get_field_component_invalid_component_raises(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)
        values = torch.ones(data.coords.n_points, device=device)
        data.set_field_from_tensor("u", values)

        with pytest.raises(KeyError, match="not found in field components"):
            data.get_field_component("u", "invalid_component")


class TestDataContainerConsts:
    def test_set_and_get_const(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)

        value = torch.tensor(3.0, device=device)
        data.set_const("c", value)

        assert data.get_const("c") is value
        assert data.consts_active_coords["c"] is data.coords

    def test_set_const_requires_scalar(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)

        with pytest.raises(ValueError, match="must be a scalar tensor"):
            data.set_const("c", torch.ones(2, device=device))

    def test_set_unknown_const_raises(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)

        with pytest.raises(KeyError, match="Const 'missing' not found"):
            data.set_const("missing", torch.tensor(1.0, device=device))

    def test_get_unset_const_raises(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)

        with pytest.raises(ValueError, match="'c' has not been initialised"):
            data.get_const("c")


class TestDataContainerDerivedFieldEvaluation:
    def test_evaluate_derived_fields_delegates_to_equation_evaluator(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        def make_w(
            x: torch.Tensor, y: torch.Tensor, operators: Operators
        ) -> torch.Tensor:
            return x * y

        equation = Equation("make_w", make_w, {"w": 0})

        data = DataContainer(
            coords=coords,
            primary_field_label_to_rank={},
            derived_field_label_to_rank={"w": 0},
            equations=[equation],
        )

        data.evaluate_derived_fields()

        expected = coords.get_coord("x") * coords.get_coord("y")

        assert torch.allclose(data.get_field_tensor("w"), expected)

    def test_validate_eval_coords_primary_fields_accepts_descendant_batch(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)
        coords = data.coords
        data.set_field_from_tensor("u", torch.ones(coords.n_points, device=device))
        data.set_field_from_component_tensor(
            "v",
            torch.ones(coords.n_points, 2, device=device),
        )

        batch_coords = coords.get_batch(
            torch.tensor([0, 2, 4], dtype=torch.long, device=device)
        )

        data.validate_eval_coords_primary_fields(batch_coords)

    def test_validate_eval_coords_primary_fields_rejects_unset_primary(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        data = build_container(cartesian_2d, xy_2d)

        with pytest.raises(ValueError, match="has not been initialised"):
            data.validate_eval_coords_primary_fields(data.coords)
