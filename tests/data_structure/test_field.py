"""Unit tests for data_structure/field.py."""

from __future__ import annotations

import pytest
import torch

from pinninit.data_structure.coords import Coords
from pinninit.data_structure.field import (
    Field,
    generate_tensor_component_labels,
    make_field_dataclass,
)
from pinninit.geometry.coordinate_system import CoordSystem

from ..helpers import N_POINTS, make_coords


def make_field(rank: int, coords: Coords) -> Field:
    """A field whose component i has the value i at every point."""
    field_class = make_field_dataclass(rank, type(coords))
    component_labels = generate_tensor_component_labels(coords.geometric_labels, rank)
    components = {
        label: torch.full((coords.n_points,), float(i), device=coords.device)
        for i, label in enumerate(component_labels)
    }
    return field_class(label="f", coords=coords, **components)


class TestComponentLabels:
    def test_scalar_has_one_value_component(self) -> None:
        assert generate_tensor_component_labels(("x", "y"), 0) == ("value",)

    def test_vector_components_are_the_geometric_labels(self) -> None:
        assert generate_tensor_component_labels(("r", "theta"), 1) == ("r", "theta")

    def test_rank_2_components_are_pairs_of_geometric_labels(self) -> None:
        labels = generate_tensor_component_labels(("x", "y"), 2)
        assert labels == ("x_x", "x_y", "y_x", "y_y")

    def test_negative_rank_raises(self) -> None:
        with pytest.raises(ValueError, match="rank must be >= 0"):
            generate_tensor_component_labels(("x",), -1)


class TestConstruction:
    def test_missing_components_start_as_zero(
        self, coords_2d: Coords, device: torch.device
    ) -> None:
        field_class = make_field_dataclass(1, type(coords_2d))

        field = field_class(label="f", coords=coords_2d)

        expected = torch.zeros(N_POINTS, 2, device=device)
        assert torch.equal(field.component_tensor(), expected)

    def test_component_with_wrong_shape_raises(
        self, coords_2d: Coords, device: torch.device
    ) -> None:
        field_class = make_field_dataclass(1, type(coords_2d))
        with pytest.raises(ValueError, match="shape"):
            field_class(
                label="f", coords=coords_2d, x=torch.zeros(N_POINTS + 1, device=device)
            )

    def test_component_that_is_not_a_tensor_raises(
        self, coords_2d: Coords, device: torch.device
    ) -> None:
        field_class = make_field_dataclass(1, type(coords_2d))
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            field_class(label="f", coords=coords_2d, x=[0.0] * N_POINTS)  # type: ignore[arg-type]

    def test_one_class_per_rank_and_kind_of_coords(
        self, coords_2d: Coords, device: torch.device
    ) -> None:
        batch = coords_2d.get_batch(torch.tensor([0, 1], device=device))
        field_class = make_field_dataclass(1, type(coords_2d))

        assert make_field_dataclass(1, type(batch)) is field_class
        assert make_field_dataclass(0, type(coords_2d)) is not field_class
        assert field_class.coords_class is type(coords_2d)
        assert field_class.rank == 1
        assert field_class.component_labels == ("x", "y")

    def test_coords_of_another_kind_raise(
        self, coords_2d: Coords, coords_cylindrical: Coords
    ) -> None:
        field_class = make_field_dataclass(1, type(coords_2d))
        with pytest.raises(TypeError, match="must be an instance of"):
            field_class(label="f", coords=coords_cylindrical)

    def test_class_constants_cannot_be_passed_in(self, coords_2d: Coords) -> None:
        field_class = make_field_dataclass(1, type(coords_2d))
        with pytest.raises(TypeError, match="unexpected keyword"):
            field_class(label="f", coords=coords_2d, rank=2)  # type: ignore[arg-type]

    def test_components_of_different_dtypes_raise(
        self, coords_2d: Coords, device: torch.device
    ) -> None:
        field_class = make_field_dataclass(1, type(coords_2d))
        x = torch.zeros(N_POINTS, dtype=torch.float32, device=device)
        y = torch.zeros(N_POINTS, dtype=torch.float64, device=device)
        with pytest.raises(ValueError, match="same dtype"):
            field_class(label="f", coords=coords_2d, x=x, y=y)

    def test_integer_fields_are_allowed(
        self, coords_2d: Coords, device: torch.device
    ) -> None:
        # e.g. BRDR stores the step each point was last seen
        field_class = make_field_dataclass(0, type(coords_2d))
        value = torch.arange(N_POINTS, device=device)

        field = field_class(label="n", coords=coords_2d, value=value)

        assert field.dtype == torch.long
        assert torch.equal(field.tensor(), value)

    def test_cannot_set_undeclared_attribute(self, coords_2d: Coords) -> None:
        field = make_field(1, coords_2d)
        with pytest.raises(AttributeError):
            field.z = torch.zeros(N_POINTS)  # type: ignore[attr-defined]


class TestTensorViews:
    @pytest.mark.parametrize(
        ("rank", "component_shape", "tensor_shape"),
        [
            (0, (N_POINTS, 1), (N_POINTS,)),
            (1, (N_POINTS, 2), (N_POINTS, 2)),
            (2, (N_POINTS, 4), (N_POINTS, 2, 2)),
        ],
    )
    def test_shapes(
        self,
        coords_2d: Coords,
        rank: int,
        component_shape: tuple[int, ...],
        tensor_shape: tuple[int, ...],
    ) -> None:
        field = make_field(rank, coords_2d)

        assert field.component_tensor().shape == component_shape
        assert field.tensor().shape == tensor_shape

    def test_rank_2_tensor_component_ordering(
        self, cartesian_3d: CoordSystem, device: torch.device
    ) -> None:
        x = y = z = torch.zeros(N_POINTS, device=device)
        field = make_field(2, make_coords(cartesian_3d, x=x, y=y, z=z))

        # Components x_x, x_y, ..., z_z hold the values 0, 1, ..., 8
        expected = torch.arange(9.0, device=device).reshape(3, 3)
        assert torch.equal(field.tensor()[0], expected)

    def test_get_component(self, coords_2d: Coords, device: torch.device) -> None:
        field = make_field(1, coords_2d)
        expected = torch.ones(N_POINTS, device=device)
        assert torch.equal(field.get_component("y"), expected)

    def test_unknown_component_raises(self, coords_2d: Coords) -> None:
        field = make_field(1, coords_2d)
        with pytest.raises(KeyError, match="not found in field components"):
            field.get_component("z")


class TestUpdates:
    def test_update_from_tensor(self, coords_2d: Coords, device: torch.device) -> None:
        field = make_field(2, coords_2d)
        new_tensor = torch.randn(N_POINTS, 2, 2, device=device)

        field.update_from_tensor(new_tensor)

        assert torch.equal(field.tensor(), new_tensor)
        assert torch.equal(field.get_component("x_y"), new_tensor[:, 0, 1])

    def test_update_from_component_tensor(
        self, coords_2d: Coords, device: torch.device
    ) -> None:
        field = make_field(1, coords_2d)
        new_components = torch.randn(N_POINTS, 2, device=device)

        field.update_from_component_tensor(new_components)

        assert torch.equal(field.component_tensor(), new_components)
        assert torch.equal(field.get_component("y"), new_components[:, 1])

    def test_update_from_component_dict(
        self, coords_2d: Coords, device: torch.device
    ) -> None:
        field = make_field(1, coords_2d)
        x = torch.randn(N_POINTS, device=device)
        y = torch.randn(N_POINTS, device=device)

        field.update_from_component_dict({"x": x, "y": y})

        assert torch.equal(field.get_component("x"), x)
        assert torch.equal(field.get_component("y"), y)

    def test_update_from_tensor_with_wrong_shape_raises(
        self, coords_2d: Coords, device: torch.device
    ) -> None:
        field = make_field(1, coords_2d)
        with pytest.raises(ValueError, match="shape"):
            field.update_from_tensor(torch.zeros(N_POINTS, 3, device=device))

    def test_update_from_component_dict_with_missing_component_raises(
        self, coords_2d: Coords, device: torch.device
    ) -> None:
        field = make_field(1, coords_2d)
        with pytest.raises(KeyError, match="all field component labels"):
            field.update_from_component_dict(
                {"x": torch.zeros(N_POINTS, device=device)}
            )


class TestBatches:
    def test_reads_on_nested_batch(
        self, coords_2d: Coords, device: torch.device
    ) -> None:
        field = make_field(1, coords_2d)
        field.update_from_component_tensor(torch.randn(N_POINTS, 2, device=device))
        idx1 = torch.tensor([0, 2, 3, 7], device=device)
        idx2 = torch.tensor([1, 3], device=device)

        batch = coords_2d.get_batch(idx1).get_batch(idx2)

        expected = field.component_tensor()[idx1[idx2]]
        assert torch.equal(field.component_tensor(batch), expected)
        assert torch.equal(field.tensor(batch), expected)
        assert torch.equal(field.get_component("x", batch), expected[:, 0])

    def test_field_on_a_batch(self, coords_2d: Coords, device: torch.device) -> None:
        batch = coords_2d.get_batch(torch.tensor([1, 4, 6], device=device))
        field_class = make_field_dataclass(0, type(coords_2d))
        value = torch.tensor([1.0, 2.0, 3.0], device=device)

        field = field_class(label="f", coords=batch, value=value)

        assert torch.equal(field.tensor(), value)
        # The full coords are not part of the batch, so can't be read
        with pytest.raises(ValueError, match="root coords or a batch"):
            field.tensor(coords_2d)

    def test_batch_update_with_another_dtype_raises(
        self, coords_2d: Coords, device: torch.device
    ) -> None:
        field = make_field(0, coords_2d)
        batch = coords_2d.get_batch(torch.tensor([0, 1], device=device))
        values = torch.zeros(2, dtype=torch.float64, device=device)
        with pytest.raises(ValueError, match="dtype"):
            field.update_from_tensor(values, batch)

    def test_unrelated_coords_raise(
        self, coords_2d: Coords, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        field = make_field(1, coords_2d)
        unrelated = make_coords(
            cartesian_2d,
            x=torch.zeros(N_POINTS, device=device),
            y=torch.zeros(N_POINTS, device=device),
        )
        with pytest.raises(ValueError, match="root coords or a batch"):
            field.tensor(unrelated)

    def test_batch_update_writes_batch_rows_and_detaches_the_rest(
        self, coords_2d: Coords, device: torch.device
    ) -> None:
        field = make_field(0, coords_2d)
        x = coords_2d.get_coord("x")
        first = coords_2d.get_batch(torch.tensor([0, 1], device=device))
        second = coords_2d.get_batch(torch.tensor([2, 3], device=device))

        # The rows of the first batch depend on x
        field.update_from_tensor(2 * x[:2], first)
        assert torch.equal(field.tensor()[:2], 2 * x[:2].detach())
        (grad,) = torch.autograd.grad(field.tensor().sum(), x)
        assert torch.equal(grad[:4], torch.tensor([2.0, 2.0, 0.0, 0.0], device=device))

        # Updating the second batch keeps the first batch's values, but detaches them
        field.update_from_tensor(3 * x[2:4], second)
        assert torch.equal(field.tensor()[:2], 2 * x[:2].detach())
        (grad,) = torch.autograd.grad(field.tensor().sum(), x)
        assert torch.equal(grad[:4], torch.tensor([0.0, 0.0, 3.0, 3.0], device=device))
