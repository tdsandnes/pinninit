"""Unit tests for field_dataclass / make_field_dataclass."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from pinninit.data_structure.coords_dataclass import (
    CoordsProtocol,
    make_coords_dataclass,
)
from pinninit.data_structure.field_dataclass import (
    FieldProtocol,
    generate_tensor_component_labels,
    make_field_dataclass,
)

from .fixtures import (
    N_POINTS,
    make_cartesian_2d,
    make_device,
    make_pinn_environment,
)


# Local Mock Coords object to isolate field testing from coordinate logic
class MockCoords:
    def __init__(
        self,
        geometric_labels: tuple[str, ...],
        n_points: int,
        device: torch.device,
        is_batch: bool = False,
        parent: CoordsProtocol | None = None,
        parent_idx: torch.Tensor | None = None,
    ) -> None:
        self.geometric_labels = geometric_labels
        self.n_points = n_points
        self.device = device
        self.is_batch = is_batch
        self.parent = parent
        self.parent_idx = parent_idx

        # Minimal derived attributes needed by field logic
        self.coord_labels = geometric_labels
        self.internal_labels = geometric_labels
        self.tensor = torch.zeros(n_points, len(geometric_labels), device=device)
        self.geometric = self.tensor
        self.internal = self.tensor


# Local fixtures
@pytest.fixture  # type: ignore[misc]
def device() -> torch.device:
    return make_device()


@pytest.fixture  # type: ignore[misc]
def n_points() -> int:
    return N_POINTS


@pytest.fixture  # type: ignore[misc]
def coords_2d(device: torch.device, n_points: int) -> CoordsProtocol:
    return cast(
        CoordsProtocol,
        MockCoords(geometric_labels=("x", "y"), n_points=n_points, device=device),
    )


@pytest.fixture  # type: ignore[misc]
def coords_3d(device: torch.device, n_points: int) -> CoordsProtocol:
    return cast(
        CoordsProtocol,
        MockCoords(geometric_labels=("x", "y", "z"), n_points=n_points, device=device),
    )


@pytest.fixture  # type: ignore[misc]
def coords_2d_different(device: torch.device, n_points: int) -> CoordsProtocol:
    return cast(
        CoordsProtocol,
        MockCoords(geometric_labels=("x", "y"), n_points=n_points + 1, device=device),
    )


@pytest.fixture  # type: ignore[misc]
def setup_pinn_environment(
    device: torch.device,
) -> tuple[CoordsProtocol, FieldProtocol, torch.nn.Linear]:
    return make_pinn_environment(device=device)


# Helpers
def make_field(
    rank: int, coords: CoordsProtocol, device: torch.device, label: str = "f"
) -> FieldProtocol:
    Field = make_field_dataclass(
        rank=rank,
        coords=coords,
        device=device,
    )
    labels = generate_tensor_component_labels(coords.geometric_labels, rank)
    values = {
        label: torch.linspace(1.0, 5.0, coords.n_points, device=device)
        for label in labels
    }
    return Field(label=label, **values)


# Construction and basic attributes
class TestFieldConstruction:
    def test_component_labels(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        field = make_field(rank=1, coords=coords_2d, device=device)
        assert field.component_labels == ("x", "y")

    def test_coords_is_bound_to_class(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        field = make_field(rank=1, coords=coords_2d, device=device)
        assert field.coords == coords_2d

    def test_field_is_mutable(
        self,
        coords_2d: CoordsProtocol,
        coords_2d_different: CoordsProtocol,
        device: torch.device,
    ) -> None:
        field = make_field(rank=1, coords=coords_2d, device=device)

        # Mutate attribute value
        zero_update = torch.zeros_like(field.get_component("x"))
        field.x = zero_update  # type: ignore[attr-defined]

        # Ensure view allocations and underlying tracking sync automatically
        assert torch.allclose(field.get_component("x"), zero_update)
        assert torch.allclose(field.component_tensor(coords_2d)[:, 0], zero_update)
        assert torch.allclose(field.tensor(coords_2d)[:, 0], zero_update)

        # Verify that evaluating against unrelated coords fails
        with pytest.raises(ValueError, match="root coords or a batch"):
            field.component_tensor(coords_2d_different)


# Tensor operations and shapes
class TestFieldTensorShapes:
    @pytest.mark.parametrize(
        ("rank", "component_shape", "tensor_shape"),
        [
            (0, (N_POINTS, 1), (N_POINTS,)),
            (1, (N_POINTS, 2), (N_POINTS, 2)),
            (2, (N_POINTS, 4), (N_POINTS, 2, 2)),
            (3, (N_POINTS, 8), (N_POINTS, 2, 2, 2)),
        ],
    )  # type: ignore[misc]
    def test_shapes(
        self,
        rank: int,
        component_shape: tuple[int, ...],
        tensor_shape: tuple[int, ...],
        coords_2d: CoordsProtocol,
        device: torch.device,
    ) -> None:
        field = make_field(rank, coords_2d, device)

        component_tensor = field.component_tensor(coords_2d)
        tensor = field.tensor(coords_2d)

        assert component_tensor.shape == component_shape
        assert tensor.shape == tensor_shape

    def test_rank_2_3d_component_ordering(
        self, coords_3d: CoordsProtocol, device: torch.device
    ) -> None:
        Field = make_field_dataclass(
            rank=2,
            coords=coords_3d,
            device=device,
        )

        values = {
            "x_x": torch.tensor([0.0] * coords_3d.n_points, device=device),
            "x_y": torch.tensor([1.0] * coords_3d.n_points, device=device),
            "x_z": torch.tensor([2.0] * coords_3d.n_points, device=device),
            "y_x": torch.tensor([3.0] * coords_3d.n_points, device=device),
            "y_y": torch.tensor([4.0] * coords_3d.n_points, device=device),
            "y_z": torch.tensor([5.0] * coords_3d.n_points, device=device),
            "z_x": torch.tensor([6.0] * coords_3d.n_points, device=device),
            "z_y": torch.tensor([7.0] * coords_3d.n_points, device=device),
            "z_z": torch.tensor([8.0] * coords_3d.n_points, device=device),
        }

        field = Field(label="f", **values)

        tensor = field.tensor(coords_3d)

        expected_first_point = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
            ],
            device=device,
        )

        assert torch.equal(tensor[0], expected_first_point)

    def test_rank_3_3d_component_ordering(
        self, coords_3d: CoordsProtocol, device: torch.device
    ) -> None:
        Field = make_field_dataclass(
            rank=3,
            coords=coords_3d,
            device=device,
        )

        labels = generate_tensor_component_labels(coords_3d.geometric_labels, 3)

        values = {
            label: torch.full((coords_3d.n_points,), float(i), device=device)
            for i, label in enumerate(labels)
        }

        field = Field(label="f", **values)
        tensor = field.tensor(coords_3d)

        for i, label in enumerate(labels):
            a, b, c = label.split("_")

            ia = coords_3d.geometric_labels.index(a)
            ib = coords_3d.geometric_labels.index(b)
            ic = coords_3d.geometric_labels.index(c)

            assert torch.all(tensor[:, ia, ib, ic] == float(i))


# State updates
class TestFieldUpdateMethods:
    def test_update_from_component_dict_rank_1(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        field = make_field(rank=1, coords=coords_2d, device=device)

        new_x = torch.zeros_like(field.get_component("x"))
        new_y = torch.ones_like(field.get_component("y")) * 7.0

        field.update_from_component_dict(
            {
                "x": new_x,
                "y": new_y,
            }
        )

        assert torch.allclose(field.get_component("x"), new_x)
        assert torch.allclose(field.get_component("y"), new_y)

        expected = torch.stack((new_x, new_y), dim=1)
        assert torch.allclose(field.component_tensor(coords_2d), expected)

    def test_update_from_component_dict_missing_component_raises(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        field = make_field(rank=1, coords=coords_2d, device=device)

        with pytest.raises(
            KeyError,
            match="must contain all field component labels",
        ):
            field.update_from_component_dict(
                {"x": torch.zeros_like(field.get_component("x"))}
            )

    def test_update_from_component_dict_extra_component_raises(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        field = make_field(rank=1, coords=coords_2d, device=device)

        with pytest.raises(
            KeyError,
            match="must contain all field component labels",
        ):
            field.update_from_component_dict(
                {
                    "x": torch.zeros_like(field.get_component("x")),
                    "y": torch.zeros_like(field.get_component("x")),
                    "z": torch.zeros_like(field.get_component("x")),
                }
            )

    def test_mismatched_points_length_update_raises(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        field = make_field(rank=1, coords=coords_2d, device=device)

        bad_length_tensor = torch.ones(
            coords_2d.n_points + 5,
            device=device,
        )

        with pytest.raises(ValueError, match="has shape"):
            field.update_from_component_dict(
                {
                    "x": bad_length_tensor,
                    "y": torch.zeros_like(field.get_component("y")),
                }
            )

    def test_update_from_component_tensor_rank_1(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        field = make_field(rank=1, coords=coords_2d, device=device)

        new_tensor = torch.ones_like(field.component_tensor(coords_2d)) * 8.0
        field.update_from_component_tensor(new_tensor)

        assert torch.allclose(
            field.get_component("x"), torch.ones_like(field.get_component("x")) * 8.0
        )
        assert torch.allclose(
            field.get_component("y"), torch.ones_like(field.get_component("y")) * 8.0
        )

    def test_update_from_component_tensor_wrong_n_points_raises(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        field = make_field(rank=1, coords=coords_2d, device=device)

        bad_tensor = torch.zeros(
            (coords_2d.n_points + 1, 2),
            device=device,
        )

        with pytest.raises(ValueError, match="Component tensor must have shape"):
            field.update_from_component_tensor(bad_tensor)

    def test_update_from_tensor_rank_1(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        field = make_field(rank=1, coords=coords_2d, device=device)

        # Ensure x-slice values differ entirely from y-slice values
        base_tensor = field.tensor(coords_2d)
        new_tensor = torch.zeros_like(base_tensor)
        new_tensor[:, 0] = 12.0
        new_tensor[:, 1] = 99.0

        field.update_from_tensor(new_tensor)

        assert torch.allclose(field.tensor(coords_2d), new_tensor)
        assert torch.allclose(field.get_component("x"), new_tensor[:, 0])
        assert torch.allclose(field.get_component("y"), new_tensor[:, 1])

    def test_update_from_tensor_rank_1_wrong_shape_raises(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        field = make_field(rank=1, coords=coords_2d, device=device)

        bad_tensor = torch.zeros(
            (coords_2d.n_points, 3),
            device=device,
        )

        with pytest.raises(ValueError):
            field.update_from_tensor(bad_tensor)

    def test_update_from_tensor_rank_2_wrong_shape_raises(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        field = make_field(rank=2, coords=coords_2d, device=device)

        bad_tensor = torch.zeros(
            (coords_2d.n_points, 2, 3),
            device=device,
        )

        with pytest.raises(ValueError):
            field.update_from_tensor(bad_tensor)

    def test_update_keeps_all_representations_consistent(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        field = make_field(rank=2, coords=coords_2d, device=device)

        new_tensor = torch.arange(
            coords_2d.n_points * 4,
            dtype=torch.float32,
            device=device,
        ).reshape(coords_2d.n_points, 2, 2)

        field.update_from_tensor(new_tensor)

        assert torch.allclose(
            field.tensor(coords_2d),
            new_tensor,
        )

        expected_component_tensor = new_tensor.reshape(
            coords_2d.n_points,
            4,
        )

        assert torch.allclose(
            field.component_tensor(coords_2d),
            expected_component_tensor,
        )

        for i, label in enumerate(field.component_labels):
            assert torch.allclose(
                getattr(field, label),
                expected_component_tensor[:, i],
            )

    def test_field_created_with_label_only(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        Field = make_field_dataclass(rank=1, coords=coords_2d, device=device)
        field = Field(label="velocity")

        assert torch.allclose(
            field.get_component("x"), torch.zeros(coords_2d.n_points, device=device)
        )
        assert torch.allclose(
            field.get_component("y"), torch.zeros(coords_2d.n_points, device=device)
        )

    def test_field_label_only_then_update_from_tensor(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        Field = make_field_dataclass(rank=1, coords=coords_2d, device=device)
        field = Field(label="velocity")

        new_tensor = torch.zeros(coords_2d.n_points, 2, device=device)
        new_tensor[:, 0] = 3.0
        new_tensor[:, 1] = 7.0

        field.update_from_tensor(new_tensor)

        assert torch.allclose(
            field.get_component("x"),
            torch.full((coords_2d.n_points,), 3.0, device=device),
        )
        assert torch.allclose(
            field.get_component("y"),
            torch.full((coords_2d.n_points,), 7.0, device=device),
        )
        assert torch.allclose(field.tensor(coords_2d), new_tensor)


class TestScalarFieldBehavior:
    def test_scalar_field_construction_and_shapes(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        # Rank 0 should have component labels ("value",)
        Field = make_field_dataclass(rank=0, coords=coords_2d, device=device)
        value = torch.linspace(10.0, 50.0, coords_2d.n_points, device=device)
        scalar_field = Field(label="f", value=value)

        assert scalar_field.component_labels == ("value",)

        # component_tensor has shape (n_points, 1)
        component_tensor = scalar_field.component_tensor(coords_2d)
        assert component_tensor.shape == (coords_2d.n_points, 1)
        assert torch.allclose(component_tensor.squeeze(1).detach(), value)

        # tensor has shape (n_points,)
        tensor = scalar_field.tensor(coords_2d)
        assert tensor.shape == (coords_2d.n_points,)
        assert torch.allclose(tensor.detach(), value)

    def test_scalar_field_batch_evaluation(self, device: torch.device) -> None:
        # Set up realistic coordinates for batching
        coord_system = make_cartesian_2d(device=device)
        Coords = make_coords_dataclass(
            {"x", "y"}, coord_system=coord_system, device=device
        )
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        y = torch.tensor([5.0, 6.0, 7.0, 8.0], device=device)
        coords = Coords(coord_system=coord_system, x=x, y=y)

        # Generate Scalar Field
        Field = make_field_dataclass(rank=0, coords=coords, device=device)
        raw_values = torch.tensor([100.0, 200.0, 300.0, 400.0], device=device)
        scalar_field = Field(label="f", value=raw_values)

        # Slice batch coordinates: indices [1, 3]
        indices = torch.tensor([1, 3], device=device)
        batched_coords = coords.get_batch(indices)

        # Evaluate the components and vectors relative to the batch
        sliced_component_tensor = scalar_field.component_tensor(batched_coords)
        sliced_tensor = scalar_field.tensor(batched_coords)

        # Shape Verification
        assert sliced_component_tensor.shape == (2, 1)
        assert sliced_tensor.shape == (2,)

        # Value Slicing Verification
        expected_values = torch.tensor([200.0, 400.0], device=device)
        assert torch.allclose(sliced_component_tensor[:, 0], expected_values)
        assert torch.allclose(sliced_tensor, expected_values)

    def test_scalar_field_update_from_tensor(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        field = make_field(rank=0, coords=coords_2d, device=device)

        # Scalar structured tensors have shape (n_points,)
        new_flat_tensor = torch.ones(coords_2d.n_points, device=device) * 55.0
        field.update_from_tensor(new_flat_tensor)

        assert torch.allclose(field.get_component("value"), new_flat_tensor)
        assert torch.allclose(field.tensor(coords_2d), new_flat_tensor)
        assert torch.allclose(field.component_tensor(coords_2d)[:, 0], new_flat_tensor)

    def test_scalar_field_update_from_component_tensor_wrong_shape_raises(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        field = make_field(rank=0, coords=coords_2d, device=device)

        bad_tensor = torch.zeros(
            (coords_2d.n_points,),
            device=device,
        )

        with pytest.raises(ValueError, match="Component tensor must have shape"):
            field.update_from_component_tensor(bad_tensor)

    def test_scalar_field_update_from_tensor_wrong_shape_raises(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        field = make_field(rank=0, coords=coords_2d, device=device)

        bad_tensor = torch.zeros(
            (coords_2d.n_points, 1),
            device=device,
        )

        with pytest.raises(ValueError, match="Tensor must have shape"):
            field.update_from_tensor(bad_tensor)

    def test_scalar_field_created_with_label_only(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        Field = make_field_dataclass(rank=0, coords=coords_2d, device=device)
        field = Field(label="pressure")

        assert torch.allclose(
            field.get_component("value"),
            torch.zeros(coords_2d.n_points, device=device),
        )


class TestGenerateTensorComponentLabels:
    def test_rank_0_returns_value(self) -> None:
        labels = generate_tensor_component_labels(("x", "y"), rank=0)

        assert labels == ("value",)

    def test_rank_1_labels_cartesian_2d(self) -> None:
        labels = generate_tensor_component_labels(("x", "y"), rank=1)

        assert labels == ("x", "y")

    def test_rank_1_labels_spherical(self) -> None:
        labels = generate_tensor_component_labels(("r", "theta", "phi"), rank=1)

        assert labels == ("r", "theta", "phi")

    def test_rank_2_labels_3d_cartesian(self) -> None:
        labels = generate_tensor_component_labels(("x", "y", "z"), rank=2)

        assert labels == ("x_x", "x_y", "x_z", "y_x", "y_y", "y_z", "z_x", "z_y", "z_z")

    def test_rank_2_labels_spherical(self) -> None:
        labels = generate_tensor_component_labels(("r", "theta", "phi"), rank=2)

        assert labels == (
            "r_r",
            "r_theta",
            "r_phi",
            "theta_r",
            "theta_theta",
            "theta_phi",
            "phi_r",
            "phi_theta",
            "phi_phi",
        )

    def test_rank_3_labels_2d_cartesian(self) -> None:
        labels = generate_tensor_component_labels(("x", "y"), rank=3)

        assert labels == (
            "x_x_x",
            "x_x_y",
            "x_y_x",
            "x_y_y",
            "y_x_x",
            "y_x_y",
            "y_y_x",
            "y_y_y",
        )


# Tensor evaluation and Autograd graph validation
class TestFieldCoordinateEvaluationAndAutograd:
    def test_evaluation_with_real_nested_sub_batch_coords_slices(
        self, device: torch.device
    ) -> None:
        coord_system = make_cartesian_2d(device=device)
        Coords = make_coords_dataclass(
            {"x", "y"}, coord_system=coord_system, device=device
        )
        x = torch.tensor([10.0, 20.0, 30.0, 40.0], device=device, requires_grad=True)
        y = torch.tensor(
            [100.0, 200.0, 300.0, 400.0], device=device, requires_grad=True
        )
        coords = Coords(coord_system=coord_system, x=x, y=y)

        # Build a real vector field bound to these coordinates
        f_x = torch.tensor([10.0, 20.0, 30.0, 40.0], device=device, requires_grad=True)
        f_y = torch.tensor([50.0, 60.0, 70.0, 80.0], device=device, requires_grad=True)
        Field = make_field_dataclass(rank=1, coords=coords, device=device)
        field_instance = Field(label="f", x=f_x, y=f_y)

        # Create Level 1 sub-batch: pick indices [0, 2, 3]
        parent_idx_l1 = torch.tensor([0, 2, 3], device=device)
        sub_batch_coords_l1 = coords.get_batch(parent_idx_l1)

        # Create Level 2 sub-batch: indices [0, 2]
        parent_idx_l2 = torch.tensor([0, 2], device=device)
        sub_batch_coords_l2 = sub_batch_coords_l1.get_batch(parent_idx_l2)

        # Evaluate tensors by passing the nested batch into the full field instance
        sliced_component_tensor = field_instance.component_tensor(sub_batch_coords_l2)
        sliced_structured_tensor = field_instance.tensor(sub_batch_coords_l2)

        # Verify subset layout mapping shapes
        assert sliced_component_tensor.shape == (2, 2)
        assert sliced_structured_tensor.shape == (2, 2)

        # Verify values match absolute root parent indexes recursively
        root_comp_tensor = field_instance.component_tensor(coords)

        # Index 0 of nested batch -> Index 0 of batch L1 -> Index 0 of root parent
        assert torch.allclose(sliced_component_tensor[0], root_comp_tensor[0])
        # Index 1 of nested batch -> Index 2 of batch L1 -> Index 3 of root parent
        assert torch.allclose(sliced_component_tensor[1], root_comp_tensor[3])

        # Check structured tensor shape mapping consistency
        root_structured_tensor = field_instance.tensor(coords)
        assert torch.allclose(sliced_structured_tensor[0], root_structured_tensor[0])
        assert torch.allclose(sliced_structured_tensor[1], root_structured_tensor[3])

    def test_evaluation_with_invalid_sub_batch_raises(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        field_instance = make_field(rank=1, coords=coords_2d, device=device)

        # Mock coord that claims to be a variant but lacks batch validation flags
        invalid_coords = cast(
            CoordsProtocol,
            MockCoords(
                geometric_labels=("x", "y"), n_points=2, device=device, is_batch=False
            ),
        )

        with pytest.raises(ValueError, match="root coords or a batch"):
            field_instance.component_tensor(invalid_coords)

    def test_batch_update_then_autograd_reflects_updated_values(
        self, device: torch.device
    ) -> None:
        """
        Build a scalar field initialised to x everywhere.
        Update a batch of points to y.
        Verify that autodiff wrt x and y gives the right derivatives
        at updated vs non-updated points.
        """
        coord_system = make_cartesian_2d(device=device)
        Coords = make_coords_dataclass(
            {"x", "y"}, coord_system=coord_system, device=device
        )
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, requires_grad=True)
        y = torch.tensor([5.0, 6.0, 7.0, 8.0], device=device, requires_grad=True)
        coords = Coords(coord_system=coord_system, x=x, y=y)

        # Initialise scalar field to x everywhere
        Field = make_field_dataclass(rank=0, coords=coords, device=device)
        field = Field(label="f", value=x.clone())

        # Update points [1, 3] to y
        batch_idx = torch.tensor([1, 3], device=device)
        batch_coords = coords.get_batch(batch_idx)
        batch_y = torch.stack([y[1], y[3]])
        field.update_from_tensor(batch_y, coords=batch_coords)

        # Evaluate full field and sum
        full_tensor = field.tensor(coords)  # shape (4,)
        loss = full_tensor.sum()  # Expected: x[0] + y[1] + x[2] + y[3]

        loss.backward()  # type: ignore[no-untyped-call]

        # Points not in batch
        assert x.grad is not None
        assert torch.allclose(
            x.grad[0], torch.tensor(1.0, device=device)
        )  # dloss/dx[0] = 1
        assert torch.allclose(
            x.grad[1], torch.tensor(0.0, device=device)
        )  # updated, not x
        assert torch.allclose(
            x.grad[2], torch.tensor(1.0, device=device)
        )  # dloss/dx[2] = 1
        assert torch.allclose(
            x.grad[3], torch.tensor(0.0, device=device)
        )  # updated, not x

        # Points in batch
        assert y.grad is not None
        assert torch.allclose(
            y.grad[0], torch.tensor(0.0, device=device)
        )  # not updated
        assert torch.allclose(
            y.grad[1], torch.tensor(1.0, device=device)
        )  # dloss/dy[1] = 1
        assert torch.allclose(
            y.grad[2], torch.tensor(0.0, device=device)
        )  # not updated
        assert torch.allclose(
            y.grad[3], torch.tensor(1.0, device=device)
        )  # dloss/dy[3] = 1


class TestFieldNeuralNetworkIntegration:
    def test_nn_raw_tensor_evaluation_and_data_loss(
        self,
        setup_pinn_environment: tuple[CoordsProtocol, FieldProtocol, torch.nn.Linear],
        device: torch.device,
    ) -> None:
        """
        Data Loss: Evaluates an NN directly on sliced batch coordinates,
        and computes the loss against an evaluated slice of the reference Field
        """
        coords, reference_field, nn_layer = setup_pinn_environment
        idx = torch.tensor([0, 2], device=device)

        # Slice coordinates and pass to NN
        batched_coords = coords.get_batch(idx)
        nn_output = nn_layer(batched_coords.internal)

        # Evaluate reference field at the batched coordinates
        reference_tensor_eval = reference_field.tensor(batched_coords)

        # Compute loss directly between raw NN tensor and sliced field tensor
        data_loss = torch.sum((nn_output - reference_tensor_eval) ** 2)

        # Analytical Verification of the Loss value
        # Hand-calculated total squared error sum = 3956.5
        expected_loss_value = 3956.5
        assert torch.isclose(
            data_loss, torch.tensor(expected_loss_value, device=device)
        )

        # Check pre-backward state
        assert nn_layer.weight.grad is None
        assert nn_layer.bias.grad is None

        # Backward pass
        data_loss.backward(retain_graph=True)  # type: ignore[no-untyped-call]

        # Verify gradients flowed back to the NN parameters
        assert nn_layer.weight.grad is not None
        assert nn_layer.bias.grad is not None

    def test_nn_raw_tensor_spatial_derivatives_and_physics_loss(
        self,
        setup_pinn_environment: tuple[CoordsProtocol, FieldProtocol, torch.nn.Linear],
        device: torch.device,
    ) -> None:
        """
        Physics Loss: Computes spatial derivatives (gradients of the raw NN outputs
        with respect to the individual batch coordinate tensors) directly.
        """
        coords, _, nn_layer = setup_pinn_environment
        idx = torch.tensor([0, 2], device=device)

        # Slice coordinates and pass to NN
        batched_coords = coords.get_batch(idx)
        nn_output = nn_layer(batched_coords.internal)

        # Extract the individual predicted components from the raw tensor
        u_x = nn_output[:, 0]
        u_y = nn_output[:, 1]

        # Target the underlying batch coordinate leaves for autograd tracking
        batch_inputs = (batched_coords.get_coord("x"), batched_coords.get_coord("y"))

        # Calculate spatial derivatives directly from the raw components
        du_x_dx = torch.autograd.grad(
            outputs=u_x.sum(),
            inputs=batch_inputs,
            retain_graph=True,
            allow_unused=False,
        )[0]

        du_y_dy = torch.autograd.grad(
            outputs=u_y.sum(),
            inputs=batch_inputs,
            retain_graph=False,
            allow_unused=False,
        )[1]

        # Analytical Verification:
        # u_x = 2.0 * x  ->  du_x/dx = 2.0
        # u_y = 3.0 * y  ->  du_y/dy = 3.0
        assert du_x_dx.shape == batched_coords.get_coord("x").shape
        assert du_y_dy.shape == batched_coords.get_coord("y").shape
        assert torch.allclose(
            du_x_dx, torch.ones_like(batched_coords.get_coord("x")) * 2.0
        )
        assert torch.allclose(
            du_y_dy, torch.ones_like(batched_coords.get_coord("y")) * 3.0
        )


class TestGetComponentErrors:
    def test_invalid_component_label_raises_keyerror(
        self,
        coords_2d: CoordsProtocol,
        device: torch.device,
    ) -> None:
        field = make_field(rank=2, coords=coords_2d, device=device)

        with pytest.raises(KeyError, match="not found in field components"):
            field.get_component("invalid_label")


# Validation errors
class TestFieldValidationErrors:
    def test_negative_rank_raises(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        with pytest.raises(ValueError, match="rank must be >= 0"):
            make_field_dataclass(rank=-1, coords=coords_2d, device=device)

    def test_mismatched_points_length_raises(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        Field = make_field_dataclass(rank=0, coords=coords_2d, device=device)
        # Create a tensor with length that does not match coords.n_points
        bad_length_tensor = torch.ones(coords_2d.n_points + 5, device=device)

        with pytest.raises(ValueError, match="must have shape"):
            Field(label="f", value=bad_length_tensor)

    def test_field_label_clashes_with_coordinate_label_raises(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        with pytest.raises(ValueError, match="clashes with a coordinate label"):
            make_field(rank=0, coords=coords_2d, device=device, label="x")

    def test_field_label_clashes_with_component_label_raises(
        self, coords_2d: CoordsProtocol, device: torch.device
    ) -> None:
        with pytest.raises(
            ValueError, match="clashes with a generated component label"
        ):
            make_field(rank=2, coords=coords_2d, device=device, label="x_x")
