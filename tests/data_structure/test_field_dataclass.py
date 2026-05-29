"""Unit tests for field_dataclass / make_field_dataclass."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pinninit.data_structure.coords_dataclass import make_coords_dataclass
from pinninit.data_structure.field_dataclass import (
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
        parent: MockCoords | None = None,
        parent_idx: torch.Tensor | None = None,
    ):
        self.geometric_labels = geometric_labels
        self.n_points = n_points
        self.device = device
        self.is_batch = is_batch
        self.parent = parent
        self.parent_idx = parent_idx


# Local fixtures
@pytest.fixture
def device():
    return make_device()


@pytest.fixture
def n_points():
    return N_POINTS


@pytest.fixture
def coords_2d(device, n_points):
    return MockCoords(geometric_labels=("x", "y"), n_points=n_points, device=device)


@pytest.fixture
def setup_pinn_environment(device):
    """Bridge fixture that calls the factory function with required dependencies."""
    return make_pinn_environment(
        make_coords_dataclass=make_coords_dataclass,
        make_field_dataclass=make_field_dataclass,
        device=device,
    )


# Helpers
def make_field(rank, coords, device):
    Field = make_field_dataclass(rank=rank, coords=coords, device=device)

    labels = generate_tensor_component_labels(coords.geometric_labels, rank)

    values = {
        label: torch.linspace(1.0, 5.0, coords.n_points, device=device)
        for label in labels
    }

    return Field(**values)


# Construction and basic attributes
class TestFieldConstruction:
    def test_component_labels(self, coords_2d, device):
        Field = make_field_dataclass(rank=1, coords=coords_2d, device=device)
        field = Field(x=torch.ones(coords_2d.n_points, device=device),
                      y=torch.ones(coords_2d.n_points, device=device))

        assert field.component_labels == ("x", "y")

    def test_coords_is_bound_to_class(self, coords_2d, device):
        FieldClass = make_field_dataclass(rank=1, coords=coords_2d, device=device)
        assert FieldClass.coords == coords_2d


# Tensor operations and shapes
class TestFieldTensorShapes:
    @pytest.mark.parametrize(
        ("rank", "comp_shape", "tensor_shape"),
        [
            (0, (N_POINTS, 1), (N_POINTS,)),
            (1, (N_POINTS, 2), (N_POINTS, 2)),
            (2, (N_POINTS, 4), (N_POINTS, 2, 2)),
            (3, (N_POINTS, 8), (N_POINTS, 2, 2, 2)),
        ],
    )
    def test_shapes(self, rank, comp_shape, tensor_shape, coords_2d, device):

        field = make_field(rank, coords_2d, device)

        ct = field.component_tensor(coords_2d)
        st = field.tensor(coords_2d)

        assert ct.shape == comp_shape
        assert st.shape == tensor_shape


# State updates
class TestFieldUpdateMethods:
    def test_update_from_component_dict_rank_1(self, coords_2d, device):
        field = make_field(rank=1, coords=coords_2d, device=device)

        new_x = torch.zeros_like(field.x)
        field.update_from_component_dict({"x": new_x})

        assert torch.allclose(field.x, new_x)

    def test_update_from_component_tensor_rank_1(self, coords_2d, device):
        field = make_field(rank=1, coords=coords_2d, device=device)

        new_tensor = torch.ones_like(field.component_tensor(coords_2d)) * 8.0
        field.update_from_component_tensor(new_tensor)

        assert torch.allclose(field.x, torch.ones_like(field.x) * 8.0)
        assert torch.allclose(field.y, torch.ones_like(field.y) * 8.0)

    def test_update_from_tensor_rank_1(self, coords_2d, device):
        field = make_field(rank=1, coords=coords_2d, device=device)

        # Ensure x-slice values differ entirely from y-slice values
        base_tensor = field.tensor(coords_2d)
        new_tensor = torch.zeros_like(base_tensor)
        new_tensor[:, 0] = 12.0  # target x updating
        new_tensor[:, 1] = 99.0  # target y updating
        
        field.update_from_tensor(new_tensor)
        
        assert torch.allclose(field.tensor(coords_2d), new_tensor)
        assert torch.allclose(field.x, new_tensor[:, 0])
        assert torch.allclose(field.y, new_tensor[:, 1])


class TestGenerateTensorComponentLabels:
    def test_rank_0_returns_value(self):
        labels = generate_tensor_component_labels(("x", "y"), rank=0)

        assert labels == ("value",)

    def test_rank_1_labels_cartesian_2d(self):
        labels = generate_tensor_component_labels(("x", "y"), rank=1)

        assert labels == ("x", "y")

    def test_rank_1_labels_spherical(self):
        labels = generate_tensor_component_labels(("r", "theta", "phi"), rank=1)

        assert labels == ("r", "theta", "phi")

    def test_rank_2_labels_3d_cartesian(self):
        labels = generate_tensor_component_labels(("x", "y", "z"), rank=2)

        assert labels == ("x_x", "x_y", "x_z", "y_x", "y_y", "y_z", "z_x", "z_y", "z_z")

    def test_rank_2_labels_spherical(self):
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

    def test_rank_3_labels_2d_cartesian(self):
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

    def test_evaluation_with_real_nested_sub_batch_coords_slices(self, device):
        coord_system = make_cartesian_2d(device=device)
        Coords = make_coords_dataclass(("x", "y"), device=device)
        x = torch.tensor([10.0, 20.0, 30.0, 40.0], device=device, requires_grad=True)
        y = torch.tensor(
            [100.0, 200.0, 300.0, 400.0], device=device, requires_grad=True
        )
        coords = Coords(coord_system=coord_system, x=x, y=y)

        # Build a real vector field bound to these coordinates
        f_x = torch.tensor([10.0, 20.0, 30.0, 40.0], device=device, requires_grad=True)
        f_y = torch.tensor([50.0, 60.0, 70.0, 80.0], device=device, requires_grad=True)
        Field = make_field_dataclass(rank=1, coords=coords, device=device)
        field_instance = Field(x=f_x, y=f_y)

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

    def test_evaluation_with_invalid_sub_batch_raises(self, coords_2d, device):
        field_instance = make_field(rank=1, coords=coords_2d, device=device)

        # Mock coord that claims to be a variant but lacks batch validation flags
        invalid_coords = MockCoords(
            geometric_labels=("x", "y"), n_points=2, device=device, is_batch=False
        )

        with pytest.raises(
            ValueError, match="Evaluation coordinates must be a valid subset"
        ):
            field_instance.component_tensor(invalid_coords)

    def test_component_tensor_grad_does_not_flow_to_field_components(self, device):
        coord_system = make_cartesian_2d(device=device)
        Coords = make_coords_dataclass(("x", "y"), device=device)
        x = torch.linspace(0.1, 0.9, 4, device=device, requires_grad=True)
        y = torch.linspace(0.2, 0.8, 4, device=device, requires_grad=True)
        coords = Coords(coord_system=coord_system, x=x, y=y)

        f_x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, requires_grad=True)
        f_y = torch.tensor([5.0, 6.0, 7.0, 8.0], device=device, requires_grad=True)
        Field = make_field_dataclass(rank=1, coords=coords, device=device)
        field_instance = Field(x=f_x, y=f_y)

        ct = field_instance.component_tensor(coords)
        loss = ct.sum()
        loss.backward()

        # detach in _loop_parent_component_tensor severs the graph
        assert f_x.grad is None
        assert f_y.grad is None


class TestFieldNeuralNetworkIntegration:
    def test_nn_raw_tensor_evaluation_and_data_loss(
        self, setup_pinn_environment, device
    ):
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

        # Backward pass
        data_loss.backward()

        # Verify gradients flowed back to the NN parameters
        assert nn_layer.weight.grad is not None
        assert nn_layer.bias.grad is not None
        assert not torch.all(nn_layer.weight.grad == 0.0)

    def test_nn_raw_tensor_spatial_derivatives_and_physics_loss(
        self, setup_pinn_environment, device
    ):
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
        batch_inputs = (batched_coords.x, batched_coords.y)

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
        assert du_x_dx.shape == batched_coords.x.shape
        assert du_y_dy.shape == batched_coords.y.shape
        assert torch.allclose(du_x_dx, torch.ones_like(batched_coords.x) * 2.0)
        assert torch.allclose(du_y_dy, torch.ones_like(batched_coords.y) * 3.0)


# Validation errors
class TestFieldValidationErrors:
    def test_negative_rank_raises(self, coords_2d, device):
        # This matches the exception raised inside generate_tensor_component_labels
        with pytest.raises(ValueError, match="rank must be >= 0"):
            make_field_dataclass(rank=-1, coords=coords_2d, device=device)

    @pytest.mark.parametrize(
        "invalid_input",
        [
            [1.0, 2.0, 3.0],
            (1.0, 2.0, 3.0),
            np.array([1.0, 2.0, 3.0]),
        ],
    )
    def test_invalid_component_type_raises(self, coords_2d, device, invalid_input):
        Field = make_field_dataclass(rank=0, coords=coords_2d, device=device)

        with pytest.raises(TypeError, match="torch.Tensor"):
            Field(value=invalid_input)

    def test_mismatched_points_length_raises(self, coords_2d, device):
        Field = make_field_dataclass(rank=0, coords=coords_2d, device=device)
        # Create a tensor with length that does not match coords.n_points
        bad_length_tensor = torch.ones(coords_2d.n_points + 5, device=device)

        with pytest.raises(ValueError, match="got shape"):
            Field(value=bad_length_tensor)

    def test_update_from_component_tensor_non_tensor_raises(self, coords_2d, device):
        field_instance = make_field(rank=1, coords=coords_2d, device=device)

        with pytest.raises(TypeError, match="not a torch.Tensor"):
            field_instance.update_from_component_tensor(
                [[1.0, 2.0]] * coords_2d.n_points
            )

