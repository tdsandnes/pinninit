"""
Unit tests for loss/loss_function.py.
"""

from __future__ import annotations

import pytest
import torch

from pinninit.data_structure.batching import make_dataloader
from pinninit.data_structure.coords_dataclass import (
    CoordsProtocol,
    make_coords_dataclass,
)
from pinninit.data_structure.data_container import DataContainer
from pinninit.data_structure.field_dataclass import make_field_dataclass
from pinninit.geometry.coordinate_system import CoordSystem
from pinninit.loss.evaluation_context import EvaluationContext
from pinninit.loss.loss_function import (
    LOSS_FUNCTION_REGISTRY,
    LossFunctionLabels,
)
from tests.data_structure.fixtures import (
    N_POINTS,
    make_cartesian_2d,
    make_device,
    make_xy_2d,
)


def make_coords(
    coord_system: CoordSystem,
    x: torch.Tensor,
    y: torch.Tensor,
) -> CoordsProtocol:
    Coords = make_coords_dataclass(
        coord_labels={"x", "y"},
        coord_system=coord_system,
        device=x.device,
    )
    return Coords(x=x, y=y)


def make_output_container(
    coords: CoordsProtocol,
    field_label: str,
    device: torch.device,
) -> DataContainer:
    container = DataContainer(
        coords=coords,
        primary_field_label_to_rank={field_label: 0},
        derived_field_label_to_rank={},
        const_labels=None,
        equations=None,
    )
    container.set_field_from_tensor(
        field_label,
        torch.zeros(coords.n_points, device=device),
    )
    return container


def make_eval_context_with_weights(
    coords: CoordsProtocol,
    device: torch.device,
    weight_values: torch.Tensor,
) -> EvaluationContext:
    output_dc = make_output_container(coords, "u", device)
    Field = make_field_dataclass(rank=0, coords=coords, device=device)
    weights = Field(label="w", value=weight_values)
    dataloader = make_dataloader(coords=coords, batch_size=N_POINTS, shuffle=False)
    return EvaluationContext(
        output_data=output_dc,
        loss_function=LOSS_FUNCTION_REGISTRY[LossFunctionLabels.MSE],
        dataloader=dataloader,
        point_weights=weights,
    )


def make_eval_context_without_weights(
    coords: CoordsProtocol,
    device: torch.device,
) -> EvaluationContext:
    output_dc = make_output_container(coords, "u", device)
    dataloader = make_dataloader(coords=coords, batch_size=N_POINTS, shuffle=False)
    return EvaluationContext(
        output_data=output_dc,
        loss_function=LOSS_FUNCTION_REGISTRY[LossFunctionLabels.MSE],
        dataloader=dataloader,
    )


def scalar_residuals(N: int) -> torch.Tensor:
    x = torch.linspace(1.0, 5.0, N)
    return x.unsqueeze(1)


def vector_residuals(N: int) -> torch.Tensor:
    r1 = torch.linspace(1.0, 5.0, N)
    r2 = torch.linspace(2.0, 6.0, N)
    return torch.stack([r1, r2], dim=1)


def uniform_weights(N: int) -> torch.Tensor:
    return torch.ones(N)


def nontrivial_weights(N: int) -> torch.Tensor:
    return torch.linspace(0.5, 2.0, N)


@pytest.fixture  # type: ignore[misc]
def device() -> torch.device:
    return make_device()


@pytest.fixture  # type: ignore[misc]
def cartesian_2d(device: torch.device) -> CoordSystem:
    return make_cartesian_2d(device)


@pytest.fixture  # type: ignore[misc]
def xy_2d(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return make_xy_2d(device)


@pytest.fixture  # type: ignore[misc]
def coords(
    cartesian_2d: CoordSystem,
    xy_2d: tuple[torch.Tensor, torch.Tensor],
) -> CoordsProtocol:
    x, y = xy_2d
    return make_coords(cartesian_2d, x, y)


class TestMseLoss:
    def test_zero_residuals_give_zero_loss(self, device: torch.device) -> None:
        residuals = torch.zeros((N_POINTS, 1), device=device)
        assert torch.allclose(
            LOSS_FUNCTION_REGISTRY[LossFunctionLabels.MSE](residuals),
            torch.tensor(0.0, device=device),
        )

    def test_constant_residuals_give_squared_value(self, device: torch.device) -> None:
        residuals = torch.full((N_POINTS,), 3.0, device=device)
        assert torch.allclose(
            LOSS_FUNCTION_REGISTRY[LossFunctionLabels.MSE](residuals),
            torch.tensor(9.0, device=device),
        )

    def test_mixed_residuals_give_mean_of_squares(self, device: torch.device) -> None:
        residuals = scalar_residuals(4).to(device)
        expected = (residuals**2).mean(dim=0)
        assert torch.allclose(
            LOSS_FUNCTION_REGISTRY[LossFunctionLabels.MSE](residuals), expected
        )

    def test_accepts_none_context_and_coords(self, device: torch.device) -> None:
        residuals = torch.tensor([2.0, 4.0], device=device)
        loss = LOSS_FUNCTION_REGISTRY[LossFunctionLabels.MSE](
            residuals, eval_context=None, eval_coords=None
        )
        assert torch.allclose(loss, torch.tensor(10.0, device=device))

    def test_returns_scalar_component_tensor(self, device: torch.device) -> None:
        residuals = scalar_residuals(N_POINTS).to(device)
        assert LOSS_FUNCTION_REGISTRY[LossFunctionLabels.MSE](residuals).ndim == 1

    def test_single_point_residual(self, device: torch.device) -> None:
        residuals = torch.tensor([5.0], device=device)
        assert torch.allclose(
            LOSS_FUNCTION_REGISTRY[LossFunctionLabels.MSE](residuals),
            torch.tensor(25.0, device=device),
        )

    def test_returns_per_component_values(self, coords: CoordsProtocol) -> None:
        N = coords.n_points
        residuals = vector_residuals(N).to(coords.device)

        result = LOSS_FUNCTION_REGISTRY[LossFunctionLabels.MSE](residuals)
        expected = (residuals**2).mean(dim=0)

        assert result.shape == (2,)
        assert torch.allclose(result, expected)


class TestWeightedMeanMseLoss:
    def test_uniform_unit_weights_match_standard_mse(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        weight_values = uniform_weights(N_POINTS).to(device)
        eval_context = make_eval_context_with_weights(coords, device, weight_values)
        residuals = scalar_residuals(N_POINTS).to(device)

        result = LOSS_FUNCTION_REGISTRY[LossFunctionLabels.WEIGHTED_MEAN_MSE](
            residuals, eval_context, coords
        )
        expected = LOSS_FUNCTION_REGISTRY[LossFunctionLabels.MSE](residuals)

        assert torch.allclose(result, expected)

    def test_zero_weight_excludes_point_from_loss(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        weight_values = uniform_weights(N_POINTS).to(device)
        weight_values[0] = 0.0
        eval_context = make_eval_context_with_weights(coords, device, weight_values)

        residuals = scalar_residuals(N_POINTS).to(device)
        residuals[0] = 100.0

        result = LOSS_FUNCTION_REGISTRY[LossFunctionLabels.WEIGHTED_MEAN_MSE](
            residuals, eval_context, coords
        )
        expected = LOSS_FUNCTION_REGISTRY[LossFunctionLabels.MSE](residuals[1:])
        assert torch.allclose(result, expected)

    def test_doubled_weights_leave_loss_unchanged(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        unit_weight_values = uniform_weights(N_POINTS).to(device)
        double_weight_values = 2 * unit_weight_values

        residuals = scalar_residuals(N_POINTS).to(device)

        unit_ctx = make_eval_context_with_weights(coords, device, unit_weight_values)
        double_ctx = make_eval_context_with_weights(
            coords, device, double_weight_values
        )

        assert torch.allclose(
            LOSS_FUNCTION_REGISTRY[LossFunctionLabels.WEIGHTED_MEAN_MSE](
                residuals, unit_ctx, coords
            ),
            LOSS_FUNCTION_REGISTRY[LossFunctionLabels.WEIGHTED_MEAN_MSE](
                residuals, double_ctx, coords
            ),
        )

    def test_missing_point_weights_raises(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        eval_context = make_eval_context_without_weights(coords, device)
        residuals = scalar_residuals(N_POINTS).to(device)
        with pytest.raises(ValueError, match="point_weights"):
            LOSS_FUNCTION_REGISTRY[LossFunctionLabels.WEIGHTED_MEAN_MSE](
                residuals, eval_context, coords
            )

    def test_residual_length_mismatch_raises(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        weight_values = uniform_weights(N_POINTS).to(device)
        eval_context = make_eval_context_with_weights(coords, device, weight_values)
        wrong_length_residuals = scalar_residuals(N_POINTS + 2).to(device)
        with pytest.raises(ValueError, match="must match"):
            LOSS_FUNCTION_REGISTRY[LossFunctionLabels.WEIGHTED_MEAN_MSE](
                wrong_length_residuals, eval_context, coords
            )

    def test_returns_scalar_component_tensor(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        weight_values = uniform_weights(N_POINTS).to(device)
        eval_context = make_eval_context_with_weights(coords, device, weight_values)
        residuals = scalar_residuals(N_POINTS).to(device)
        assert (
            LOSS_FUNCTION_REGISTRY[LossFunctionLabels.WEIGHTED_MEAN_MSE](
                residuals, eval_context, coords
            ).ndim
            == 1
        )

    def test_returns_per_component_values(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        weight_values = nontrivial_weights(N_POINTS).to(device)
        eval_context = make_eval_context_with_weights(
            coords,
            device,
            weight_values,
        )
        residuals = vector_residuals(N_POINTS).to(device)

        result = LOSS_FUNCTION_REGISTRY[LossFunctionLabels.WEIGHTED_MEAN_MSE](
            residuals, eval_context, coords
        )

        w_sum = weight_values.sum()
        expected = (weight_values.unsqueeze(1) * (residuals**2)).sum(dim=0) / w_sum

        assert result.shape == (2,)
        assert torch.allclose(result, expected)


class TestWeightedPointAvgMseLoss:
    def test_uniform_unit_weights_match_standard_mse(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        weight_values = uniform_weights(N_POINTS).to(device)
        eval_context = make_eval_context_with_weights(coords, device, weight_values)
        residuals = scalar_residuals(N_POINTS).to(device)

        result = LOSS_FUNCTION_REGISTRY[LossFunctionLabels.WEIGHTED_POINT_AVERAGED_MSE](
            residuals, eval_context, coords
        )
        expected = LOSS_FUNCTION_REGISTRY[LossFunctionLabels.MSE](residuals)

        assert torch.allclose(result, expected)

    def test_constant_weight_scales_loss(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        weight_values = 2 * uniform_weights(N_POINTS).to(device)
        eval_context = make_eval_context_with_weights(coords, device, weight_values)
        residuals = scalar_residuals(N_POINTS).to(device)

        result = LOSS_FUNCTION_REGISTRY[LossFunctionLabels.WEIGHTED_POINT_AVERAGED_MSE](
            residuals, eval_context, coords
        )
        expected = 2 * LOSS_FUNCTION_REGISTRY[LossFunctionLabels.MSE](residuals)
        assert torch.allclose(result, expected)

    def test_differs_from_weighted_mean_mse_with_non_uniform_weights(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        weight_values = nontrivial_weights(N_POINTS).to(device)
        residuals = scalar_residuals(N_POINTS).to(device)

        point_avg_ctx = make_eval_context_with_weights(coords, device, weight_values)
        mean_ctx = make_eval_context_with_weights(coords, device, weight_values)

        point_avg_result = LOSS_FUNCTION_REGISTRY[
            LossFunctionLabels.WEIGHTED_POINT_AVERAGED_MSE
        ](residuals, point_avg_ctx, coords)
        mean_result = LOSS_FUNCTION_REGISTRY[LossFunctionLabels.WEIGHTED_MEAN_MSE](
            residuals, mean_ctx, coords
        )

        assert not torch.allclose(point_avg_result, mean_result)

    def test_missing_point_weights_raises(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        eval_context = make_eval_context_without_weights(coords, device)
        residuals = scalar_residuals(N_POINTS).to(device)
        with pytest.raises(ValueError, match="point_weights"):
            LOSS_FUNCTION_REGISTRY[LossFunctionLabels.WEIGHTED_POINT_AVERAGED_MSE](
                residuals, eval_context, coords
            )

    def test_residual_length_mismatch_raises(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        weight_values = uniform_weights(N_POINTS).to(device)
        eval_context = make_eval_context_with_weights(coords, device, weight_values)
        wrong_length_residuals = scalar_residuals(N_POINTS + 2).to(device)
        with pytest.raises(ValueError, match="must match"):
            LOSS_FUNCTION_REGISTRY[LossFunctionLabels.WEIGHTED_POINT_AVERAGED_MSE](
                wrong_length_residuals, eval_context, coords
            )

    def test_returns_scalar_tensor(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        weight_values = uniform_weights(N_POINTS).to(device)
        eval_context = make_eval_context_with_weights(coords, device, weight_values)
        residuals = scalar_residuals(N_POINTS).to(device)
        assert (
            LOSS_FUNCTION_REGISTRY[LossFunctionLabels.WEIGHTED_POINT_AVERAGED_MSE](
                residuals, eval_context, coords
            ).ndim
            == 1
        )

    def test_returns_per_component_values(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        N = coords.n_points
        weight_values = nontrivial_weights(N).to(device)
        eval_context = make_eval_context_with_weights(coords, device, weight_values)
        residuals = vector_residuals(N).to(device)

        result = LOSS_FUNCTION_REGISTRY[LossFunctionLabels.WEIGHTED_POINT_AVERAGED_MSE](
            residuals, eval_context, coords
        )
        expected = (weight_values.unsqueeze(1) * (residuals**2)).mean(dim=0)

        assert result.shape == (2,)
        assert torch.allclose(result, expected)


class TestLossFunctionRegistry:
    @pytest.mark.parametrize("label", list(LossFunctionLabels))  # type: ignore[misc]
    def test_registry_contains_callable(self, label: LossFunctionLabels) -> None:
        loss_function = LOSS_FUNCTION_REGISTRY[label]
        assert callable(loss_function)

    def test_registry_has_no_missing_or_extra_entries(self) -> None:
        assert set(LOSS_FUNCTION_REGISTRY) == set(LossFunctionLabels)

    @pytest.mark.parametrize("label", list(LossFunctionLabels))  # type: ignore[misc]
    def test_loss_functions_preserve_component_axis(
        self,
        label: LossFunctionLabels,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        N_components = 10

        residuals = torch.randn(N_POINTS, N_components, device=device)

        weight_values = torch.ones(N_POINTS, device=device)
        loss_fn = LOSS_FUNCTION_REGISTRY[label]

        eval_context = EvaluationContext(
            output_data=make_output_container(coords, "u", device),
            loss_function=loss_fn,
            dataloader=make_dataloader(
                coords=coords, batch_size=N_POINTS, shuffle=False
            ),
            point_weights=make_field_dataclass(rank=0, coords=coords, device=device)(
                label="w",
                value=weight_values,
            ),
        )

        out = eval_context.loss_function(residuals, eval_context, coords)
        assert out.shape == (N_components,)
