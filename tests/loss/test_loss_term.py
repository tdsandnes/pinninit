"""
Unit tests for loss/loss_term.py
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
from pinninit.loss.loss_function import LOSS_FUNCTION_REGISTRY, LossFunctionLabels
from pinninit.loss.loss_term import (
    LOSS_TERM_REGISTRY,
    DataLoss,
    LossTerm,
    LossTermLabels,
    PhysicsLoss,
)
from tests.data_structure.fixtures import (
    N_POINTS,
    make_cartesian_2d,
    make_device,
    make_xy_2d,
)


def make_coords(
    coord_system: CoordSystem, x: torch.Tensor, y: torch.Tensor
) -> CoordsProtocol:
    Coords = make_coords_dataclass(
        coord_labels={"x", "y"},
        coord_system=coord_system,
        device=x.device,
    )
    return Coords(x=x, y=y)


def make_container(
    coords: CoordsProtocol, device: torch.device, value: float
) -> DataContainer:
    dc = DataContainer(
        coords=coords,
        primary_field_label_to_rank={"u": 0},
        derived_field_label_to_rank={},
        const_labels=None,
        equations=None,
    )
    dc.set_field_from_tensor(
        "u",
        torch.full((coords.n_points,), value, device=device),
    )
    return dc


def make_context(
    coords: CoordsProtocol,
    device: torch.device,
    input_value: float = 1.0,
    output_value: float = 2.0,
) -> EvaluationContext:
    input_dc = make_container(coords, device, input_value)
    output_dc = make_container(coords, device, output_value)

    dataloader = make_dataloader(coords=coords, batch_size=N_POINTS, shuffle=False)

    return EvaluationContext(
        input_data=input_dc,
        output_data=output_dc,
        dataloader=dataloader,
        loss_function=LOSS_FUNCTION_REGISTRY[LossFunctionLabels.MSE],
    )


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
    cartesian_2d: CoordSystem, xy_2d: tuple[torch.Tensor, torch.Tensor]
) -> CoordsProtocol:
    x, y = xy_2d
    Coords = make_coords_dataclass(
        coord_labels={"x", "y"},
        coord_system=cartesian_2d,
        device=x.device,
    )
    return Coords(x=x, y=y)


class TestDataLoss:
    def test_residual_is_prediction_minus_target(
        self, coords: CoordsProtocol, device: torch.device
    ) -> None:
        ctx = make_context(coords, device, input_value=1.0, output_value=3.0)

        term = DataLoss(field_label="u")
        residual = term.compute_residual(ctx, coords)
        expected_residual = torch.full((coords.n_points, 1), 2.0, device=device)

        assert torch.allclose(residual, expected_residual)

    def test_missing_input_data_raises(
        self, coords: CoordsProtocol, device: torch.device
    ) -> None:
        output_dc = make_container(coords, device, 2.0)

        ctx = EvaluationContext(
            input_data=None,
            output_data=output_dc,
            dataloader=make_dataloader(
                coords=coords, batch_size=N_POINTS, shuffle=False
            ),
            loss_function=LOSS_FUNCTION_REGISTRY[LossFunctionLabels.MSE],
        )

        term = DataLoss(field_label="u")

        with pytest.raises(ValueError, match="input_data"):
            term.compute_residual(ctx, coords)

    def test_compute_loss_passes_through_loss_function(
        self, coords: CoordsProtocol, device: torch.device
    ) -> None:
        ctx = make_context(coords, device, input_value=1.0, output_value=3.0)

        term = DataLoss(field_label="u")
        loss = term.compute_loss(ctx, coords)

        expected_residual = torch.full((coords.n_points,), 2.0, device=device)
        expected = ctx.loss_function(expected_residual, ctx, coords)

        assert torch.allclose(loss, expected)


class TestPhysicsLoss:
    def test_residual_is_output_only(
        self, coords: CoordsProtocol, device: torch.device
    ) -> None:
        ctx = make_context(coords, device, output_value=4.0, input_value=999.0)

        term = PhysicsLoss(field_label="u")
        residual = term.compute_residual(ctx, coords)

        assert torch.allclose(
            residual, torch.full((coords.n_points,), 4.0, device=device)
        )

    def test_compute_loss_uses_output_only(
        self, coords: CoordsProtocol, device: torch.device
    ) -> None:
        ctx = make_context(coords, device, output_value=4.0)

        term = PhysicsLoss(field_label="u")
        loss = term.compute_loss(ctx, coords)

        expected_residual = torch.full((coords.n_points,), 4.0, device=device)
        expected = ctx.loss_function(expected_residual, ctx, coords)

        assert torch.allclose(loss, expected)


class TestLossTermRegistry:
    def test_registry_matches_enum(self) -> None:
        assert set(LOSS_TERM_REGISTRY.keys()) == set(LossTermLabels)

    @pytest.mark.parametrize("label", list(LossTermLabels))  # type: ignore[misc]
    def test_registry_returns_callable_class(self, label: LossTermLabels) -> None:
        term_cls = LOSS_TERM_REGISTRY[label]
        assert callable(term_cls)

    @pytest.mark.parametrize("label", list(LossTermLabels))  # type: ignore[misc]
    def test_instantiation_from_registry(self, label: LossTermLabels) -> None:
        term_cls = LOSS_TERM_REGISTRY[label]
        term = term_cls(field_label="u")
        assert isinstance(term, term_cls)


class TestLossTermShapePropagation:
    @pytest.mark.parametrize("label", list(LossTermLabels))  # type: ignore[misc]
    def test_registry_classes_are_loss_terms(self, label: LossTermLabels) -> None:
        cls = LOSS_TERM_REGISTRY[label]
        assert issubclass(cls, LossTerm)

    @pytest.mark.parametrize("label", list(LossTermLabels))  # type: ignore[misc]
    def test_compute_loss_preserves_field_component_structure_scalar(
        self,
        label: LossTermLabels,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        TermCls = LOSS_TERM_REGISTRY[label]

        scalar_output = make_container(coords, device, value=1.0)
        scalar_input = make_container(coords, device, value=0.5)

        ctx = EvaluationContext(
            input_data=scalar_input,
            output_data=scalar_output,
            dataloader=make_dataloader(
                coords=coords, batch_size=N_POINTS, shuffle=False
            ),
            loss_function=LOSS_FUNCTION_REGISTRY[LossFunctionLabels.MSE],
        )

        term = TermCls(field_label="u")

        residual_scalar = term.compute_residual(ctx, coords)
        assert residual_scalar.shape == (coords.n_points, 1)

        loss_scalar = term.compute_loss(ctx, coords)
        assert loss_scalar.shape == (1,)

    @pytest.mark.parametrize("label", list(LossTermLabels))  # type: ignore[misc]
    def test_compute_loss_preserves_field_component_structure_vector(
        self,
        label: LossTermLabels,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        TermCls = LOSS_TERM_REGISTRY[label]

        vector_output = DataContainer(
            coords=coords,
            primary_field_label_to_rank={"v": 1},
            derived_field_label_to_rank={},
            const_labels=None,
            equations=None,
        )

        vector_input = DataContainer(
            coords=coords,
            primary_field_label_to_rank={"v": 1},
            derived_field_label_to_rank={},
            const_labels=None,
            equations=None,
        )

        n_components = len(
            make_field_dataclass(rank=1, coords=coords, device=device)(
                label="v"
            ).component_labels
        )

        v_pred = torch.randn(coords.n_points, n_components, device=device)
        v_true = torch.randn(coords.n_points, n_components, device=device)

        vector_output.set_field_from_component_tensor("v", v_pred)
        vector_input.set_field_from_component_tensor("v", v_true)

        ctx_vec = EvaluationContext(
            input_data=vector_input,
            output_data=vector_output,
            dataloader=make_dataloader(
                coords=coords, batch_size=N_POINTS, shuffle=False
            ),
            loss_function=LOSS_FUNCTION_REGISTRY[LossFunctionLabels.MSE],
        )

        term_vec = TermCls(field_label="v")

        residual_vec = term_vec.compute_residual(ctx_vec, coords)
        assert residual_vec.shape == (coords.n_points, n_components)

        loss_vec = term_vec.compute_loss(ctx_vec, coords)
        assert loss_vec.shape == (n_components,)

        expected = ctx_vec.loss_function(residual_vec, ctx_vec, coords)
        assert torch.allclose(loss_vec, expected)
