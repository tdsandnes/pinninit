"""
Unit tests for loss/evaluation_context.py.
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
from pinninit.data_structure.field_dataclass import FieldProtocol, make_field_dataclass
from pinninit.geometry.coordinate_system import CoordSystem
from pinninit.loss.evaluation_context import EvaluationContext
from pinninit.loss.loss_function import mse_loss
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


def make_data_container(
    coords: CoordsProtocol,
    field_label_to_rank: dict[str, int],
) -> DataContainer:
    return DataContainer(
        coords=coords,
        primary_field_label_to_rank=field_label_to_rank,
        derived_field_label_to_rank={},
        const_labels=None,
        equations=None,
    )


def make_output_container(
    coords: CoordsProtocol,
    field_label: str,
    device: torch.device,
    fill: float = 0.0,
) -> DataContainer:
    container = make_data_container(coords, {field_label: 0})
    container.set_field_from_tensor(
        field_label,
        torch.full((coords.n_points,), fill, device=device),
    )
    return container


def make_input_container(
    coords: CoordsProtocol,
    field_label: str,
    device: torch.device,
    fill: float = 1.0,
) -> DataContainer:
    container = make_data_container(coords, {field_label: 0})
    container.set_field_from_tensor(
        field_label,
        torch.full((coords.n_points,), fill, device=device),
    )
    return container


def make_scalar_field(
    coords: CoordsProtocol,
    device: torch.device,
    fill: float = 1.0,
    label: str = "w",
) -> FieldProtocol:
    Field = make_field_dataclass(rank=0, coords=coords, device=device)
    return Field(
        label=label,
        value=torch.full((coords.n_points,), fill, device=device),
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
    cartesian_2d: CoordSystem,
    xy_2d: tuple[torch.Tensor, torch.Tensor],
) -> CoordsProtocol:
    x, y = xy_2d
    return make_coords(cartesian_2d, x, y)


class TestEvaluationContextConstruction:
    def test_coords_derived_from_output_data(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        output_dc = make_output_container(coords, "u", device)
        dataloader = make_dataloader(coords=coords, batch_size=N_POINTS, shuffle=False)
        ctx = EvaluationContext(
            output_data=output_dc,
            loss_function=mse_loss,
            dataloader=dataloader,
        )
        assert ctx.coords is coords

    def test_input_data_none_is_valid(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        output_dc = make_output_container(coords, "u", device)
        dataloader = make_dataloader(coords=coords, batch_size=N_POINTS, shuffle=False)
        ctx = EvaluationContext(
            output_data=output_dc,
            loss_function=mse_loss,
            dataloader=dataloader,
            input_data=None,
        )
        assert ctx.input_data is None

    def test_input_and_output_sharing_coords_is_valid(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        output_dc = make_output_container(coords, "u", device)
        input_dc = make_input_container(coords, "u", device)
        dataloader = make_dataloader(coords=coords, batch_size=N_POINTS, shuffle=False)
        ctx = EvaluationContext(
            output_data=output_dc,
            loss_function=mse_loss,
            dataloader=dataloader,
            input_data=input_dc,
        )
        assert ctx.input_data is not None
        assert ctx.input_data.coords is ctx.output_data.coords

    def test_input_output_mismatched_coords_raises(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        x, y = xy_2d
        coords_a = make_coords(cartesian_2d, x, y)
        coords_b = make_coords(cartesian_2d, x, y)

        output_dc = make_output_container(coords_a, "u", device)
        input_dc = make_input_container(coords_b, "u", device)
        dataloader = make_dataloader(
            coords=coords_a, batch_size=N_POINTS, shuffle=False
        )

        with pytest.raises(ValueError, match="identical coordinates"):
            EvaluationContext(
                output_data=output_dc,
                loss_function=mse_loss,
                dataloader=dataloader,
                input_data=input_dc,
            )

    def test_point_weights_none_by_default(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        output_dc = make_output_container(coords, "u", device)
        dataloader = make_dataloader(coords=coords, batch_size=N_POINTS, shuffle=False)
        ctx = EvaluationContext(
            output_data=output_dc,
            loss_function=mse_loss,
            dataloader=dataloader,
        )
        assert ctx.point_weights is None

    def test_scalar_rank_zero_point_weights_accepted(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        output_dc = make_output_container(coords, "u", device)
        weights = make_scalar_field(coords, device)
        dataloader = make_dataloader(coords=coords, batch_size=N_POINTS, shuffle=False)
        ctx = EvaluationContext(
            output_data=output_dc,
            loss_function=mse_loss,
            dataloader=dataloader,
            point_weights=weights,
        )
        assert ctx.point_weights is weights

    def test_rank_one_point_weights_raises(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        output_dc = make_output_container(coords, "u", device)
        Field = make_field_dataclass(rank=1, coords=coords, device=device)
        vector_field = Field(label="w")
        dataloader = make_dataloader(coords=coords, batch_size=N_POINTS, shuffle=False)

        with pytest.raises(ValueError, match="rank=0"):
            EvaluationContext(
                output_data=output_dc,
                loss_function=mse_loss,
                dataloader=dataloader,
                point_weights=vector_field,
            )

    def test_point_weights_on_different_coords_raises(
        self,
        cartesian_2d: CoordSystem,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        x, y = xy_2d
        coords_a = make_coords(cartesian_2d, x, y)
        coords_b = make_coords(cartesian_2d, x, y)

        output_dc = make_output_container(coords_a, "u", device)
        weights_on_b = make_scalar_field(coords_b, device)
        dataloader = make_dataloader(
            coords=coords_a, batch_size=N_POINTS, shuffle=False
        )

        with pytest.raises(ValueError, match="coords mismatch"):
            EvaluationContext(
                output_data=output_dc,
                loss_function=mse_loss,
                dataloader=dataloader,
                point_weights=weights_on_b,
            )

    def test_loss_function_stored(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        output_dc = make_output_container(coords, "u", device)
        dataloader = make_dataloader(coords=coords, batch_size=N_POINTS, shuffle=False)
        ctx = EvaluationContext(
            output_data=output_dc,
            loss_function=mse_loss,
            dataloader=dataloader,
        )
        assert ctx.loss_function is mse_loss

    def test_dataloader_stored(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        output_dc = make_output_container(coords, "u", device)
        dataloader = make_dataloader(coords=coords, batch_size=N_POINTS, shuffle=False)
        ctx = EvaluationContext(
            output_data=output_dc,
            loss_function=mse_loss,
            dataloader=dataloader,
        )
        assert ctx.dataloader is dataloader


class TestEvaluationContextUpdatePointWeights:
    def test_update_replaces_stored_weight_values(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        output_dc = make_output_container(coords, "u", device)
        weights = make_scalar_field(coords, device, fill=1.0)
        dataloader = make_dataloader(coords=coords, batch_size=N_POINTS, shuffle=False)
        ctx = EvaluationContext(
            output_data=output_dc,
            loss_function=mse_loss,
            dataloader=dataloader,
            point_weights=weights,
        )

        new_values = torch.full((coords.n_points,), 5.0, device=device)
        ctx.update_point_weights(new_values)

        assert ctx.point_weights is not None
        stored = ctx.point_weights.tensor(coords)
        assert torch.allclose(stored, new_values)

    def test_update_with_batch_coords_writes_to_subset(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        output_dc = make_output_container(coords, "u", device)
        weights = make_scalar_field(coords, device, fill=0.0)
        dataloader = make_dataloader(coords=coords, batch_size=N_POINTS, shuffle=False)
        ctx = EvaluationContext(
            output_data=output_dc,
            loss_function=mse_loss,
            dataloader=dataloader,
            point_weights=weights,
        )

        idx = torch.tensor([0, 2, 4], dtype=torch.long, device=device)
        batch_coords = coords.get_batch(idx)
        batch_values = torch.full((3,), 9.0, device=device)
        ctx.update_point_weights(batch_values, coords=batch_coords)

        assert ctx.point_weights is not None
        full_weights = ctx.point_weights.tensor(coords)
        assert torch.allclose(full_weights[idx], batch_values)

    def test_update_wrong_shape_raises(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        output_dc = make_output_container(coords, "u", device)
        weights = make_scalar_field(coords, device, fill=1.0)
        dataloader = make_dataloader(coords=coords, batch_size=N_POINTS, shuffle=False)
        ctx = EvaluationContext(
            output_data=output_dc,
            loss_function=mse_loss,
            dataloader=dataloader,
            point_weights=weights,
        )

        wrong_shape = torch.ones(coords.n_points + 5, device=device)
        with pytest.raises(ValueError, match="weights shape"):
            ctx.update_point_weights(wrong_shape)

    def test_update_when_point_weights_none_raises(
        self,
        coords: CoordsProtocol,
        device: torch.device,
    ) -> None:
        output_dc = make_output_container(coords, "u", device)
        dataloader = make_dataloader(coords=coords, batch_size=N_POINTS, shuffle=False)
        ctx = EvaluationContext(
            output_data=output_dc,
            loss_function=mse_loss,
            dataloader=dataloader,
        )
        new_values = torch.ones(coords.n_points, device=device)
        with pytest.raises(ValueError, match="not initialised"):
            ctx.update_point_weights(new_values)
