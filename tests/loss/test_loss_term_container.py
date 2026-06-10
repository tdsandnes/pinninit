"""
Unit tests for loss/loss_term_container.py
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
from pinninit.geometry.coordinate_system import CoordSystem
from pinninit.geometry.domain import Domain
from pinninit.loss.evaluation_context import EvaluationContext
from pinninit.loss.loss_function import LOSS_FUNCTION_REGISTRY, LossFunctionLabels
from pinninit.loss.loss_term import DataLoss, PhysicsLoss
from pinninit.loss.loss_term_container import LossTermContainer
from pinninit.nn.neural_network import NeuralNetwork
from tests.data_structure.fixtures import (
    BATCH_SIZE,
    N_POINTS,
    make_cartesian_2d,
    make_device,
    make_domain_2d_cartesian,
    make_xy_2d,
)


class ConstantNN(NeuralNetwork):
    """Fake network that always returns a fixed constant for a single scalar field."""

    def __init__(self, field_label: str, value: float, device: torch.device) -> None:
        super().__init__()
        self.output_field_labels = (field_label,)
        self._value = value
        self._device = device

    def evaluate(self, coords: CoordsProtocol) -> torch.Tensor:
        return torch.full(
            (coords.n_points, 1), self._value, device=self._device, requires_grad=True
        )


# Helpers
def make_coords(
    coord_system: CoordSystem, x: torch.Tensor, y: torch.Tensor
) -> CoordsProtocol:
    Coords = make_coords_dataclass(
        coord_labels={"x", "y"},
        coord_system=coord_system,
        device=x.device,
    )
    return Coords(x=x, y=y)


def make_data_container(coords: CoordsProtocol, value: float) -> DataContainer:
    dc = DataContainer(
        coords=coords,
        primary_field_label_to_rank={"u": 0},
        derived_field_label_to_rank={},
        const_labels=None,
        equations=None,
    )
    dc.set_field_from_tensor(
        "u",
        torch.full((coords.n_points,), value, device=coords.x.device),  # type: ignore[attr-defined]
    )
    return dc


def make_eval_context(
    coords: CoordsProtocol,
    input_value: float | None = None,
    output_value: float = 0.0,
) -> EvaluationContext:
    input_dc = (
        make_data_container(coords, input_value) if input_value is not None else None
    )
    output_dc = make_data_container(coords, output_value)
    dl = make_dataloader(coords=coords, batch_size=BATCH_SIZE, shuffle=False)
    return EvaluationContext(
        input_data=input_dc,
        output_data=output_dc,
        dataloader=dl,
        loss_function=LOSS_FUNCTION_REGISTRY[LossFunctionLabels.MSE],
    )


def make_container(
    coords: CoordsProtocol,
    domain: Domain,
    loss_term: DataLoss | PhysicsLoss,
    weight: float = 1.0,
    input_value: float | None = 0.0,
    output_value: float = 0.0,
) -> LossTermContainer:
    train_ctx = make_eval_context(
        coords, input_value=input_value, output_value=output_value
    )
    val_ctx = make_eval_context(
        coords, input_value=input_value, output_value=output_value
    )
    return LossTermContainer(
        label="test",
        train_eval_context=train_ctx,
        val_eval_context=val_ctx,
        domain=domain,
        loss_term=loss_term,
        loss_term_weight=weight,
    )


# Fixtures
@pytest.fixture  # type: ignore[misc]
def device() -> torch.device:
    return make_device()


@pytest.fixture  # type: ignore[misc]
def cartesian_2d(device: torch.device) -> CoordSystem:
    return make_cartesian_2d(device)


@pytest.fixture  # type: ignore[misc]
def domain(cartesian_2d: CoordSystem, device: torch.device) -> Domain:
    return make_domain_2d_cartesian(cartesian_2d, device)


@pytest.fixture  # type: ignore[misc]
def coords(cartesian_2d: CoordSystem, device: torch.device) -> CoordsProtocol:
    x, y = make_xy_2d(device)
    return make_coords(cartesian_2d, x, y)


@pytest.fixture  # type: ignore[misc]
def nn_model(device: torch.device) -> ConstantNN:
    return ConstantNN(field_label="u", value=3.0, device=device)


# Tests
class TestLossTermContainerConstruction:
    def test_stores_label(self, coords: CoordsProtocol, domain: Domain) -> None:
        ltc = make_container(coords, domain, DataLoss("u"))
        assert ltc.label == "test"

    def test_stores_loss_term(self, coords: CoordsProtocol, domain: Domain) -> None:
        term = DataLoss("u")
        ltc = make_container(coords, domain, term)
        assert ltc.loss_term is term

    def test_stores_weight(self, coords: CoordsProtocol, domain: Domain) -> None:
        ltc = make_container(coords, domain, DataLoss("u"), weight=2.5)
        assert ltc.loss_term_weight == 2.5

    def test_default_weight_is_one(
        self, coords: CoordsProtocol, domain: Domain
    ) -> None:
        ltc = make_container(coords, domain, DataLoss("u"))
        assert ltc.loss_term_weight == 1.0

    def test_out_of_domain_train_coords_raises(
        self, cartesian_2d: CoordSystem, domain: Domain, device: torch.device
    ) -> None:
        # coords well outside [0, 1] x [0, 1] domain
        x = torch.full((N_POINTS,), 5.0, device=device, requires_grad=True)
        y = torch.full((N_POINTS,), 5.0, device=device, requires_grad=True)
        bad_coords = make_coords(cartesian_2d, x, y)

        good_coords_x, good_coords_y = make_xy_2d(device)
        good_coords = make_coords(cartesian_2d, good_coords_x, good_coords_y)

        good_ctx = make_eval_context(good_coords)
        bad_ctx = make_eval_context(bad_coords)

        with pytest.raises(ValueError, match="domain"):
            LossTermContainer(
                label="test",
                train_eval_context=bad_ctx,
                val_eval_context=good_ctx,
                domain=domain,
                loss_term=DataLoss("u"),
            )

    def test_out_of_domain_val_coords_raises(
        self, cartesian_2d: CoordSystem, domain: Domain, device: torch.device
    ) -> None:
        x = torch.full((N_POINTS,), 5.0, device=device, requires_grad=True)
        y = torch.full((N_POINTS,), 5.0, device=device, requires_grad=True)
        bad_coords = make_coords(cartesian_2d, x, y)

        good_coords_x, good_coords_y = make_xy_2d(device)
        good_coords = make_coords(cartesian_2d, good_coords_x, good_coords_y)

        good_ctx = make_eval_context(good_coords)
        bad_ctx = make_eval_context(bad_coords)

        with pytest.raises(ValueError, match="domain"):
            LossTermContainer(
                label="test",
                train_eval_context=good_ctx,
                val_eval_context=bad_ctx,
                domain=domain,
                loss_term=DataLoss("u"),
            )


class TestApplyModel:
    def test_apply_model_writes_output_data(
        self,
        coords: CoordsProtocol,
        domain: Domain,
        nn_model: ConstantNN,
        device: torch.device,
    ) -> None:
        ltc = make_container(coords, domain, DataLoss("u"), output_value=0.0)
        ltc._apply_model(nn_model, coords, ltc.train_eval_context)

        result = ltc.train_eval_context.output_data.get_field_component_tensor(
            "u", coords
        )
        expected = torch.full((N_POINTS, 1), 3.0, device=device)
        assert torch.allclose(result, expected)

    def test_apply_model_uses_provided_coords(
        self,
        coords: CoordsProtocol,
        domain: Domain,
        device: torch.device,
    ) -> None:
        """Model is called with the coords passed in, not full-set coords."""
        model = ConstantNN(field_label="u", value=7.0, device=device)
        ltc = make_container(coords, domain, DataLoss("u"))
        ltc._apply_model(model, coords, ltc.train_eval_context)

        out = ltc.train_eval_context.output_data.get_field_component_tensor("u", coords)
        assert torch.allclose(out, torch.full((N_POINTS, 1), 7.0, device=device))


class TestComputeLoss:
    def test_compute_loss_returns_scalar_tensor(
        self,
        coords: CoordsProtocol,
        domain: Domain,
        nn_model: ConstantNN,
    ) -> None:
        ltc = make_container(coords, domain, DataLoss("u"), input_value=3.0)
        loss = ltc.compute_loss(nn_model, ltc.train_eval_context, coords, detach=True)
        assert loss.shape == torch.Size([])

    def test_compute_loss_detach_true_has_no_grad(
        self,
        coords: CoordsProtocol,
        domain: Domain,
        nn_model: ConstantNN,
    ) -> None:
        ltc = make_container(coords, domain, DataLoss("u"), input_value=1.0)
        loss = ltc.compute_loss(nn_model, ltc.train_eval_context, coords, detach=True)
        assert not loss.requires_grad

    def test_compute_loss_detach_false_has_grad(
        self,
        coords: CoordsProtocol,
        domain: Domain,
        nn_model: ConstantNN,
    ) -> None:
        ltc = make_container(coords, domain, DataLoss("u"), input_value=1.0)
        loss = ltc.compute_loss(nn_model, ltc.train_eval_context, coords, detach=False)
        assert loss.requires_grad

    def test_compute_loss_value_data_loss(
        self,
        coords: CoordsProtocol,
        domain: Domain,
        device: torch.device,
    ) -> None:
        # model outputs 3.0, target is 1.0 => residual = 2.0, MSE = 4.0
        model = ConstantNN(field_label="u", value=3.0, device=device)
        ltc = make_container(coords, domain, DataLoss("u"), input_value=1.0)
        loss = ltc.compute_loss(model, ltc.train_eval_context, coords, detach=True)
        assert torch.allclose(loss, torch.tensor(4.0))

    def test_compute_loss_value_physics_loss(
        self,
        coords: CoordsProtocol,
        domain: Domain,
        device: torch.device,
    ) -> None:
        # model outputs 2.0 => residual = 2.0, MSE = 4.0
        model = ConstantNN(field_label="u", value=2.0, device=device)
        ltc = make_container(
            coords, domain, PhysicsLoss("u"), input_value=None, output_value=0.0
        )
        loss = ltc.compute_loss(model, ltc.train_eval_context, coords, detach=True)
        assert torch.allclose(loss, torch.tensor(4.0))


class TestComputeTrainLoss:
    def test_unweighted_train_loss_uses_train_context(
        self,
        coords: CoordsProtocol,
        domain: Domain,
        device: torch.device,
    ) -> None:
        model = ConstantNN(field_label="u", value=3.0, device=device)
        ltc = make_container(coords, domain, DataLoss("u"), input_value=1.0)
        loss = ltc.compute_unweighted_train_loss(model)
        assert torch.allclose(loss, torch.tensor(4.0))

    def test_unweighted_train_loss_is_detached_by_default(
        self,
        coords: CoordsProtocol,
        domain: Domain,
        nn_model: ConstantNN,
    ) -> None:
        ltc = make_container(coords, domain, DataLoss("u"), input_value=1.0)
        loss = ltc.compute_unweighted_train_loss(nn_model)
        assert not loss.requires_grad

    def test_weighted_train_loss_scales_by_weight(
        self,
        coords: CoordsProtocol,
        domain: Domain,
        device: torch.device,
    ) -> None:
        model = ConstantNN(field_label="u", value=3.0, device=device)
        ltc = make_container(coords, domain, DataLoss("u"), weight=0.5, input_value=1.0)
        weighted = ltc.compute_weighted_train_loss(model)
        unweighted = ltc.compute_unweighted_train_loss(model)
        assert torch.allclose(weighted, 0.5 * unweighted)


class TestComputeValLoss:
    def test_unweighted_val_loss_uses_val_context(
        self,
        coords: CoordsProtocol,
        domain: Domain,
        device: torch.device,
    ) -> None:
        model = ConstantNN(field_label="u", value=3.0, device=device)
        ltc = make_container(coords, domain, DataLoss("u"), input_value=1.0)
        loss = ltc.compute_unweighted_val_loss(model)
        assert torch.allclose(loss, torch.tensor(4.0))

    def test_unweighted_val_loss_is_detached_by_default(
        self,
        coords: CoordsProtocol,
        domain: Domain,
        nn_model: ConstantNN,
    ) -> None:
        ltc = make_container(coords, domain, DataLoss("u"), input_value=1.0)
        loss = ltc.compute_unweighted_val_loss(nn_model)
        assert not loss.requires_grad

    def test_weighted_val_loss_scales_by_weight(
        self,
        coords: CoordsProtocol,
        domain: Domain,
        device: torch.device,
    ) -> None:
        model = ConstantNN(field_label="u", value=3.0, device=device)
        ltc = make_container(coords, domain, DataLoss("u"), weight=2.0, input_value=1.0)
        weighted = ltc.compute_weighted_val_loss(model)
        unweighted = ltc.compute_unweighted_val_loss(model)
        assert torch.allclose(weighted, 2.0 * unweighted)


class TestIterLoss:
    def test_iter_train_loss_yields_one_loss_per_batch(
        self,
        coords: CoordsProtocol,
        domain: Domain,
        nn_model: ConstantNN,
    ) -> None:
        ltc = make_container(coords, domain, DataLoss("u"), input_value=1.0)
        losses = list(ltc.iter_train_loss(nn_model))
        n_batches = -(-N_POINTS // BATCH_SIZE)  # ceiling division
        assert len(losses) == n_batches

    def test_iter_train_loss_default_not_detached(
        self,
        coords: CoordsProtocol,
        domain: Domain,
        nn_model: ConstantNN,
    ) -> None:
        ltc = make_container(coords, domain, DataLoss("u"), input_value=1.0)
        for loss in ltc.iter_train_loss(nn_model):
            assert loss.requires_grad

    def test_iter_weighted_train_loss_scales_each_batch(
        self,
        coords: CoordsProtocol,
        domain: Domain,
        device: torch.device,
    ) -> None:
        weight = 3.0
        model = ConstantNN(field_label="u", value=2.0, device=device)
        ltc = make_container(
            coords, domain, DataLoss("u"), weight=weight, input_value=0.0
        )

        unweighted = list(ltc.iter_train_loss(model, detach=True))
        weighted = list(ltc.iter_weighted_train_loss(model, detach=True))

        assert len(weighted) == len(unweighted)
        for w, u in zip(weighted, unweighted, strict=True):
            assert torch.allclose(w, weight * u)

    def test_iter_val_loss_yields_one_loss_per_batch(
        self,
        coords: CoordsProtocol,
        domain: Domain,
        nn_model: ConstantNN,
    ) -> None:
        ltc = make_container(coords, domain, DataLoss("u"), input_value=1.0)
        losses = list(ltc.iter_val_loss(nn_model))
        n_batches = -(-N_POINTS // BATCH_SIZE)
        assert len(losses) == n_batches

    def test_iter_val_loss_default_is_detached(
        self,
        coords: CoordsProtocol,
        domain: Domain,
        nn_model: ConstantNN,
    ) -> None:
        ltc = make_container(coords, domain, DataLoss("u"), input_value=1.0)
        for loss in ltc.iter_val_loss(nn_model):
            assert not loss.requires_grad

    def test_iter_weighted_val_loss_scales_each_batch(
        self,
        coords: CoordsProtocol,
        domain: Domain,
        device: torch.device,
    ) -> None:
        weight = 0.25
        model = ConstantNN(field_label="u", value=2.0, device=device)
        ltc = make_container(
            coords, domain, DataLoss("u"), weight=weight, input_value=0.0
        )

        unweighted = list(ltc.iter_val_loss(model))
        weighted = list(ltc.iter_weighted_val_loss(model))

        assert len(weighted) == len(unweighted)
        for w, u in zip(weighted, unweighted, strict=True):
            assert torch.allclose(w, torch.tensor(weight) * u)
