"""Unit tests for the loss module."""

from __future__ import annotations

import pytest
import torch

from pinninit.data_structure.batching import Batches
from pinninit.data_structure.coords import Coords
from pinninit.data_structure.data_container import DataContainer
from pinninit.data_structure.equation import Equation
from pinninit.data_structure.field import make_field_dataclass
from pinninit.differential_equations.operators import Operators
from pinninit.geometry.coordinate_system import CoordSystem
from pinninit.geometry.domain import Domain
from pinninit.loss.evaluation_context import EvaluationContext
from pinninit.loss.loss_function import (
    mse_loss,
    weighted_mse_loss,
    weighted_point_avg_mse_loss,
)
from pinninit.loss.loss_term import DataLoss, PhysicsLoss
from pinninit.loss.loss_term_container import LossTermContainer

from ..helpers import N_POINTS


class Model:
    """Stands in for the neural network: outputs the field u = x + y."""

    output_field_labels = ("u",)

    def evaluate(self, coords: Coords) -> torch.Tensor:
        return (coords.get_coord("x") + coords.get_coord("y")).unsqueeze(1)


class TestLossFunctions:
    def test_loss_functions(
        self, eval_context: EvaluationContext, coords_2d: Coords, device: torch.device
    ) -> None:
        residuals = torch.arange(N_POINTS, dtype=torch.float, device=device)
        weights = torch.arange(1, N_POINTS + 1, dtype=torch.float, device=device)
        eval_context.update_point_weights(weights)
        r = residuals.unsqueeze(1)

        mse = mse_loss(r, eval_context, coords_2d)
        weighted = weighted_mse_loss(r, eval_context, coords_2d)
        point_avg = weighted_point_avg_mse_loss(r, eval_context, coords_2d)

        assert torch.allclose(mse, (residuals**2).mean())
        assert torch.allclose(weighted, (weights * residuals**2).sum() / weights.sum())
        assert torch.allclose(point_avg, (weights * residuals**2).mean())


class TestPointWeights:
    def test_point_weights_start_at_one(self, eval_context: EvaluationContext) -> None:
        assert torch.equal(eval_context.get_point_weights(), torch.ones(N_POINTS))

    def test_update_on_a_batch(
        self, eval_context: EvaluationContext, coords_2d: Coords, device: torch.device
    ) -> None:
        batch = coords_2d.get_batch(torch.tensor([5, 2], device=device))

        eval_context.update_point_weights(
            torch.tensor([3.0, 4.0], device=device), batch
        )

        assert eval_context.get_point_weights(batch).tolist() == [3.0, 4.0]
        expected = [1.0, 1.0, 4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0]
        assert eval_context.get_point_weights().tolist() == expected

    def test_modifier_multiplies_the_raw_weights(
        self, data: DataContainer, coords_2d: Coords, device: torch.device
    ) -> None:
        ScalarField = make_field_dataclass(0, type(coords_2d))
        modifier = torch.arange(N_POINTS, dtype=torch.float, device=device)
        eval_context = EvaluationContext(
            output_data=data,
            loss_function=weighted_mse_loss,
            batches=Batches(coords_2d, n_batches=1),
            point_weight_modifier=ScalarField(
                label="modifier", coords=coords_2d, value=modifier
            ),
        )

        eval_context.update_point_weights(torch.full((N_POINTS,), 2.0, device=device))

        assert torch.equal(eval_context.get_point_weights(), 2.0 * modifier)
        assert torch.equal(
            eval_context.get_raw_point_weights(),
            torch.full((N_POINTS,), 2.0, device=device),
        )

    def test_setting_all_point_weights_copies_them(
        self, eval_context: EvaluationContext, coords_2d: Coords, device: torch.device
    ) -> None:
        # e.g. the trainer's best-model weights, which must not change when
        # training goes on
        best_weights = torch.full((N_POINTS,), 2.0, device=device)
        eval_context.update_point_weights(best_weights)
        batch = coords_2d.get_batch(torch.tensor([5, 2], device=device))

        eval_context.update_point_weights(
            torch.tensor([3.0, 4.0], device=device), batch
        )

        assert torch.equal(best_weights, torch.full((N_POINTS,), 2.0, device=device))


class TestLossTerms:
    def test_data_loss_residual_is_prediction_minus_data(
        self, data: DataContainer, coords_2d: Coords
    ) -> None:
        x, y = coords_2d.get_coord("x"), coords_2d.get_coord("y")
        input_data = DataContainer(coords_2d, primary_field_label_to_rank={"u": 0})
        input_data.set_field_from_tensor("u", x)
        data.set_field_from_tensor("u", x + y)
        eval_context = EvaluationContext(
            output_data=data,
            loss_function=mse_loss,
            batches=Batches(coords_2d, n_batches=1),
            input_data=input_data,
        )

        residuals = DataLoss("u").compute_residual(eval_context, coords_2d)

        assert torch.allclose(residuals, y.unsqueeze(1))

    def test_physics_loss_residual_is_the_field(
        self, eval_context: EvaluationContext, coords_2d: Coords
    ) -> None:
        x = coords_2d.get_coord("x")
        eval_context.output_data.set_field_from_tensor("w", x, allow_derived=True)

        residuals = PhysicsLoss("w").compute_residual(eval_context, coords_2d)

        assert torch.equal(residuals, x.unsqueeze(1))


class TestLossTermContainer:
    def test_full_loss_is_exact_over_batches(
        self, cartesian_2d: CoordSystem, coords_2d: Coords, device: torch.device
    ) -> None:
        # The model gives u = x + y and the residual is the derived field
        # w = d(u)/dx * u = x + y. The weighted MSE over all points is not the
        # average of the batches' weighted MSEs
        def make_w(u: torch.Tensor, operators: Operators) -> torch.Tensor:
            return operators.d_dcoord(u, "x") * u

        output_data = DataContainer(
            coords_2d,
            primary_field_label_to_rank={"u": 0},
            derived_field_label_to_rank={"w": 0},
            equations=[Equation("w", make_w, {"w": 0})],
        )
        eval_context = EvaluationContext(
            output_data=output_data,
            loss_function=weighted_mse_loss,
            batches=Batches(coords_2d, n_batches=3),
        )
        weights = torch.arange(1, N_POINTS + 1, dtype=torch.float, device=device)
        eval_context.update_point_weights(weights)
        container = LossTermContainer(
            label="w",
            train_eval_context=eval_context,
            val_eval_context=eval_context,
            domain=Domain(cartesian_2d),
            loss_term=PhysicsLoss("w"),
        )

        loss = container.compute_full_train_loss(Model())  # type: ignore[arg-type]

        w = (coords_2d.get_coord("x") + coords_2d.get_coord("y")).detach()
        assert torch.allclose(loss, (weights * w**2).sum() / weights.sum())

    def test_data_loss_on_a_derived_field(
        self, cartesian_2d: CoordSystem, coords_2d: Coords
    ) -> None:
        # The model gives u = x + y, the derived field is w = 2 * u and the
        # data are w = x, so the residual is x + 2 * y
        def make_w(u: torch.Tensor, operators: Operators) -> torch.Tensor:
            return 2 * u

        x, y = coords_2d.get_coord("x"), coords_2d.get_coord("y")
        input_data = DataContainer(coords_2d, primary_field_label_to_rank={"w": 0})
        input_data.set_field_from_tensor("w", x)
        output_data = DataContainer(
            coords_2d,
            primary_field_label_to_rank={"u": 0},
            derived_field_label_to_rank={"w": 0},
            equations=[Equation("w", make_w, {"w": 0})],
        )
        eval_context = EvaluationContext(
            output_data=output_data,
            loss_function=mse_loss,
            batches=Batches(coords_2d, n_batches=1),
            input_data=input_data,
        )
        container = LossTermContainer(
            label="w",
            train_eval_context=eval_context,
            val_eval_context=eval_context,
            domain=Domain(cartesian_2d),
            loss_term=DataLoss("w"),
        )

        loss = container.compute_full_train_loss(Model())  # type: ignore[arg-type]

        assert torch.allclose(loss, ((x + 2 * y) ** 2).mean())

    def test_loss_term_on_an_unknown_field_raises(
        self, eval_context: EvaluationContext, cartesian_2d: CoordSystem
    ) -> None:
        with pytest.raises(ValueError, match="not an output or derived field"):
            LossTermContainer(
                label="missing",
                train_eval_context=eval_context,
                val_eval_context=eval_context,
                domain=Domain(cartesian_2d),
                loss_term=PhysicsLoss("missing"),
            )

    def test_data_loss_needs_input_data(
        self, eval_context: EvaluationContext, cartesian_2d: CoordSystem
    ) -> None:
        with pytest.raises(ValueError, match="needs input data"):
            LossTermContainer(
                label="u",
                train_eval_context=eval_context,
                val_eval_context=eval_context,
                domain=Domain(cartesian_2d),
                loss_term=DataLoss("u"),
            )

    def test_points_outside_the_domain_raise(
        self, eval_context: EvaluationContext, cartesian_2d: CoordSystem
    ) -> None:
        with pytest.raises(ValueError, match="outside of domain"):
            LossTermContainer(
                label="u",
                train_eval_context=eval_context,
                val_eval_context=eval_context,
                domain=Domain(cartesian_2d, geometric_bounds={"x": (0.0, 0.5)}),
                loss_term=PhysicsLoss("u"),
            )

    def test_point_weights_need_a_weighted_loss_function(
        self, data: DataContainer, cartesian_2d: CoordSystem, coords_2d: Coords
    ) -> None:
        eval_context = EvaluationContext(
            output_data=data,
            loss_function=mse_loss,
            batches=Batches(coords_2d, n_batches=1),
        )
        with pytest.raises(ValueError, match="ignores them"):
            LossTermContainer(
                label="u",
                train_eval_context=eval_context,
                val_eval_context=eval_context,
                domain=Domain(cartesian_2d),
                loss_term=PhysicsLoss("u"),
                adaptive_point_weights=True,
            )
