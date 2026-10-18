"""Unit tests for the rules the trainer and its modules are built on."""

from __future__ import annotations

import os
import signal
from typing import TYPE_CHECKING, Any

import pytest
import torch
import torch.nn as nn

from pinninit.data_structure.batching import Batches
from pinninit.data_structure.coords import Coords
from pinninit.data_structure.data_container import DataContainer
from pinninit.geometry.coordinate_system import CoordSystem
from pinninit.geometry.domain import Domain
from pinninit.loss.evaluation_context import EvaluationContext
from pinninit.loss.loss_function import weighted_point_avg_mse_loss
from pinninit.loss.loss_term import DataLoss
from pinninit.loss.loss_term_container import LossTermContainer, WeightMethodLabels
from pinninit.optimizer.pytorch_optimizers import LBFGS, Adam
from pinninit.trainer.trainer import Trainer
from pinninit.trainer_modules.adaptive_weights_W23 import AdaptiveWeightsW23
from pinninit.trainer_modules.balanced_residual_decay_rate_C25 import (
    BalancedResidualDecayRateC25,
)
from pinninit.trainer_modules.learning_rate_schedule import LearningRateSchedule
from pinninit.trainer_modules.loss_term_staggering import LossTermStaggering
from pinninit.trainer_modules.residual_based_attention_A24 import (
    ResidualBasedAttentionA24,
)
from pinninit.trainer_modules.trainer_module import TrainerModule

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pinninit.optimizer.optimizer import Optimizer
    from pinninit.trainer.trainer_logger import TrainerLogger


class Model(nn.Module):
    """A small network for the field u(x, y)."""

    output_field_labels = ("u",)

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))

    def evaluate(self, coords: Coords) -> torch.Tensor:
        inputs = torch.stack([coords.get_coord("x"), coords.get_coord("y")], dim=1)
        output: torch.Tensor = self.net(inputs)
        return output


def make_trainer(
    coords: Coords,
    coord_system: CoordSystem,
    modules: list[TrainerModule],
    optimizer_class: Callable[..., Optimizer] = Adam,
    staggering_b: int = 0,
    **trainer_kwargs: Any,
) -> Trainer:
    """A trainer fitting u to 2 + sin(x) (loss term 'a') and 0 (loss term 'b',
    with a much smaller loss), with adaptive point weights and W23 weights on
    both."""
    torch.manual_seed(0)
    x, y = coords.get_coord("x").detach(), coords.get_coord("y").detach()

    def eval_context(target: torch.Tensor, n_batches: int) -> EvaluationContext:
        data = DataContainer(coords, primary_field_label_to_rank={"u": 0})
        data.set_field_from_tensor("u", target)
        return EvaluationContext(
            output_data=DataContainer(coords, primary_field_label_to_rank={"u": 0}),
            loss_function=weighted_point_avg_mse_loss,
            batches=Batches(coords, n_batches),
            input_data=data,
        )

    containers = {
        label: LossTermContainer(
            label=label,
            train_eval_context=eval_context(target, n_batches=2),
            val_eval_context=eval_context(target, n_batches=1),
            domain=Domain(coord_system),
            loss_term=DataLoss("u"),
            weight_method_label=WeightMethodLabels.ADAPTIVE_W23,
            weight_group_label=label,
            staggering=staggering,
            adaptive_point_weights=True,
        )
        for label, target, staggering in [
            ("a", 2 + torch.sin(x), 0),
            ("b", torch.zeros_like(y), staggering_b),
        ]
    }
    model = Model()
    optimizer = optimizer_class(model.parameters(), lr=0.01)
    return Trainer(
        model,  # type: ignore[arg-type]
        optimizer,
        containers,
        modules,
        print_freq=100,
        **trainer_kwargs,
    )


def weights(trainer: Trainer) -> dict[str, tuple[float, torch.Tensor]]:
    """The loss term weight and raw point weights of each loss term."""
    return {
        label: (
            container.loss_term_weight,
            container.train_eval_context.get_raw_point_weights().clone(),
        )
        for label, container in trainer.all_loss_containers.items()
    }


def assert_same_weights(
    first: dict[str, tuple[float, torch.Tensor]],
    second: dict[str, tuple[float, torch.Tensor]],
) -> None:
    assert first.keys() == second.keys()
    for label in first:
        assert first[label][0] == second[label][0]
        assert torch.equal(first[label][1], second[label][1])


class WeightRecorder(TrainerModule):
    """Records the weights it sees before each optimizer step."""

    def __init__(self) -> None:
        self.seen: list[dict[str, tuple[float, torch.Tensor]]] = []

    def on_pre_optimizer_step(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        self.seen.append(weights(trainer))


class SigtermSender(TrainerModule):
    """Sends this process SIGTERM at the end of iteration iter."""

    def __init__(self, iter: int | None) -> None:
        self.iter = iter

    def on_step_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        if trainer.iter == self.iter:
            os.kill(os.getpid(), signal.SIGTERM)


class TestTrainerModules:
    def test_module_order_does_not_change_training(
        self, coords_2d: Coords, cartesian_2d: CoordSystem
    ) -> None:
        # Staggering and the learning rate schedule both see a plateau at the
        # end of epoch 2 (the val loss hasn't halved), so they both act at the
        # same epoch end. Each decides from the epoch's val losses: once
        # staggering adds 'b', the geometric mean val loss is much smaller, so
        # a module reading it after staggering wouldn't see a plateau.
        def train(reverse: bool) -> Trainer:
            modules: list[TrainerModule] = [
                ResidualBasedAttentionA24(),
                AdaptiveWeightsW23(loss_function="weighted_point_avg_mse"),
                LossTermStaggering(patience=1, frac_tol=0.5),
                LearningRateSchedule(
                    decay_factor=0.5, n_events=2, patience=1, frac_tol=0.5
                ),
            ]
            if reverse:
                modules.reverse()
            trainer = make_trainer(
                coords_2d,
                cartesian_2d,
                modules,
                staggering_b=1,
                max_epoch=5,
                log_point_weight_info=True,
            )
            trainer.train()
            return trainer

        first, second = train(reverse=False), train(reverse=True)

        events = {(e["epoch"], e["event"]) for e in first.logger.events}
        assert {(2, "activated_loss_term_b"), (2, "learning_rate_decay")} <= events
        assert events == {(e["epoch"], e["event"]) for e in second.logger.events}
        for p, q in zip(
            first.model.parameters(), second.model.parameters(), strict=True
        ):
            assert torch.equal(p, q)
        assert_same_weights(weights(first), weights(second))
        assert (
            first.logger.epoch_point_weight_info_log
            == second.logger.epoch_point_weight_info_log
        )
        # The epoch is logged before the modules act on it: 'b' is activated
        # at the end of epoch 2, so it isn't in that epoch's log
        assert list(first.logger.epoch_point_weight_info_log[1]) == ["a"]

    def test_modules_see_the_weights_the_loss_was_computed_with(
        self, coords_2d: Coords, cartesian_2d: CoordSystem
    ) -> None:
        # The weights the modules compute in a step are only applied once all
        # modules have run, so a module after them sees the same weights as
        # one before them
        before, after = WeightRecorder(), WeightRecorder()
        modules = [
            before,
            ResidualBasedAttentionA24(),
            AdaptiveWeightsW23(),
            after,
        ]
        trainer = make_trainer(coords_2d, cartesian_2d, modules, max_epoch=2)

        trainer.train()

        assert len(before.seen) == len(after.seen) == trainer.iter
        for seen_before, seen_after in zip(before.seen, after.seen, strict=True):
            assert_same_weights(seen_before, seen_after)
        # The weights did change from step to step
        assert not torch.equal(before.seen[0]["a"][1], before.seen[-1]["a"][1])
        assert before.seen[0]["a"][0] != before.seen[-1]["a"][0]

    def test_updating_the_same_weights_twice_in_a_step_raises(
        self, coords_2d: Coords, cartesian_2d: CoordSystem
    ) -> None:
        modules: list[TrainerModule] = [
            ResidualBasedAttentionA24(),
            BalancedResidualDecayRateC25(),
        ]
        trainer = make_trainer(coords_2d, cartesian_2d, modules, max_epoch=1)

        with pytest.raises(ValueError, match="updated twice in one step"):
            trainer.train()

    def test_adaptive_weights_stay_fixed_with_a_fixed_loss_optimizer(
        self, coords_2d: Coords, cartesian_2d: CoordSystem
    ) -> None:
        # L-BFGS needs the loss to stay the same from step to step
        modules: list[TrainerModule] = [
            ResidualBasedAttentionA24(),
            AdaptiveWeightsW23(),
        ]
        trainer = make_trainer(
            coords_2d, cartesian_2d, modules, optimizer_class=LBFGS, max_epoch=3
        )
        initial = [p.detach().clone() for p in trainer.model.parameters()]

        trainer.train()

        assert any(
            not torch.equal(p, q)
            for p, q in zip(initial, trainer.model.parameters(), strict=True)
        )
        for weight, point_weights in weights(trainer).values():
            assert weight == 1.0
            assert torch.equal(point_weights, torch.ones_like(point_weights))

    def test_resuming_training_gives_the_same_training(
        self, coords_2d: Coords, cartesian_2d: CoordSystem, tmp_path: Path
    ) -> None:
        # Training continued from a checkpoint, or from the end of the last
        # call to train, is the same as training without stopping, although
        # the best model is loaded when training stops
        def make(sigterm_iter: int | None = None, **kwargs: Any) -> Trainer:
            modules: list[TrainerModule] = [
                ResidualBasedAttentionA24(),
                AdaptiveWeightsW23(),
                LossTermStaggering(patience=1, frac_tol=0.5),
                LearningRateSchedule(
                    decay_factor=0.5, n_events=2, patience=1, frac_tol=0.5
                ),
                SigtermSender(sigterm_iter),
            ]
            return make_trainer(
                coords_2d, cartesian_2d, modules, staggering_b=1, **kwargs
            )

        uninterrupted = make(max_epoch=8)
        uninterrupted.train()

        # SIGTERM in the middle of epoch 5 (2 steps per epoch) stops training
        # at its end, with a checkpoint. The best model is then from epoch 3.
        checkpoint_path = tmp_path / "checkpoint.pth"
        stopped = make(
            sigterm_iter=9,
            max_epoch=8,
            checkpoint_path=checkpoint_path,
            stop_on_sigterm=True,
        )
        stopped.train()
        assert stopped.stopped_by_sigterm
        assert stopped.epoch == 5
        assert stopped.best_model_epoch == 3
        resumed = make(max_epoch=8)
        resumed.load(checkpoint_path)
        resumed.train()

        continued = make(max_epoch=5)
        continued.train()
        continued.max_epoch = 8
        continued.train()

        # The same best model, and the same state at the end of training
        expected = uninterrupted.state_dict()
        for trainer in (resumed, continued):
            assert not trainer.stopped_by_sigterm
            assert trainer.best_model_epoch == uninterrupted.best_model_epoch
            assert_same_weights(weights(uninterrupted), weights(trainer))
            state = trainer.state_dict()
            assert state["iter"] == expected["iter"]
            assert state["loss_term_weights"] == expected["loss_term_weights"]
            for key in ("model_state", "point_weights"):
                assert state[key].keys() == expected[key].keys()
                for name, value in state[key].items():
                    assert torch.equal(value, expected[key][name])
