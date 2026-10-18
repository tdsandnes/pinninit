from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .point_in_step import PointInStep, PointInStepRecorder

if TYPE_CHECKING:
    from typing import Any

    from .trainer import Trainer


class TrainerLogger:
    def __init__(
        self,
        trainer: Trainer,
        print_freq: int = 1,
        log_point_weight_info: bool = False,
        point_weight_snapshot_freq: int = 0,
    ):
        self.trainer = trainer
        self.print_freq = print_freq  # print every print_freq epochs
        # Log stats of the adaptive point weights at every epoch end, and the
        # point weights every point_weight_snapshot_freq epochs (if > 0)
        self.log_point_weight_info = log_point_weight_info
        self.point_weight_snapshot_freq = point_weight_snapshot_freq
        self.events: list[dict[str, Any]] = []

        # The training losses of the step's first loss evaluation (an optimizer
        # such as L-BFGS can evaluate the loss several times in one step)
        self.unweighted_train_losses = PointInStepRecorder(PointInStep.FIRST)

        # Accumulator for averaging train losses across steps within an epoch
        self._epoch_train_loss_accum: list[dict[str, float]] = []

        # Per-epoch logs (full-set val loss, mean train loss, weights at epoch end)
        self.epoch_train_losses_log: list[dict[str, float]] = []
        self.epoch_val_losses_log: list[dict[str, float]] = []
        self.epoch_loss_term_weights_log: list[dict[str, float]] = []
        self.epoch_learnt_consts_log: list[dict[str, float]] = []
        # Wall clock time spent training at the end of each epoch, in seconds
        self.epoch_train_time_log: list[float] = []
        self.epoch_point_weight_info_log: list[dict[str, dict[str, float]]] = []
        self.point_weight_snapshots: list[dict[str, Any]] = []

    # The logs, as saved in checkpoints
    log_names = (
        "events",
        "epoch_train_losses_log",
        "epoch_val_losses_log",
        "epoch_loss_term_weights_log",
        "epoch_learnt_consts_log",
        "epoch_train_time_log",
        "epoch_point_weight_info_log",
        "point_weight_snapshots",
    )

    def state_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.log_names}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name in self.log_names:
            setattr(self, name, state[name])

    def on_step_start(self) -> None:
        self.unweighted_train_losses.reset()

    def accumulate_train_losses(self) -> None:
        """Stash this step's train losses for the epoch mean."""
        if self.unweighted_train_losses.has_value:
            self._epoch_train_loss_accum.append(self.unweighted_train_losses.value)
        else:
            raise RuntimeError(
                "Unweighted training losses were not recorded during step."
            )

    def on_epoch_end(self) -> None:
        """Called at the end of each epoch, before the trainer modules act on
        it. Records the epoch's losses and weights in the epoch logs, and
        prints them if the epoch is a print checkpoint."""
        trainer = self.trainer
        full_val_losses = trainer.epoch_val_losses

        # Mean of the per-step batch losses over the epoch (the full-set loss
        # only if all batches are the same size)
        epoch_train_losses = (
            {
                label: sum(step[label] for step in self._epoch_train_loss_accum)
                / len(self._epoch_train_loss_accum)
                for label in self._epoch_train_loss_accum[0]
            }
            if self._epoch_train_loss_accum
            else {}
        )

        # Weights at the end of the epoch
        epoch_weights = {
            label: container.loss_term_weight
            for label, container in trainer.loss_containers.items()
        }

        # Record into epoch logs
        self.epoch_train_losses_log.append(epoch_train_losses)
        self.epoch_val_losses_log.append(full_val_losses)
        self.epoch_loss_term_weights_log.append(epoch_weights)
        self.epoch_learnt_consts_log.append(self._learnt_consts())
        self.epoch_train_time_log.append(trainer.elapsed_train_time)
        self._log_point_weights()

        # Reset accumulator for next epoch
        self._epoch_train_loss_accum = []

        # Print if on a print checkpoint
        if trainer.epoch % self.print_freq == 0:
            self._print_summary(
                f"Epoch: {trainer.epoch}  (iter {trainer.iter}, "
                f"{_format_time(trainer.elapsed_train_time)} elapsed)",
                "Training loss (epoch mean of batch losses)",
                epoch_train_losses,
                full_val_losses,
            )

    def _log_point_weights(self) -> None:
        """Log the adaptive point weights, if asked to."""
        trainer = self.trainer
        point_weights = {
            label: container.train_eval_context.get_raw_point_weights()
            for label, container in trainer.loss_containers.items()
            if container.adaptive_point_weights
        }

        if self.log_point_weight_info:
            self.epoch_point_weight_info_log.append(
                {
                    label: {
                        stat: float(getattr(weights, stat)())
                        for stat in ("mean", "median", "min", "max")
                    }
                    for label, weights in point_weights.items()
                }
            )

        if (
            self.point_weight_snapshot_freq > 0
            and trainer.epoch % self.point_weight_snapshot_freq == 0
        ):
            self.point_weight_snapshots.append(
                {
                    "epoch": trainer.epoch,
                    "iter": trainer.iter,
                    "weights": {
                        label: weights.clone().cpu()
                        for label, weights in point_weights.items()
                    },
                }
            )

    def _learnt_consts(self) -> dict[str, float]:
        return {
            name.removeprefix("learnt_const_"): float(param.detach())
            for name, param in self.trainer.model.named_parameters()
            if name.startswith("learnt_const_")
        }

    def _learnt_consts_str(self) -> str:
        """Formatted line of learnt constants, empty if there are none."""
        consts = self._learnt_consts()
        if not consts:
            return ""
        return f"Learnt constants:             {consts} \n"

    def _point_weights_str(self) -> str:
        """Formatted summary of per-point weights, one line per container whose
        weights are adapted or reshaped by a modifier, empty if none are."""
        lines = []
        for label, container in self.trainer.loss_containers.items():
            eval_context = container.train_eval_context
            has_modifier = eval_context.point_weight_modifier is not None
            if container.adaptive_point_weights or has_modifier:
                point_weights = eval_context.get_point_weights()
                lines.append(
                    f"Point weights ({label}): "
                    f"mean {point_weights.mean().item():.4g}, "
                    f"median {point_weights.median().item():.4g}, "
                    f"std {point_weights.std().item():.4g}, "
                    f"min {point_weights.min().item():.4g}, "
                    f"max {point_weights.max().item():.4g} \n"
                )
        return "".join(lines)

    def _skip_alphas_str(self) -> str:
        """Formatted line of skip-connection scalars in block order,
        empty if the model has none."""
        model = self.trainer.model
        if not getattr(model, "skip_connection", False):
            return ""
        assert isinstance(model.skip_alphas, nn.ParameterList)
        alphas = [round(float(a.detach()), 4) for a in model.skip_alphas]
        return f"Skip connection alphas:       {alphas} \n"

    def _module_summaries_str(self) -> str:
        """The trainer modules' extra summary lines."""
        return "".join(module.summary() for module in self.trainer.trainer_modules)

    def _print_summary(
        self,
        title: str,
        train_title: str,
        train_losses: dict[str, float],
        val_losses: dict[str, float],
    ) -> None:
        """Print the losses of each loss term (before its loss term weight) and
        the weights."""
        trainer = self.trainer
        weights = {
            label: container.loss_term_weight
            for label, container in trainer.loss_containers.items()
        }

        def weighted_sum_and_average(losses: dict[str, float]) -> tuple[float, float]:
            # Over the terms in losses: a term activated at the end of an epoch
            # has no training loss for that epoch
            weighted_sum = sum(weights[label] * loss for label, loss in losses.items())
            return weighted_sum, weighted_sum / sum(weights[label] for label in losses)

        train_sum, train_average = weighted_sum_and_average(train_losses)
        val_sum, val_average = weighted_sum_and_average(val_losses)
        print(
            "-------------------------------------------------------------------\n"
            f"{title}: \n"
            "-------------------------------------------------------------------\n"
            f"{train_title}: \n"
            f"- per loss term:              {train_losses} \n"
            f"- weighted sum:               {train_sum} \n"
            f"- weighted average:           {train_average} \n"
            f"Validation loss (full set): \n"
            f"- per loss term:              {val_losses} \n"
            f"- weighted sum:               {val_sum} \n"
            f"- weighted average:           {val_average} \n"
            f"- geometric mean:             {trainer.full_geometric_mean_val_loss} \n"
            f"Loss term weights:            {weights} \n"
            f"{self._learnt_consts_str()}"
            f"{self._point_weights_str()}"
            f"{self._skip_alphas_str()}"
            f"{self._module_summaries_str()}"
        )

    def print_train_time(self) -> None:
        print(f"Training took {_format_time(self.trainer.train_time)}")

    def record_train_losses(self, losses: dict[str, torch.Tensor]) -> None:
        self.unweighted_train_losses.record(
            {label: loss.detach().cpu().item() for label, loss in losses.items()}
        )

    def log_event(self, event: str) -> None:
        self.events.append(
            {"event": event, "iter": self.trainer.iter, "epoch": self.trainer.epoch}
        )

    def print_losses_and_weights(self, timing_str: str) -> None:
        """Print the full-set losses and the weights, e.g. before training."""
        trainer = self.trainer
        train_losses = {
            label: container.compute_full_train_loss(trainer.model).item()
            for label, container in trainer.loss_containers.items()
        }
        self._print_summary(
            f"Epoch: {trainer.epoch}  (iter {trainer.iter}, {timing_str})",
            "Training loss (full set)",
            train_losses,
            trainer.compute_full_unweighted_val_losses(),
        )


def _format_time(seconds: float) -> str:
    """Wall clock time as hours:minutes:seconds, e.g. 1:02:03."""
    return str(timedelta(seconds=round(seconds)))
