from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .point_in_step import PointInStep, PointInStepRecorder

if TYPE_CHECKING:
    from typing import Any

    from .trainer import Trainer


class TrainerLogger:
    def __init__(self, trainer: Trainer, print_freq: int = 1):
        self.trainer = trainer
        self.print_freq = print_freq  # now means "print every N epochs"
        self.events: list[dict[str, Any]] = []

        # Per-step recorders (unchanged)
        self.unweighted_train_losses = PointInStepRecorder(PointInStep.FIRST)

        # Accumulator for averaging train losses across steps within an epoch
        self._epoch_train_loss_accum: list[dict[str, float]] = []

        # Per-epoch logs (full-set val loss, mean train loss, weights at epoch end)
        self.epoch_train_losses_log: list[dict[str, float]] = []
        self.epoch_val_losses_log: list[dict[str, float]] = []
        self.epoch_loss_term_weights_log: list[dict[str, float]] = []
        self.epoch_learnt_consts_log: list[dict[str, float]] = []

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
        """Called at the end of each epoch. Computes full-set val losses,
        averages accumulated train losses, updates epoch logs, and prints
        if the epoch is a print checkpoint."""
        trainer = self.trainer

        # Full-set validation losses (chunked to keep peak memory bounded)
        full_val_losses = trainer.compute_full_unweighted_val_losses()

        # Mean of per-step minibatch train losses over the epoch
        # (approximation — exact only when all batches are the same size)
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

        # Reset accumulator for next epoch
        self._epoch_train_loss_accum = []

        # Print if on a print checkpoint
        if trainer.epoch % self.print_freq == 0:
            self._print_epoch(epoch_train_losses, full_val_losses, epoch_weights)

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

    def _brdr_s_str(self) -> str:
        """BRDR scaling factor line, empty if no BRDR module is active."""
        for module in self.trainer.trainer_modules:
            s = getattr(module, "scaling_factor_s", None)
            if s is not None:
                return f"BRDR scaling factor s: {s:.6g} \n"
        return ""

    def _print_epoch(
        self,
        train_losses: dict[str, float],
        val_losses: dict[str, float],
        weights: dict[str, float],
    ) -> None:
        trainer = self.trainer
        weighted_sum_train = sum(
            weights[label] * loss for label, loss in train_losses.items()
        )
        train_weight_sum = sum(weights[label] for label in train_losses)
        weighted_sum_val = sum(
            weights[label] * loss for label, loss in val_losses.items()
        )
        val_weight_sum = sum(weights[label] for label in val_losses)
        val_geometric_mean = math.exp(
            sum(math.log(loss) for loss in val_losses.values()) / len(val_losses)
        )
        print(
            "-------------------------------------------------------------------\n"
            f"Epoch: {trainer.epoch}  (iter {trainer.iter}): \n"
            "-------------------------------------------------------------------\n"
            f"Training Loss (epoch mean of minibatch estimates): \n"
            f"- unweighted contributions:   {train_losses} \n"
            f"- weighted sum:               {weighted_sum_train} \n"
            f"- weighted average:           {weighted_sum_train / train_weight_sum} \n"
            f"Validation Loss (full set): \n"
            f"- unweighted contributions:   {val_losses} \n"
            f"- weighted sum:               {weighted_sum_val} \n"
            f"- weighted average:           {weighted_sum_val / val_weight_sum} \n"
            f"- geometric mean:             {val_geometric_mean} \n"
            f"Loss term weights:            {weights} \n"
            f"{self._learnt_consts_str()}"
            f"{self._point_weights_str()}"
            f"{self._skip_alphas_str()}"
            f"{self._brdr_s_str()}"
        )

    def record_train_losses(self, losses: dict[str, torch.Tensor]) -> None:
        self.unweighted_train_losses.record(
            {label: loss.detach().cpu().item() for label, loss in losses.items()}
        )

    def log_event(self, event: str) -> None:
        self.events.append(
            {"event": event, "iter": self.trainer.iter, "epoch": self.trainer.epoch}
        )

    def print_losses_and_weights(self, timing_str: str) -> None:
        trainer = self.trainer
        loss_term_weights = {
            label: container.loss_term_weight
            for label, container in trainer.loss_containers.items()
        }
        train_losses = {
            label: container.compute_full_train_loss(trainer.model).item()
            for label, container in trainer.loss_containers.items()
        }
        val_losses = {
            label: container.compute_full_val_loss(trainer.model).item()
            for label, container in trainer.loss_containers.items()
        }
        print(
            "-------------------------------------------------------------------\n"
            f"Epoch: {trainer.epoch}  Iter: {trainer.iter}  ({timing_str}): \n"
            "-------------------------------------------------------------------\n"
            f"Training Loss: \n"
            f"- unweighted contributions:   {train_losses} \n"
            f"Validation Loss (full set): \n"
            f"- unweighted contributions:   {val_losses} \n"
            f"Loss term weights:            {loss_term_weights} \n"
            f"{self._learnt_consts_str()}"
            f"{self._point_weights_str()}"
            f"{self._skip_alphas_str()}"
            f"{self._brdr_s_str()}"
        )
