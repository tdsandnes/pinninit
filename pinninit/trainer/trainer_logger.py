from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .point_in_step import PointInStep, PointInStepRecorder

if TYPE_CHECKING:
    from typing import Any

    from .trainer import Trainer


class TrainerLogger:
    def __init__(self, trainer: Trainer, print_freq: int = 500):
        self.trainer = trainer
        self.print_freq = print_freq
        self.events: list[dict[str, Any]] = []

        # Initialise objects to record quantities at specific points in step
        # Record train losses at begining of step to avoid having to recompute it at end
        self.unweighted_train_losses = PointInStepRecorder(PointInStep.FIRST)
        # Record val losses at end of step to avoid having to recompute it at beginning
        self.unweighted_val_losses = PointInStepRecorder(PointInStep.LAST)
        # Record weights at beginning of step. They should not change within step
        self.loss_term_weights = PointInStepRecorder(PointInStep.FIRST)

        # Logs of recorded quantities
        self.pre_step_train_losses_log: list[list[float]] = []
        self.post_step_val_losses_log: list[list[float]] = []
        self.loss_term_weights_log: list[list[float]] = []

    def on_step_start(self) -> None:
        # Reset recorders
        self.unweighted_train_losses.reset()
        self.unweighted_val_losses.reset()
        self.loss_term_weights.reset()

    def log_values(self) -> None:
        # Update logs
        if self.unweighted_train_losses.has_value:
            self.pre_step_train_losses_log.append(self.unweighted_train_losses.value)
        else:
            raise RuntimeError(
                "Unweighted training losses were not recorded " "during step."
            )
        if self.unweighted_val_losses.has_value:
            self.post_step_val_losses_log.append(self.unweighted_val_losses.value)
        else:
            raise RuntimeError(
                "Unweighted validation losses were not recorded " "during step."
            )
        if self.loss_term_weights.has_value:
            self.loss_term_weights_log.append(self.loss_term_weights.value)
        else:
            raise RuntimeError("Loss term weights were not recorded during step.")

    def record_train_losses(self, losses: list[torch.Tensor]) -> None:
        self.unweighted_train_losses.record(
            [loss.detach().cpu().item() for loss in losses]
        )

    def record_val_losses(self, losses: list[torch.Tensor]) -> None:
        self.unweighted_val_losses.record(
            [loss.detach().cpu().item() for loss in losses]
        )

    def record_loss_term_weights(self, weights: list[torch.Tensor]) -> None:
        self.loss_term_weights.record(
            [weight.detach().cpu().item() for weight in weights]
        )

    def log_event(self, event: str, iter: int) -> None:
        self.events.append(
            {
                "event": event,
                "iter": iter,
            }
        )

    def print_losses_and_weights(self, trainer: Trainer, timing_str: str) -> None:
        loss_term_weights = [
            group.weight.detach().cpu().item() for group in trainer.loss_groups.values()
        ]
        train_losses = [
            group.compute_unweighted_loss_train(trainer.model).item()
            for group in trainer.loss_groups.values()
        ]
        val_losses = [
            group.compute_unweighted_loss_val(trainer.model).item()
            for group in trainer.loss_groups.values()
        ]

        print(
            "-------------------------------------------------------------------\n"
            f"Iteration: {trainer.iter} ({timing_str}): \n"
            "-------------------------------------------------------------------\n"
            f"Training Loss: \n"
            f"- unweighted contributions:   {train_losses} \n"
            f"Validation Loss: \n"
            f"- unweighted contributions:   {val_losses} \n"
            f"Loss group weights:           {loss_term_weights} \n"
        )

    def print_after_step(self, trainer: Trainer) -> None:
        if trainer.iter % self.print_freq != 0:
            return

        train_losses = self.pre_step_train_losses_log[-1]
        val_losses = self.post_step_val_losses_log[-1]
        loss_term_weights = self.loss_term_weights_log[-1]

        weighted_sum_train_loss = sum(
            [
                weight * loss
                for weight, loss in zip(loss_term_weights, train_losses, strict=True)
            ]
        )
        weighted_sum_val_loss = sum(
            [
                weight * loss
                for weight, loss in zip(loss_term_weights, val_losses, strict=True)
            ]
        )
        weighted_avg_train_loss = weighted_sum_train_loss / sum(loss_term_weights)
        weighted_avg_val_loss = weighted_sum_val_loss / sum(loss_term_weights)

        print(
            "-------------------------------------------------------------------\n"
            f"Iteration: {trainer.iter}: \n"
            "-------------------------------------------------------------------\n"
            f"Pre-step Training Loss: \n"
            f"- unweighted contributions:   {train_losses} \n"
            f"- weighted sum:               {weighted_sum_train_loss} \n"
            f"- weighted average:           {weighted_avg_train_loss} \n"
            f"Post-step Validation Loss: \n"
            f"- unweighted contributions:   {val_losses} \n"
            f"- weighted sum:               {weighted_sum_val_loss} \n"
            f"- weighted average:           {weighted_avg_val_loss} \n"
            f"Loss group weights:           {loss_term_weights} \n"
        )
