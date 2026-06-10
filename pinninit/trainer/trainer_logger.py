from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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
        self.unweighted_val_losses = PointInStepRecorder(PointInStep.LAST)
        self.loss_term_weights = PointInStepRecorder(PointInStep.FIRST)

        # Per-step logs (unchanged, useful for fine-grained plotting)
        self.pre_step_train_losses_log: list[dict[str, float]] = []
        self.post_step_val_losses_log: list[dict[str, float]] = []
        self.loss_term_weights_log: list[dict[str, float]] = []

        # Accumulator for averaging train losses across steps within an epoch
        self._epoch_train_loss_accum: list[dict[str, float]] = []

        # Per-epoch logs (full-set val loss, mean train loss, weights at epoch end)
        self.epoch_train_losses_log: list[dict[str, float]] = []
        self.epoch_val_losses_log: list[dict[str, float]] = []
        self.epoch_loss_term_weights_log: list[dict[str, float]] = []

    def on_step_start(self) -> None:
        self.unweighted_train_losses.reset()
        self.unweighted_val_losses.reset()
        self.loss_term_weights.reset()

    def log_values(self) -> None:
        if self.unweighted_train_losses.has_value:
            self.pre_step_train_losses_log.append(self.unweighted_train_losses.value)
            # Accumulate for epoch average
            self._epoch_train_loss_accum.append(self.unweighted_train_losses.value)
        else:
            raise RuntimeError(
                "Unweighted training losses were not recorded during step."
            )
        if self.unweighted_val_losses.has_value:
            self.post_step_val_losses_log.append(self.unweighted_val_losses.value)
        else:
            raise RuntimeError(
                "Unweighted validation losses were not recorded during step."
            )
        if self.loss_term_weights.has_value:
            self.loss_term_weights_log.append(self.loss_term_weights.value)
        else:
            raise RuntimeError("Loss term weights were not recorded during step.")

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

        # Reset accumulator for next epoch
        self._epoch_train_loss_accum = []

        # Print if on a print checkpoint
        if trainer.epoch % self.print_freq == 0:
            self._print_epoch(epoch_train_losses, full_val_losses, epoch_weights)

    def _print_epoch(
        self,
        train_losses: dict[str, float],
        val_losses: dict[str, float],
        weights: dict[str, float],
    ) -> None:
        trainer = self.trainer
        weight_sum = sum(weights.values())
        weighted_sum_train = sum(
            w * loss
            for w, loss in zip(weights.values(), train_losses.values(), strict=True)
        )
        weighted_sum_val = sum(
            w * loss
            for w, loss in zip(weights.values(), val_losses.values(), strict=True)
        )
        print(
            "-------------------------------------------------------------------\n"
            f"Epoch: {trainer.epoch}  (iter {trainer.iter}): \n"
            "-------------------------------------------------------------------\n"
            f"Training Loss (epoch mean of minibatch estimates): \n"
            f"- unweighted contributions:   {train_losses} \n"
            f"- weighted sum:               {weighted_sum_train} \n"
            f"- weighted average:           {weighted_sum_train / weight_sum} \n"
            f"Validation Loss (full set): \n"
            f"- unweighted contributions:   {val_losses} \n"
            f"- weighted sum:               {weighted_sum_val} \n"
            f"- weighted average:           {weighted_sum_val / weight_sum} \n"
            f"Loss term weights:            {weights} \n"
        )

    def record_train_losses(self, losses: dict[str, torch.Tensor]) -> None:
        self.unweighted_train_losses.record(
            {label: loss.detach().cpu().item() for label, loss in losses.items()}
        )

    def record_val_losses(self, losses: dict[str, torch.Tensor]) -> None:
        self.unweighted_val_losses.record(
            {label: loss.detach().cpu().item() for label, loss in losses.items()}
        )

    def record_loss_term_weights(self, weights: dict[str, float]) -> None:
        self.loss_term_weights.record(dict(weights))

    def log_event(self, event: str, iter: int) -> None:
        self.events.append({"event": event, "iter": iter, "epoch": self.trainer.epoch})

    def print_losses_and_weights(self, timing_str: str) -> None:
        trainer = self.trainer
        loss_term_weights = [
            container.loss_term_weight for container in trainer.loss_containers.values()
        ]
        train_losses = [
            container.compute_unweighted_train_loss(trainer.model).item()
            for container in trainer.loss_containers.values()
        ]
        val_losses = [
            container.compute_full_val_loss(trainer.model).item()
            for container in trainer.loss_containers.values()
        ]
        print(
            "-------------------------------------------------------------------\n"
            f"Epoch: {trainer.epoch}  Iter: {trainer.iter}  ({timing_str}): \n"
            "-------------------------------------------------------------------\n"
            f"Training Loss: \n"
            f"- unweighted contributions:   {train_losses} \n"
            f"Validation Loss (full set): \n"
            f"- unweighted contributions:   {val_losses} \n"
            f"Loss term weights:            {loss_term_weights} \n"
        )
