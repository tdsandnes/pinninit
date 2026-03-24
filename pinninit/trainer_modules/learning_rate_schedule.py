from __future__ import annotations

from typing import TYPE_CHECKING

from ..trainer.trainer import Trainer
from .trainer_module import TrainerModule
from .trainer_module_utils import LossPlateauDetector

if TYPE_CHECKING:
    from ..trainer.trainer_logger import TrainerLogger


class LearningRateSchedule(TrainerModule):
    """Trainer module for updating the optimizer learning rate dynamically."""

    # Class attributes
    runs_during_coast = True

    def __init__(
        self,
        decay_factor: float,
        n_events: int,
        patience: int,
        warmup_iters: int = 0,
        lr_init: float | None = None,
        lr_after_warmup: float | None = None,
        frac_tol: float = 0.0,
        min_iter: int = 0,
        coast_iters_after_update: int = 0,
    ) -> None:
        self.decay_factor = decay_factor
        self.n_events = n_events
        self.patience = patience
        self.warmup_iters = warmup_iters
        self.lr_init = lr_init
        self.lr_after_warmup = lr_after_warmup
        self.frac_tol = frac_tol
        self.min_iter = min_iter
        self.coast_iters_after_update = coast_iters_after_update
        self.event_count = 0
        self.stop_barrier_label = self.__class__.__name__

    def on_train_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        # Initialise plateau detectors
        self._init_plateau_detectors(trainer)

        # Training can only stop after learning rate schedule is finished
        trainer.state.init_stop_barrier(self.stop_barrier_label)

        # Set initial learning rate
        param_groups = trainer.optimizer.optimizer.param_groups

        # Use optimizer lr if lr_after_warmup not provided
        if self.lr_after_warmup is None:
            self.lr_after_warmup = param_groups[0]["lr"]

        # If lr_init not provided, default to 0
        if self.lr_init is None:
            self.lr_init = self.lr_after_warmup

        # Set initial LR
        for group in param_groups:
            group["lr"] = self.lr_init

    def on_train_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        pass

    def on_step_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        pass

    def on_step_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        # Don't do anything if already finished
        if self.event_count >= self.n_events:
            return

        # Wait until min_iter to update learning rate
        if trainer.iter < self.min_iter:
            return

        # Linear warmup
        if (
            self.warmup_iters > 0
            and self.lr_init is not None
            and self.lr_after_warmup is not None
            and trainer.iter < self.warmup_iters
        ):
            current_iter_in_warmup = trainer.iter + 1
            total_warmup_iters = self.warmup_iters
            lr = self.lr_init + (self.lr_after_warmup - self.lr_init) * (
                current_iter_in_warmup / total_warmup_iters
            )
            for group in trainer.optimizer.optimizer.param_groups:
                group["lr"] = lr

            # Print message when warmup finishes
            if trainer.iter + 1 == self.warmup_iters:
                print(f"Learning rate warmup completed on iteration {trainer.iter}.")
                if logger is not None:
                    logger.log_event("learning_rate_warmup_complete", trainer.iter)

            return

        # Get validation losses
        if trainer.logger.unweighted_val_losses.has_value:
            unweighted_val_losses = trainer.logger.unweighted_val_losses.value
        else:
            raise RuntimeError(
                f"Validation losses have not been recorded for iter {trainer.iter}."
            )

        # Update loss plateau detectors
        for label, plateau_detector in self.plateau_detectors.items():
            if label in unweighted_val_losses:
                plateau_detector.step(unweighted_val_losses[label])
            elif label == self.weighted_avg_label:
                plateau_detector.step(float(trainer.weighted_avg_val_loss))
            else:
                raise KeyError(f"No loss value found for label {label}")

        # If validation loss has plateaued, decay learning rate
        if all(
            detector.plateau_detected for detector in self.plateau_detectors.values()
        ):
            # Reload the best model
            trainer.load_best_model()

            # Decay lr
            for group in trainer.optimizer.optimizer.param_groups:
                old_lr = group["lr"]
                new_lr = old_lr * self.decay_factor
                group["lr"] = new_lr
                print(
                    f"Learning rate decayed from {old_lr:.6g} to {new_lr:.6g} on "
                    f"iteration {trainer.iter}."
                )

            # Log learning rate decay
            if logger is not None:
                logger.log_event("learning_rate_decay", trainer.iter)

            # Notify trainer that its state has changed
            trainer.state.state_updated = True

            # Set coast phase for a number of iterations
            if self.coast_iters_after_update > 0:
                trainer.state.coast_until_iter = max(
                    trainer.state.coast_until_iter,
                    trainer.iter + self.coast_iters_after_update,
                )

            # Re-initialise plateau detectors
            self._init_plateau_detectors(trainer)

            # Update counter
            self.event_count += 1

            # Remove stop barrier
            if self.event_count >= self.n_events:
                trainer.state.update_stop_barrier(self.stop_barrier_label, True)

    def on_state_change(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        if self.event_count < self.n_events:
            return

        # Fully re-initialise plateau detectors in case new loss groups have been added
        self._init_plateau_detectors(trainer)

    def _init_plateau_detectors(self, trainer: Trainer) -> None:
        # Initialise plateau detector for each loss term group
        self.plateau_detectors = {}
        #    group_label: LossPlateauDetector(self.patience, self.frac_tol)
        #    for group_label in trainer.loss_groups
        # }
        # Add plateau detector for weighted average val loss
        self.weighted_avg_label = "weighted_avg"
        self.plateau_detectors[self.weighted_avg_label] = LossPlateauDetector(
            self.patience, self.frac_tol
        )
