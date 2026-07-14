from __future__ import annotations

from typing import TYPE_CHECKING

from .trainer_module import PlateauTrainerModule

if TYPE_CHECKING:
    from typing import Any

    from ..trainer.trainer import Trainer
    from ..trainer.trainer_logger import TrainerLogger


class LearningRateSchedule(PlateauTrainerModule):
    """Trainer module for updating the optimizer learning rate dynamically."""

    # Class attributes
    runs_during_coast = True

    def __init__(
        self,
        decay_factor: float,
        n_events: int,
        patience: int,
        warmup_epochs: int = 0,
        lr_init: float | None = None,
        lr_after_warmup: float | None = None,
        frac_tol: float = 0.0,
        min_epoch: int = 0,
        coast_epochs_after_update: int = 0,
    ) -> None:
        super().__init__(patience, frac_tol, min_epoch)
        self.decay_factor = decay_factor
        self.n_events = n_events
        self.warmup_epochs = warmup_epochs
        self.lr_init = lr_init
        self.lr_after_warmup = lr_after_warmup
        self.coast_epochs_after_update = coast_epochs_after_update
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

        # If lr_init not provided, default to lr_after_warmup
        if self.lr_init is None:
            self.lr_init = self.lr_after_warmup

        # Set initial LR
        for group in param_groups:
            group["lr"] = self.lr_init

    def on_epoch_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        # Don't do anything if already finished
        if self.event_count >= self.n_events:
            return

        # Wait until min_epoch to update learning rate
        if trainer.epoch < self.min_epoch:
            return

        # Linear warmup
        if (
            self.warmup_epochs > 0
            and self.lr_init is not None
            and self.lr_after_warmup is not None
            and trainer.epoch <= self.warmup_epochs
        ):
            lr = self.lr_init + (self.lr_after_warmup - self.lr_init) * (
                trainer.epoch / self.warmup_epochs
            )
            for group in trainer.optimizer.optimizer.param_groups:
                group["lr"] = lr

            # Print message when warmup finishes
            if trainer.epoch == self.warmup_epochs:
                print(f"Learning rate warmup completed after epoch {trainer.epoch}.")
                if logger is not None:
                    logger.log_event("learning_rate_warmup_complete")

            return

        # Return if coasting
        if trainer.state.is_coasting:
            return

        # If validation loss has plateaued, decay learning rate
        if self._step_plateau_detectors_full(trainer):
            # Load the best model
            trainer.load_best_model()

            # Decay lr
            for group in trainer.optimizer.optimizer.param_groups:
                old_lr = group["lr"]
                new_lr = old_lr * self.decay_factor
                group["lr"] = new_lr
                print(
                    f"Learning rate decayed from {old_lr:.6g} to {new_lr:.6g} on "
                    f"epoch {trainer.epoch}."
                )

            # Log learning rate decay
            if logger is not None:
                logger.log_event("learning_rate_decay")

            # Notify trainer that its state has changed
            trainer.state.state_updated = True

            # Set coast phase for a number of epochs
            if self.coast_epochs_after_update > 0:
                trainer.state.coast_until_epoch = max(
                    trainer.state.coast_until_epoch,
                    trainer.epoch + self.coast_epochs_after_update,
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
        if self.event_count >= self.n_events:
            return

        # Fully re-initialise plateau detectors in case new containers were added
        self._init_plateau_detectors(trainer)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["event_count"] = self.event_count
        state["lr_init"] = self.lr_init
        state["lr_after_warmup"] = self.lr_after_warmup
        return state

    def load_state_dict(self, state: dict[str, Any], trainer: Trainer) -> None:
        super().load_state_dict(state, trainer)
        self.event_count = state["event_count"]
        self.lr_init = state["lr_init"]
        self.lr_after_warmup = state["lr_after_warmup"]
