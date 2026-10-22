from __future__ import annotations

from typing import TYPE_CHECKING

from .trainer_module import PlateauTrainerModule

if TYPE_CHECKING:
    from ..trainer.trainer import Trainer
    from ..trainer.trainer_logger import TrainerLogger


class EarlyStopping(PlateauTrainerModule):
    """Trainer module for early stopping based on validation loss."""

    # Class attributes
    runs_during_coast = False
    stops_training = True

    def on_train_start(self, trainer: Trainer, logger: TrainerLogger) -> None:
        # Start detecting plateaus from scratch
        self.plateau_detector.reset()

    def on_epoch_end(self, trainer: Trainer, logger: TrainerLogger) -> None:
        # Wait until min_epoch to update the plateau detector
        if trainer.epoch < self.min_epoch:
            return

        # If validation loss has plateaued, request stop
        if self._step_plateau_detector(trainer):
            trainer.state.stop_requested = True

            # Log early stopping request
            logger.log_event("early_stopping_requested")

    def on_state_change(self, trainer: Trainer, logger: TrainerLogger) -> None:
        # Start detecting plateaus from scratch, as the trainer state changed
        self.plateau_detector.reset()
