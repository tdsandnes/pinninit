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

    def on_train_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        # Initialise plateau detectors
        self._init_plateau_detectors(trainer)

    def on_epoch_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        # Wait until min_epoch to update plateau detectors
        if trainer.epoch < self.min_epoch:
            return

        # If validation loss has plateaued, request stop
        if self._step_plateau_detectors_full(trainer):
            trainer.state.stop_requested = True

            # Log early stopping request
            if logger is not None:
                logger.log_event("early_stopping_requested")

    def on_state_change(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        # Fully re-initialise plateau detectors in case new containers were added
        self._init_plateau_detectors(trainer)
