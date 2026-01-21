from __future__ import annotations

from typing import TYPE_CHECKING

from ..trainer.trainer import Trainer
from .trainer_module import TrainerModule
from .trainer_module_utils import LossPlateauDetector

if TYPE_CHECKING:
    from ..trainer.trainer_logger import TrainerLogger


class EarlyStopping(TrainerModule):
    """Trainer module for early stopping based on validation loss."""

    # Class attributes
    runs_during_coast = False

    def __init__(self, patience: int, frac_tol: float = 0.0, min_iter: int = 0) -> None:
        self.patience = patience
        self.frac_tol = frac_tol
        self.min_iter = min_iter

    def on_train_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        # Initialise plateau detector for each loss term group
        self.plateau_detectors = [
            LossPlateauDetector(self.patience, self.frac_tol)
            for group in trainer.loss_groups.values()
        ]

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
        # Wait until min_iter to update plateau detectors
        if trainer.iter >= self.min_iter:
            # Get validation losses
            if trainer.logger.unweighted_val_losses.has_value:
                unweighted_val_losses = trainer.logger.unweighted_val_losses.value
            else:
                raise RuntimeError(
                    f"Validation losses have not been recorded for iter {trainer.iter}."
                )

            # Update loss plateau detectors
            for plateau_detector, val_loss in zip(
                self.plateau_detectors, unweighted_val_losses, strict=True
            ):
                plateau_detector.step(val_loss)

            # If validation loss has plateaued, request stop
            if all(detector.plateau_detected for detector in self.plateau_detectors):
                trainer.state.stop_requested = True

                # Log early stopping request
                if logger is not None:
                    logger.log_event("early_stopping_requested", trainer.iter)

    def on_state_change(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        # Reset counter if another module changed trainer state
        for plateau_detector in self.plateau_detectors:
            plateau_detector.reset()
