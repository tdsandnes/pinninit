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
        self.plateau_detector = LossPlateauDetector(self.patience, self.frac_tol)

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
        # Wait until min_iter to update plateau detector
        if trainer.iter >= self.min_iter:
            # Update loss plateau detector
            plateau_quantity = trainer.weighted_avg_val_loss
            self.plateau_detector.step(plateau_quantity)

            # If validation loss has plateaued, request stop
            if self.plateau_detector.plateau_detected:
                trainer.state.stop_requested = True

                # Log early stopping request
                if logger is not None:
                    logger.log_event("early_stopping_requested", trainer.iter)

    def on_state_change(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        # Reset counter if another module changed trainer state
        self.plateau_detector.reset()
