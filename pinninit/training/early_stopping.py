from __future__ import annotations

from .trainer import Trainer, TrainerModule
from .training_utils import LossPlateauDetector


class EarlyStopping(TrainerModule):
    """Trainer module for early stopping based on validation loss."""

    def __init__(self, patience: int, frac_tol: float = 0.0, min_iter: int = 0) -> None:
        self.patience = patience
        self.frac_tol = frac_tol
        self.min_iter = min_iter

    def on_train_start(self, trainer: Trainer) -> None:
        self.plateau_detector = LossPlateauDetector(self.patience, self.frac_tol)

    def on_train_end(self, trainer: Trainer) -> None:
        pass

    def on_step_start(self, trainer: Trainer) -> None:
        pass

    def on_step_end(self, trainer: Trainer) -> None:
        # Wait until min_iter to update plateau detector
        if trainer.iter >= self.min_iter:
            # Update loss plateau detector
            plateau_quantity = trainer.weighted_average_val_loss
            self.plateau_detector.step(plateau_quantity)

            # If validation loss has plateaued, request stop
            if self.plateau_detector.plateau_detected:
                trainer._stop_requested = True

    def on_state_change(self, trainer: Trainer) -> None:
        # Reset counter if another module changed trainer state
        self.plateau_detector.reset_counter()
