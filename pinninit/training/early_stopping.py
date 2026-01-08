from __future__ import annotations

from .trainer import Trainer, TrainerModule
from .training_utils import LossPlateauDetector


class EarlyStopping(TrainerModule):
    """Trainer module for early stopping based on validation loss."""

    def __init__(self, patience: int, frac_tol: float = 0.0) -> None:
        self.patience = patience
        self.frac_tol = frac_tol

    def on_train_start(self, trainer: Trainer) -> None:
        self.plateau_detector = LossPlateauDetector(self.patience, self.frac_tol)

    def on_train_end(self, trainer: Trainer) -> None:
        pass

    def on_step_start(self, trainer: Trainer) -> None:
        pass

    def on_step_end(self, trainer: Trainer) -> None:
        # Update loss plateau detector
        self.plateau_detector.step(trainer.post_step_val_loss)

        # If validation loss has plateaued, request stop
        if self.plateau_detector.plateau_detected:
            trainer._stop_requested = True

    def on_state_change(self, trainer: Trainer) -> None:
        # Reset counter if another module changed trainer state
        self.plateau_detector.reset_counter()
