from __future__ import annotations

from ..optimizer.optimizer import Optimizer
from .trainer import Trainer, TrainerModule
from .training_utils import LossPlateauDetector


class UpdateOptimizer(TrainerModule):
    """Trainer module for updating the optimizer based on validation loss."""

    def __init__(
        self, new_optimizer: Optimizer, patience: int, frac_tol: float = 0.0
    ) -> None:
        self.patience = patience
        self.frac_tol = frac_tol
        self.new_optimizer = new_optimizer
        self.updated = False

    def on_train_start(self, trainer: Trainer) -> None:
        self.plateau_detector = LossPlateauDetector(self.patience, self.frac_tol)

        # Training can only stop after optimizer has been updated
        trainer._init_stop_barrier(self.__class__.__name__)

    def on_train_end(self, trainer: Trainer) -> None:
        pass

    def on_step_start(self, trainer: Trainer) -> None:
        pass

    def on_step_end(self, trainer: Trainer) -> None:
        # Don't do anything if already updated
        if self.updated:
            return

        # Update loss plateau detector
        self.plateau_detector.step(trainer.post_step_val_loss)

        # If validation loss has plateaued, update optimizer
        if self.plateau_detector.plateau_detected:
            trainer.optimizer = self.new_optimizer
            trainer.optimizer.attach_trainer(trainer)
            self.updated = True
            print(f"Optimizer updated on iteration {trainer.iter}.")

            # Remove stop barrier
            trainer._update_stop_barrier(self.__class__.__name__, True)

            # Notify trainer that its state has changed
            trainer._state_updated = True

    def on_state_change(self, trainer: Trainer) -> None:
        pass
