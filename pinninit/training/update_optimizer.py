from __future__ import annotations

from ..optimizer.optimizer import Optimizer
from .trainer import Trainer, TrainerModule
from .training_utils import LossPlateauDetector


class UpdateOptimizer(TrainerModule):
    """Trainer module for updating the optimizer based on validation loss."""

    def __init__(
        self,
        new_optimizer: Optimizer,
        patience: int,
        frac_tol: float = 0.0,
        min_iter: int = 0,
        coast_iters_after_update: int = 0,
    ) -> None:
        self.patience = patience
        self.frac_tol = frac_tol
        self.min_iter = min_iter
        self.coast_iters_after_update = coast_iters_after_update
        self.new_optimizer = new_optimizer
        self.updated = False
        self.stop_barrier_label = self.__class__.__name__

    def runs_during_coast(self, trainer: Trainer) -> bool:
        return False

    def on_train_start(self, trainer: Trainer) -> None:
        self.plateau_detector = LossPlateauDetector(self.patience, self.frac_tol)

        # Training can only stop after optimizer has been updated
        trainer._init_stop_barrier(self.stop_barrier_label)

    def on_train_end(self, trainer: Trainer) -> None:
        pass

    def on_step_start(self, trainer: Trainer) -> None:
        # Remove stop barrier
        if self.updated and trainer._stop_barriers[self.stop_barrier_label] is False:
            trainer._update_stop_barrier(self.stop_barrier_label, True)

    def on_step_end(self, trainer: Trainer) -> None:
        # Don't do anything if already updated
        if self.updated:
            return

        # Wait until min_iter to update plateau detector
        if trainer.iter >= self.min_iter:
            # Update loss plateau detector
            plateau_quantity = trainer.weighted_average_val_loss
            self.plateau_detector.step(plateau_quantity)

            # If validation loss has plateaued, update optimizer
            if self.plateau_detector.plateau_detected:
                # Reload the best model before switching optimizer
                trainer.load_best_model()

                # Update optimizer
                trainer.optimizer = self.new_optimizer
                trainer.optimizer.attach_trainer(trainer)
                self.updated = True
                print(f"Optimizer updated on iteration {trainer.iter}.")

                # Log optimizer update
                trainer.state_updates.append(
                    {
                        "event": "optimizer_update",
                        "iter": trainer.iter,
                        "weighted_average_val_loss": trainer.weighted_average_val_loss,
                    }
                )

                # Notify trainer that its state has changed
                trainer._state_updated = True

                # Set coast phase for a number of iterations
                if self.coast_iters_after_update > 0:
                    trainer.coast_until_iter = max(
                        trainer.coast_until_iter,
                        trainer.iter + self.coast_iters_after_update,
                    )

    def on_state_change(self, trainer: Trainer) -> None:
        # Reset counter if another module changed trainer state
        self.plateau_detector.reset()
