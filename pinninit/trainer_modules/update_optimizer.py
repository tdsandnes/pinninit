from __future__ import annotations

from typing import TYPE_CHECKING

from ..optimizer.optimizer import Optimizer
from ..trainer.trainer import Trainer
from .trainer_module import TrainerModule
from .trainer_module_utils import LossPlateauDetector

if TYPE_CHECKING:
    from ..trainer.trainer_logger import TrainerLogger


class UpdateOptimizer(TrainerModule):
    """Trainer module for updating the optimizer based on validation loss."""

    # Class attributes
    runs_during_coast = False

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

    def on_train_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        self.plateau_detector = LossPlateauDetector(self.patience, self.frac_tol)

        # Training can only stop after optimizer has been updated
        trainer.state.init_stop_barrier(self.stop_barrier_label)

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
        # Don't do anything if already updated
        if self.updated:
            return

        # Wait until min_iter to update plateau detector
        if trainer.iter >= self.min_iter:
            # Update loss plateau detector
            plateau_quantity = trainer.weighted_avg_val_loss
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

                # Set batching of loss term groups if needed
                if self.new_optimizer.requires_full_batch:
                    for group in trainer.loss_groups.values():
                        group.set_batching(batch_size_train=None, batch_size_val=None)
                else:
                    for group in trainer.loss_groups.values():
                        if not group.is_default_batching:
                            group.reset_to_default_batching()

                # Log optimizer update
                if logger is not None:
                    logger.log_event("optimizer_update", trainer.iter)

                # Notify trainer that its state has changed
                trainer.state.state_updated = True

                # Set coast phase for a number of iterations
                if self.coast_iters_after_update > 0:
                    trainer.state.coast_until_iter = max(
                        trainer.state.coast_until_iter,
                        trainer.iter + self.coast_iters_after_update,
                    )

                # Remove stop barrier
                trainer.state.update_stop_barrier(self.stop_barrier_label, True)

    def on_state_change(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        # Reset counter if another module changed trainer state
        self.plateau_detector.reset()
