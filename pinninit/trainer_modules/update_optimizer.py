from __future__ import annotations

from typing import TYPE_CHECKING

from .loss_term_staggering import LossTermStaggering
from .trainer_module import PlateauTrainerModule

if TYPE_CHECKING:
    from ..optimizer.optimizer import Optimizer
    from ..trainer.trainer import Trainer
    from ..trainer.trainer_logger import TrainerLogger


class UpdateOptimizer(PlateauTrainerModule):
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
        super().__init__(patience, frac_tol, min_iter)
        self.coast_iters_after_update = coast_iters_after_update
        self.new_optimizer = new_optimizer
        self.updated = False
        self.stop_barrier_label = self.__class__.__name__

    def on_train_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        # Initialise plateau detectors
        self._init_plateau_detectors(trainer)

        # Training can only stop after optimizer has been updated
        trainer.state.init_stop_barrier(self.stop_barrier_label)

    def on_step_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        # Don't do anything if already updated
        if self.updated:
            return

        # Wait until min_iter to update plateau detector
        if trainer.iter < self.min_iter:
            return

        # Do nothing until all staggered loss terms are active (if staggering is used)
        ### Still think there's a more general way to do this, but fine for now
        staggering_label = LossTermStaggering.__name__
        if trainer.state.has_stop_barrier(staggering_label) and not (
            trainer.state.is_stop_barrier_met(staggering_label)
        ):
            return

        # If validation loss has plateaued, update optimizer
        if self._step_plateau_detectors(trainer):
            # Reload the best model before switching optimizer
            trainer.load_best_model()

            # Reset the best model tracker
            trainer.reset_best_model()

            # Update optimizer
            trainer.optimizer = self.new_optimizer
            trainer.optimizer.attach_trainer(trainer)
            self.updated = True
            print(f"Optimizer updated on iteration {trainer.iter}.")

            # Restart batch cycling so full-batch optimizers see fresh state
            for container in trainer.loss_containers.values():
                container.reset_train_batches()

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
        if self.updated:
            return

        # Fully re-initialise plateau detectors in case new containers were added
        self._init_plateau_detectors(trainer)
