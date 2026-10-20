from __future__ import annotations

from typing import TYPE_CHECKING

from .trainer_module import PlateauTrainerModule
from .trainer_module_utils import check_coast_epochs

if TYPE_CHECKING:
    from typing import Any

    from ..trainer.trainer import Trainer
    from ..trainer.trainer_logger import TrainerLogger


class LossTermStaggering(PlateauTrainerModule):
    """
    Stagger introduction of loss terms
    """

    # Class attributes
    runs_during_coast = False

    def __init__(
        self,
        patience: int,
        frac_tol: float = 0.0,
        min_epoch: int = 0,
        coast_epochs_after_update: int = 0,
    ) -> None:
        super().__init__(patience, frac_tol, min_epoch)
        check_coast_epochs(coast_epochs_after_update)
        self.coast_epochs_after_update = coast_epochs_after_update

        self.staggering_container_dict: dict[int, list[str]] = {}
        self.current_stagger_idx = 0
        self.stop_barrier_label = self.__class__.__name__
        self.completed = False

    def on_train_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        self.current_stagger_idx = 0

        # Get the staggering values of containers
        self.staggering_container_dict.clear()
        for label, container in trainer.all_loss_containers.items():
            self.staggering_container_dict.setdefault(container.staggering, []).append(
                label
            )

        # Validate keys are consecutive integers starting at 0
        keys = sorted(self.staggering_container_dict.keys())
        expected_keys = list(range(len(keys)))

        if keys != expected_keys:
            raise ValueError(
                f"Staggering keys must be consecutive integers starting at 0. "
                f"Expected {expected_keys}, got {keys}."
            )

        # Start detecting plateaus from scratch
        self.plateau_detector.reset()

        # Training can only stop after all loss terms have been activated
        trainer.state.init_stop_barrier(self.stop_barrier_label)

        # With a single stage, every loss term is already active
        self.completed = len(self.staggering_container_dict) == 1
        if self.completed:
            trainer.state.update_stop_barrier(self.stop_barrier_label, True)

    def on_epoch_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        # Wait until min_epoch to update the plateau detector
        if trainer.epoch < self.min_epoch:
            return

        # All stagger groups already activated
        if self.completed:
            return

        # If validation loss has plateaued, activate next stagger group
        if self._step_plateau_detector(trainer):
            # Update counter
            self.current_stagger_idx += 1
            current_stagger = self.current_stagger_idx

            # Reload the best model before activating loss containers
            trainer.load_best_model()

            # Reset the best model tracker
            trainer.reset_best_model()

            # Activate loss containers with value of counter
            containers_to_activate = self.staggering_container_dict[current_stagger]
            for label in containers_to_activate:
                trainer.activate_loss_container(label)

                # Log loss container activation
                print(f"Activated loss term {label} on epoch {trainer.epoch}.")
                if logger is not None:
                    logger.log_event("activated_loss_term_" + label)

            # Start detecting plateaus from scratch, with the new loss terms
            self.plateau_detector.reset()

            # Check if all loss terms are now active
            if self.current_stagger_idx == len(self.staggering_container_dict) - 1:
                self.completed = True

                # Remove stop barrier
                trainer.state.update_stop_barrier(self.stop_barrier_label, True)

            # Notify trainer that its state has changed
            trainer.state.state_updated = True

            # Set coast phase for a number of epochs
            if self.coast_epochs_after_update > 0:
                trainer.state.coast_until_epoch = max(
                    trainer.state.coast_until_epoch,
                    trainer.epoch + self.coast_epochs_after_update,
                )

    def on_state_change(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        # Start detecting plateaus from scratch, as the trainer state changed
        self.plateau_detector.reset()

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["current_stagger_idx"] = self.current_stagger_idx
        state["completed"] = self.completed
        return state

    def load_state_dict(self, state: dict[str, Any], trainer: Trainer) -> None:
        super().load_state_dict(state, trainer)
        self.current_stagger_idx = state["current_stagger_idx"]
        self.completed = state["completed"]
        # Re-activate the stagger groups that were active at save time
        for idx in range(1, self.current_stagger_idx + 1):
            for label in self.staggering_container_dict[idx]:
                trainer.activate_loss_container(label)
