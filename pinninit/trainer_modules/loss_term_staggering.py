from __future__ import annotations

from typing import TYPE_CHECKING

from ..trainer.trainer import Trainer
from .trainer_module import TrainerModule
from .trainer_module_utils import LossPlateauDetector

if TYPE_CHECKING:
    from ..trainer.trainer_logger import TrainerLogger


class LossTermStaggering(TrainerModule):
    """
    Stagger introduction of loss terms
    """

    # Class attributes
    runs_during_coast = False

    def __init__(
        self,
        patience: int,
        frac_tol: float = 0.0,
        min_iter: int = 0,
        coast_iters_after_update: int = 0,
    ) -> None:
        self.patience = patience
        self.frac_tol = frac_tol
        self.min_iter = min_iter
        self.coast_iters_after_update = coast_iters_after_update

        self.staggering_group_dict: dict[int, list[str]] = {}
        self.current_stagger_idx = 0
        self.stop_barrier_label = self.__class__.__name__
        self.completed = False

    def on_train_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        self.current_stagger_idx = 0

        # Get the staggering values of groups
        self.staggering_group_dict.clear()
        for label, group in trainer._loss_groups.items():
            self.staggering_group_dict.setdefault(group.staggering, []).append(label)

        # Validate keys are consecutive integers starting at 0
        keys = sorted(self.staggering_group_dict.keys())
        expected_keys = list(range(len(keys)))

        if keys != expected_keys:
            raise ValueError(
                f"Staggering keys must be consecutive integers starting at 0. "
                f"Expected {expected_keys}, got {keys}."
            )

        # Initialise plateau detectors
        self._init_plateau_detectors(trainer)

        # Training can only stop after all loss terms have been activated
        trainer.state.init_stop_barrier(self.stop_barrier_label)

    def on_step_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        pass

    def on_step_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        # Wait until min_iter to update plateau detectors
        if trainer.iter < self.min_iter:
            return

        # All stagger groups already activated
        if self.completed:
            return

        # Get validation losses
        if trainer.logger.unweighted_val_losses.has_value:
            unweighted_val_losses = trainer.logger.unweighted_val_losses.value
        else:
            raise RuntimeError(
                f"Validation losses have not been recorded for iter {trainer.iter}."
            )

        # Update loss plateau detectors
        for label, plateau_detector in self.plateau_detectors.items():
            if label in unweighted_val_losses:
                plateau_detector.step(unweighted_val_losses[label])
            elif label == self.weighted_avg_label:
                plateau_detector.step(float(trainer.weighted_avg_val_loss))
            else:
                raise KeyError(f"No loss value found for label {label}")

        # If validation loss has plateaued, request stop
        if all(
            detector.plateau_detected for detector in self.plateau_detectors.values()
        ):
            # Update counter
            self.current_stagger_idx += 1
            current_stagger = self.current_stagger_idx

            # Reload the best model before activating loss groups
            trainer.load_best_model()

            # Reset the best model tracker
            trainer.reset_best_model()

            # Activate loss groups with value of counter
            loss_groups_to_activate = self.staggering_group_dict[current_stagger]
            for label in loss_groups_to_activate:
                trainer.loss_groups[label] = trainer._loss_groups[label]

                # Log loss group activation
                print(f"Activated loss term group {label} on iteration {trainer.iter}.")
                if logger is not None:
                    logger.log_event("activated_loss_group_" + label, trainer.iter)

            # Init new plateu detectors, including the newly activated groups
            self._init_plateau_detectors(trainer)

            # Check if all loss terms are now active
            if self.current_stagger_idx == len(self.staggering_group_dict) - 1:
                self.completed = True

                # Remove stop barrier
                trainer.state.update_stop_barrier(self.stop_barrier_label, True)

            # Notify trainer that its state has changed
            trainer.state.state_updated = True

            # Set coast phase for a number of iterations
            if self.coast_iters_after_update > 0:
                trainer.state.coast_until_iter = max(
                    trainer.state.coast_until_iter,
                    trainer.iter + self.coast_iters_after_update,
                )

    def on_train_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        pass

    def on_state_change(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        # Reset counter if another module changed trainer state
        for plateau_detector in self.plateau_detectors.values():
            plateau_detector.reset()

    def _init_plateau_detectors(self, trainer: Trainer) -> None:
        # Initialise plateau detector for each loss term group
        self.plateau_detectors = {}
        #    group_label: LossPlateauDetector(self.patience, self.frac_tol)
        #    for group_label in trainer.loss_groups
        # }
        # Add plateau detector for weighted average val loss
        self.weighted_avg_label = "weighted_avg"
        self.plateau_detectors[self.weighted_avg_label] = LossPlateauDetector(
            self.patience, self.frac_tol
        )
