from __future__ import annotations

from typing import TYPE_CHECKING

from .trainer_module_utils import LossPlateauDetector

if TYPE_CHECKING:
    from typing import ClassVar

    from ..trainer.trainer import Trainer
    from ..trainer.trainer_logger import TrainerLogger


class TrainerModule:
    """Base class for trainer modules."""

    # Class attributes
    runs_during_coast: ClassVar[bool] = False

    def on_train_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        pass

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
        pass

    def on_epoch_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        """Called at the end of each epoch (a full pass over the training data)."""
        pass

    def on_state_change(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        """Called when another module changes the trainer state
        (e.g. optimizer updated)."""
        pass


class PlateauTrainerModule(TrainerModule):
    """Base class for trainer modules that act when validation loss plateaus."""

    def __init__(self, patience: int, frac_tol: float = 0.0, min_iter: int = 0) -> None:
        self.patience = patience
        self.frac_tol = frac_tol
        self.min_iter = min_iter

        self.weighted_avg_label = "weighted_avg"
        self.plateau_detectors: dict[str, LossPlateauDetector] = {}

    def _init_plateau_detectors(self, trainer: Trainer) -> None:
        # Plateau detector for weighted average val loss
        ### Could also add a detector per loss term container here
        self.plateau_detectors = {
            self.weighted_avg_label: LossPlateauDetector(self.patience, self.frac_tol)
        }

    def _step_plateau_detectors(self, trainer: Trainer) -> bool:
        """Feed latest validation losses to detectors. Returns True if all
        detectors have detected a plateau."""
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

        return all(
            detector.plateau_detected for detector in self.plateau_detectors.values()
        )

    def _reset_plateau_detectors(self) -> None:
        for plateau_detector in self.plateau_detectors.values():
            plateau_detector.reset()

    def _step_plateau_detectors_full(self, trainer: Trainer) -> bool:
        """Feed full-set validation losses to detectors. Returns True if all
        detectors have detected a plateau."""
        full_losses = trainer.compute_full_unweighted_val_losses()
        for label, detector in self.plateau_detectors.items():
            if label in full_losses:
                detector.step(full_losses[label])
            elif label == self.weighted_avg_label:
                detector.step(trainer.compute_full_weighted_avg_val_loss)
            else:
                raise KeyError(f"No loss value found for label {label}")
        return all(d.plateau_detected for d in self.plateau_detectors.values())
