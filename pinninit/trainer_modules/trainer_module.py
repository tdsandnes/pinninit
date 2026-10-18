from __future__ import annotations

from typing import TYPE_CHECKING

from .trainer_module_utils import LossPlateauDetector

if TYPE_CHECKING:
    from typing import Any, ClassVar

    from ..trainer.trainer import Trainer
    from ..trainer.trainer_logger import TrainerLogger


class TrainerModule:
    """
    Base class for trainer modules.

    A module changes the training through the trainer: update_point_weights,
    update_loss_term_weight, replace_optimizer and state.update_stop_barrier.
    The trainer applies these changes once all the modules have run, so every
    module sees the same state and the order of the modules doesn't matter.
    """

    # Class attributes
    runs_during_coast: ClassVar[bool] = False
    # Whether on_pre_optimizer_step adapts the loss (e.g. loss term or point
    # weights). It isn't called while the optimizer requires a fixed loss, and
    # loading the best model also loads the module's state_dict from then.
    adapts_loss: ClassVar[bool] = False
    # Whether the module can stop training (with trainer.state.stop_requested),
    # so that training ends without max_epoch
    stops_training: ClassVar[bool] = False

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

    def on_pre_optimizer_step(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        """Called after the training loss and gradients are computed (residuals
        cached) but before the optimizer applies the parameter update.

        The weights modules update here (with trainer.update_point_weights and
        trainer.update_loss_term_weight) are applied once all modules have
        run, so every module sees the weights this step's loss was computed
        with.

        With an optimizer that evaluates the loss in a closure (e.g. L-BFGS),
        it is called before the loss is computed, so nothing is cached. It
        isn't called for modules that adapt the loss while the optimizer
        requires a fixed loss (e.g. L-BFGS)."""
        pass

    def on_step_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        pass

    def on_epoch_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        """Called at the end of each epoch (a full pass over the training data),
        after the epoch's results are logged.

        Act on the epoch's results (e.g. trainer.epoch_val_losses), so that
        what the other modules do at the same epoch end doesn't matter: the
        changes they make through the trainer are applied once all modules
        have run, and the best model is only loaded once in a step."""
        pass

    def on_state_change(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        """Called when another module changes the trainer state
        (e.g. optimizer updated)."""
        pass

    def summary(self) -> str:
        """Extra lines for the trainer's printed summaries, empty by default."""
        return ""

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any], trainer: Trainer) -> None:
        pass


class PlateauTrainerModule(TrainerModule):
    """Base class for trainer modules that act when validation loss plateaus."""

    def __init__(
        self, patience: int, frac_tol: float = 0.0, min_epoch: int = 0
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if min_epoch < 0:
            raise ValueError(f"min_epoch must be >= 0, got {min_epoch}")
        self.patience = patience
        self.frac_tol = frac_tol
        self.min_epoch = min_epoch

        self.geometric_mean_label = "geometric_mean"
        self.plateau_detectors: dict[str, LossPlateauDetector] = {}

    def _init_plateau_detectors(self, trainer: Trainer) -> None:
        # Plateau detector for geometric mean val loss
        ### Could also add a detector per loss term container here
        self.plateau_detectors = {
            self.geometric_mean_label: LossPlateauDetector(self.patience, self.frac_tol)
        }

    def _reset_plateau_detectors(self) -> None:
        for plateau_detector in self.plateau_detectors.values():
            plateau_detector.reset()

    def _step_plateau_detectors_full(self, trainer: Trainer) -> bool:
        """Feed the epoch's full-set validation losses to detectors. Returns True
        if all detectors have detected a plateau."""
        full_losses = trainer.epoch_val_losses
        for label, detector in self.plateau_detectors.items():
            if label in full_losses:
                detector.step(full_losses[label])
            elif label == self.geometric_mean_label:
                detector.step(trainer.epoch_geometric_mean_val_loss)
            else:
                raise KeyError(f"No loss value found for label {label}")
        return all(d.plateau_detected for d in self.plateau_detectors.values())

    def state_dict(self) -> dict[str, Any]:
        return {
            "plateau_detectors": {
                label: {
                    "best": d.best,
                    "best_history": list(d.best_history),
                    "plateau_detected": d.plateau_detected,
                }
                for label, d in self.plateau_detectors.items()
            }
        }

    def load_state_dict(self, state: dict[str, Any], trainer: Trainer) -> None:
        for label, d_state in state.get("plateau_detectors", {}).items():
            if label in self.plateau_detectors:
                detector = self.plateau_detectors[label]
                detector.best = d_state["best"]
                detector.best_history = list(d_state["best_history"])
                detector.plateau_detected = d_state["plateau_detected"]
