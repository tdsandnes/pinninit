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
    module sees the same state. Modules that only change the training this way
    can run in any order (as the built-in modules do, which is tested).
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

    def on_train_start(self, trainer: Trainer, logger: TrainerLogger) -> None:
        pass

    def on_train_end(self, trainer: Trainer, logger: TrainerLogger) -> None:
        pass

    def on_step_start(self, trainer: Trainer, logger: TrainerLogger) -> None:
        pass

    def on_pre_optimizer_step(self, trainer: Trainer, logger: TrainerLogger) -> None:
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

    def on_step_end(self, trainer: Trainer, logger: TrainerLogger) -> None:
        pass

    def on_epoch_end(self, trainer: Trainer, logger: TrainerLogger) -> None:
        """Called at the end of each epoch (a full pass over the training data),
        after the epoch's results are logged.

        Act on the epoch's results (e.g. trainer.epoch_val_losses), so that
        what the other modules do at the same epoch end doesn't matter: the
        changes they make through the trainer are applied once all modules
        have run, and the best model is only loaded once in a step."""
        pass

    def on_state_change(self, trainer: Trainer, logger: TrainerLogger) -> None:
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
    """Base class for trainer modules that act when the geometric mean
    validation loss plateaus."""

    def __init__(
        self, patience: int, frac_tol: float = 0.0, min_epoch: int = 0
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if min_epoch < 0:
            raise ValueError(f"min_epoch must be >= 0, got {min_epoch}")
        self.min_epoch = min_epoch
        self.plateau_detector = LossPlateauDetector(patience, frac_tol)

    def _step_plateau_detector(self, trainer: Trainer) -> bool:
        """Give the epoch's geometric mean validation loss to the plateau
        detector. Returns True if it has detected a plateau."""
        self.plateau_detector.step(trainer.epoch_geometric_mean_val_loss)
        return self.plateau_detector.plateau_detected

    def state_dict(self) -> dict[str, Any]:
        detector = self.plateau_detector
        return {
            "plateau_detector": {
                "best": detector.best,
                "best_history": list(detector.best_history),
                "plateau_detected": detector.plateau_detected,
            }
        }

    def load_state_dict(self, state: dict[str, Any], trainer: Trainer) -> None:
        detector_state = state["plateau_detector"]
        self.plateau_detector.best = detector_state["best"]
        self.plateau_detector.best_history = list(detector_state["best_history"])
        self.plateau_detector.plateau_detected = detector_state["plateau_detected"]
