from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trainer import Trainer
    from .trainer_logger import TrainerLogger


class TrainerState:
    def __init__(self) -> None:
        self.state_updated = False
        self.coast_until_epoch: int = 0
        self.stop_requested: bool = False

        self._stop_barriers: dict[str, bool] = {}
        self._was_prev_step_coast: bool | None = None
        self._is_coasting: bool | None = None

    @property
    def is_ready_to_stop(
        self,
    ) -> bool:
        """Checks if all stop barriers have been met."""
        return all(self._stop_barriers.values()) and self.stop_requested

    @property
    def is_coasting(self) -> bool:
        """Check if trainer is in coast phase."""
        if self._is_coasting is None:
            raise ValueError(
                "Coast state has not been initialised for the current step."
            )
        return self._is_coasting

    def is_stop_barrier_met(self, label: str) -> bool:
        """Checks if the stop barrier with the given label has been met."""
        if label in self._stop_barriers:
            return self._stop_barriers[label]
        else:
            raise KeyError(f"Stop barrier '{label}' not found.")

    def has_stop_barrier(self, label: str) -> bool:
        """True if a stop barrier with the given label has been registered."""
        return label in self._stop_barriers

    def on_step_start(self, trainer: Trainer, logger: TrainerLogger | None) -> None:
        self._is_coasting = trainer.epoch <= self.coast_until_epoch
        self._check_coast_transition(trainer, logger)

    def on_step_end(self, trainer: Trainer, logger: TrainerLogger | None) -> None:
        self._was_prev_step_coast = self.is_coasting
        self._is_coasting = None

        if self.state_updated:
            # Reset stop_requested so that state_updated prevents immediate stopping
            self.stop_requested = False

            # Reset state_updated
            self.state_updated = False

    def init_stop_barrier(self, label: str) -> None:
        """Adds a stop barrier with the given label."""
        self._stop_barriers[label] = False

    def update_stop_barrier(self, label: str, status: bool) -> None:
        """Updates the status of the stop barrier with the given label."""
        if label in self._stop_barriers:
            self._stop_barriers[label] = status
        else:
            raise KeyError(f"Stop barrier '{label}' not found.")

    def _check_coast_transition(
        self, trainer: Trainer, logger: TrainerLogger | None
    ) -> None:
        if self._was_prev_step_coast is None:
            return

        # Entering coast
        if not self._was_prev_step_coast and self.is_coasting:
            print(f"Coasting started on epoch {trainer.epoch}")

            if logger is not None:
                logger.log_event("enter_coast", trainer.epoch)

            for module in trainer.trainer_modules:
                module.on_state_change(trainer, logger)

        # Exiting coast
        elif self._was_prev_step_coast and not self.is_coasting:
            print(f"Coasting ended on epoch {trainer.epoch}")

            if logger is not None:
                logger.log_event("exit_coast", trainer.epoch)

            for module in trainer.trainer_modules:
                module.on_state_change(trainer, logger)
