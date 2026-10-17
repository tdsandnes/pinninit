from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from .trainer import Trainer


class TrainerState:
    def __init__(self) -> None:
        self.state_updated = False
        self.coast_until_epoch: int = 0
        self.stop_requested: bool = False

        self._stop_barriers: dict[str, bool] = {}
        self._was_prev_step_coast: bool | None = None
        self._is_coasting: bool | None = None

    @property
    def is_ready_to_stop(self) -> bool:
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
        if label not in self._stop_barriers:
            raise KeyError(f"Stop barrier '{label}' not found.")
        return self._stop_barriers[label]

    def has_stop_barrier(self, label: str) -> bool:
        """True if a stop barrier with the given label has been registered."""
        return label in self._stop_barriers

    def init_stop_barrier(self, label: str) -> None:
        """Adds a stop barrier with the given label."""
        self._stop_barriers[label] = False

    def update_stop_barrier(self, label: str, status: bool) -> None:
        """Updates the status of the stop barrier with the given label."""
        if label not in self._stop_barriers:
            raise KeyError(f"Stop barrier '{label}' not found.")
        self._stop_barriers[label] = status

    def on_step_start(self, trainer: Trainer) -> None:
        self._is_coasting = trainer.epoch < self.coast_until_epoch

        # Entering or leaving the coast phase changes the trainer state
        if self._was_prev_step_coast not in (None, self._is_coasting):
            if self._is_coasting:
                print(f"Coasting started on epoch {trainer.epoch}")
                trainer.logger.log_event("enter_coast")
            else:
                print(f"Coasting ended on epoch {trainer.epoch}")
                trainer.logger.log_event("exit_coast")
            for module in trainer.trainer_modules:
                module.on_state_change(trainer, trainer.logger)

    def on_step_end(self) -> None:
        self._was_prev_step_coast = self.is_coasting
        self._is_coasting = None

        if self.state_updated:
            # Reset stop_requested so that state_updated prevents immediate stopping
            self.stop_requested = False
            self.state_updated = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "coast_until_epoch": self.coast_until_epoch,
            "stop_requested": self.stop_requested,
            "stop_barriers": dict(self._stop_barriers),
            "was_prev_step_coast": self._was_prev_step_coast,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the state saved by state_dict. The trainer modules must have
        registered their stop barriers already."""
        if state["stop_barriers"].keys() != self._stop_barriers.keys():
            raise ValueError(
                f"Saved stop barriers {sorted(state['stop_barriers'])} don't match "
                f"the registered stop barriers {sorted(self._stop_barriers)}"
            )
        self.coast_until_epoch = state["coast_until_epoch"]
        self.stop_requested = state["stop_requested"]
        self._stop_barriers.update(state["stop_barriers"])
        self._was_prev_step_coast = state["was_prev_step_coast"]
