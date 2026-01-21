from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import ClassVar

    from .trainer import Trainer
    from .trainer_logger import TrainerLogger


class TrainerModule(ABC):
    # Class attributes
    runs_during_coast: ClassVar[bool]

    @abstractmethod
    def on_train_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        pass

    @abstractmethod
    def on_train_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        pass

    @abstractmethod
    def on_step_start(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        pass

    @abstractmethod
    def on_step_end(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        pass

    @abstractmethod
    def on_state_change(
        self, trainer: Trainer, logger: TrainerLogger | None = None
    ) -> None:
        """Called when another module changes the trainer state
        (e.g. optimizer updated)."""
        pass
