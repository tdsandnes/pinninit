from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

import torch.optim as optim

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from ..training.trainer import Trainer


class OptimizerLabels(str, Enum):
    """Enumeration of optimizer labels."""

    ADAM = "adam"
    LBFGS = "lbfgs"


class Optimizer(ABC):
    def __init__(self, optimizer: optim.Optimizer):
        self.optimizer = optimizer
        self._trainer: Trainer | None = None

    @abstractmethod
    def attach_trainer(self, trainer: Trainer) -> None:
        pass

    @abstractmethod
    def prepare_for_step(self) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        pass

    @property
    def trainer(self) -> Trainer:
        if self._trainer is None:
            raise RuntimeError(f"{self.__class__.__name__} requires attached trainer.")
        return self._trainer


class Adam(Optimizer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        optimizer = optim.Adam(*args, **kwargs)
        super().__init__(optimizer)

    def attach_trainer(self, trainer: Trainer) -> None:
        self._trainer = trainer

    def prepare_for_step(self) -> None:
        # Compute loss
        train_loss = self.trainer.compute_train_loss()

        # Backprop
        self.optimizer.zero_grad()
        train_loss.backward(retain_graph=True)  # type: ignore[no-untyped-call]

    def step(self) -> None:
        self.optimizer.step()


class LBFGS(Optimizer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        optimizer = optim.LBFGS(*args, **kwargs)
        super().__init__(optimizer)

    def attach_trainer(self, trainer: Trainer) -> None:
        self._trainer = trainer

    def prepare_for_step(self) -> None:
        pass

    def _closure(self) -> float:
        # Compute loss
        train_loss = self.trainer.compute_train_loss()

        # Backprop
        self.optimizer.zero_grad()
        train_loss.backward(retain_graph=True)  # type: ignore[no-untyped-call]

        return float(train_loss)

    def step(self) -> None:
        self.optimizer.step(self._closure)


# Mapping from optimizer labels to classes
OPTIMIZER_REGISTRY: dict[OptimizerLabels, Callable[..., Optimizer]] = {
    OptimizerLabels.ADAM: Adam,
    OptimizerLabels.LBFGS: LBFGS,
}
