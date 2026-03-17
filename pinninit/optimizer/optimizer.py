from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch.optim as optim

if TYPE_CHECKING:
    from typing import ClassVar

    from ..training.trainer import Trainer


class Optimizer(ABC):
    # Class attributes
    _requires_closure: ClassVar[bool]
    _requires_full_batch: ClassVar[bool]

    def __init__(self, torch_optimizer: optim.Optimizer):
        self._torch_optimizer = torch_optimizer
        self._trainer: Trainer | None = None

    def attach_trainer(self, trainer: Trainer) -> None:
        self._trainer = trainer

    def closure(self) -> float:
        raise RuntimeError(
            f"{self.__class__.__name__} declares requires_closure=True but does not "
            "implement closure()."
        )

    @abstractmethod
    def prepare_for_step(self) -> None:
        pass

    def step(self) -> None:
        if self.requires_closure:
            self.optimizer.step(self.closure)
        else:
            self.optimizer.step()

    def clear(self) -> None:
        self.optimizer.state.clear()

    @property
    def requires_full_batch(self) -> bool:
        return self.__class__._requires_full_batch

    @property
    def requires_closure(self) -> bool:
        return self.__class__._requires_closure

    @property
    def optimizer(self) -> optim.Optimizer:
        return self._torch_optimizer

    @property
    def trainer(self) -> Trainer:
        if self._trainer is None:
            raise RuntimeError(f"{self.__class__.__name__} requires attached trainer.")
        return self._trainer
