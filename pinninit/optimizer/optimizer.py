from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

import torch.nn as nn
import torch.optim as optim

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Any, ClassVar

    from ..training.trainer import Trainer


class OptimizerLabels(str, Enum):
    """Enumeration of optimizer labels."""

    ADAM = "adam"
    LBFGS = "lbfgs"


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


class Adam(Optimizer):
    # Class attributes
    _requires_closure = False
    _requires_full_batch = False

    def __init__(
        self, params: Iterable[nn.Parameter], *args: Any, **kwargs: Any
    ) -> None:
        optimizer = optim.Adam(params, *args, **kwargs)
        super().__init__(optimizer)

    def prepare_for_step(self) -> None:
        # Compute loss
        train_loss = self.trainer.compute_train_loss()

        # Backprop
        self.optimizer.zero_grad()
        train_loss.backward(retain_graph=True)


class LBFGS(Optimizer):
    # Class attributes
    _requires_closure = True
    _requires_full_batch = True

    def __init__(
        self, params: Iterable[nn.Parameter], *args: Any, **kwargs: Any
    ) -> None:
        optimizer = optim.LBFGS(params, *args, **kwargs)
        super().__init__(optimizer)

    def prepare_for_step(self) -> None:
        pass

    def closure(self) -> float:
        # Compute loss
        train_loss = self.trainer.compute_train_loss()

        # Backprop
        self.optimizer.zero_grad()
        train_loss.backward(retain_graph=True)

        return float(train_loss.detach().item())


# Mapping from optimizer labels to classes
OPTIMIZER_REGISTRY: dict[OptimizerLabels, Callable[..., Optimizer]] = {
    OptimizerLabels.ADAM: Adam,
    OptimizerLabels.LBFGS: LBFGS,
}
