from collections.abc import Iterable
from enum import Enum
from typing import Any, Protocol

import torch
import torch.optim as optim


class OptimizerLabels(str, Enum):
    """Enumeration of optimizer labels."""

    ADAM = "adam"


class OptimizerProtocol(Protocol):
    def step(self, *args: Any, **kwargs: Any) -> None: ...
    def zero_grad(self, *args: Any, **kwargs: Any) -> None: ...


class Optimizer:
    """Base optimizer class. Either wrapper for existing optimizers or custom classes"""

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float):
        self.params = params
        self.lr = lr
        self.optimizer: OptimizerProtocol | None = None

    def step(self, *args: tuple[object, ...], **kwargs: dict[str, object]) -> None:
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialised in subclass")
        self.optimizer.step(*args, **kwargs)

    def zero_grad(self, *args: tuple[object, ...], **kwargs: dict[str, object]) -> None:
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialised in subclass")
        self.optimizer.zero_grad(*args, **kwargs)


class AdamOptimizer(Optimizer):
    """PyTorch implementation of Adam optimizer"""

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float):
        super().__init__(params, lr)
        self.optimizer = optim.Adam(self.params, lr=self.lr)


# Mapping from optimizer labels to classes
OPTIMIZER_REGISTRY = {
    OptimizerLabels.ADAM: AdamOptimizer,
}
