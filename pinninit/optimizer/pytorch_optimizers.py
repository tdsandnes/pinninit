from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
import torch.optim as optim

from .optimizer import Optimizer

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any


class Adam(Optimizer):
    # Class attributes
    _requires_closure = False
    _requires_full_batch = False
    _requires_fixed_loss = False

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
        train_loss.backward(retain_graph=True)  # type: ignore[no-untyped-call]


class LBFGS(Optimizer):
    # Class attributes
    _requires_closure = True
    _requires_full_batch = True
    _requires_fixed_loss = True

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
        train_loss.backward(retain_graph=True)  # type: ignore[no-untyped-call]

        return float(train_loss.detach().item())
