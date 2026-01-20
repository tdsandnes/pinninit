from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch
import torch.nn as nn
from torch.optim import Optimizer as TorchOptimizer

from pinninit.optimizer.optimizer import Optimizer as PinninitOptimizer
from pinninit.training.trainer import Trainer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Any


class DummyTorchOptimizer(TorchOptimizer):
    def __init__(
        self, params: Iterable[nn.Parameter], *args: Any, **kwargs: Any
    ) -> None:
        defaults: dict[str, Any] = {}
        super().__init__(params, defaults)
        self.step_called = False
        self.zero_grad_called = False
        self.args = args
        self.kwargs = kwargs

    @overload
    def step(self, closure: None = ...) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        self.step_called = True
        loss = None
        if closure is not None:
            loss = closure()
        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.zero_grad_called = True
        super().zero_grad(set_to_none=set_to_none)


class DummyPinninitOptimizer(PinninitOptimizer):
    _requires_closure = False
    _requires_full_batch = False

    def __init__(
        self, params: Iterable[nn.Parameter], *args: Any, **kwargs: Any
    ) -> None:
        torch_optimizer = DummyTorchOptimizer(params, *args, **kwargs)
        super().__init__(torch_optimizer=torch_optimizer)

    def _compute_and_backward_loss(self) -> float:
        # Compute loss
        train_loss = self.trainer.compute_train_loss()

        # Backprop
        self.optimizer.zero_grad()
        train_loss.backward(retain_graph=True)  # type: ignore[no-untyped-call]

        return float(train_loss)

    def closure(self) -> float:
        if not self.requires_closure:
            raise RuntimeError("closure() called but requires_closure is False")
        return self._compute_and_backward_loss()

    def prepare_for_step(self) -> None:
        if not self.requires_closure:
            self._compute_and_backward_loss()


### Eventually replace this with a class in tests/trainer/dummy_trainer.py
class DummyOptimizerTrainer(Trainer):
    def __init__(self, params: Iterable[nn.Parameter], mult_factor: float = 1.0):
        self.params = params
        self.mult_factor = mult_factor

    def compute_train_loss(self) -> torch.Tensor:
        # Sum all parameters and multiply by a factor
        return torch.sum(torch.stack([p.sum() for p in self.params])) * self.mult_factor
