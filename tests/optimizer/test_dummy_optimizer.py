from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn

from .dummy_optimizer import (
    DummyOptimizerTrainer,
    DummyPinninitOptimizer,
    DummyTorchOptimizer,
)


def test_dummy_torch_optimizer_step_and_zero_grad_called() -> None:
    # Create a dummy parameter
    param = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
    torch_optimizer = DummyTorchOptimizer([param])

    # Initially flags are False
    assert not torch_optimizer.step_called
    assert not torch_optimizer.zero_grad_called

    # Call zero_grad and step
    torch_optimizer.zero_grad()
    torch_optimizer.step()

    assert torch_optimizer.zero_grad_called
    assert torch_optimizer.step_called


@pytest.mark.parametrize("requires_closure", [True, False])  # type: ignore[misc]
def test_dummy_pinninit_optimizer_step(requires_closure: bool) -> None:
    # Dummy parameter and trainer
    param = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
    mult_factor = 2.0  # loss = 2.0 * param
    trainer = DummyOptimizerTrainer(params=[param], mult_factor=mult_factor)

    # pinninit dummy optimizer
    DummyPinninitOptimizer._requires_closure = requires_closure
    pinninit_optimizer = DummyPinninitOptimizer([param])
    pinninit_optimizer.attach_trainer(trainer)

    # Initially, torch optimizer flags are False
    torch_optimizer = cast(DummyTorchOptimizer, pinninit_optimizer.optimizer)
    assert not torch_optimizer.step_called
    assert not torch_optimizer.zero_grad_called

    # Call prepare_for_step
    pinninit_optimizer.prepare_for_step()

    if pinninit_optimizer.requires_closure:
        # Should not have called zero_grad
        assert not torch_optimizer.zero_grad_called

        # Check that no gradient produced yet
        assert param.grad is None

    else:
        # Should have called zero_grad
        assert torch_optimizer.zero_grad_called

        # Check that backward produced a gradient
        assert param.grad is not None

    pinninit_optimizer.step()

    # Gradient should exist
    assert param.grad is not None
    # d(loss)/d(param) == mult_factor for loss = mult_factor * param
    assert param.grad.item() == pytest.approx(mult_factor)


@pytest.mark.parametrize("requires_closure", [True, False])  # type: ignore[misc]
def test_dummy_pinninit_optimizer_closure(requires_closure: bool) -> None:
    # Dummy parameter and trainer
    param = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
    mult_factor = 3.0
    trainer = DummyOptimizerTrainer(params=[param], mult_factor=mult_factor)

    # pinninit dummy optimizer
    DummyPinninitOptimizer._requires_closure = requires_closure
    pinninit_optimizer = DummyPinninitOptimizer([param])
    pinninit_optimizer.attach_trainer(trainer)

    if requires_closure:
        # Run closure()
        loss = pinninit_optimizer.closure()
        assert isinstance(loss, float)
        assert param.grad is not None
        # d(loss)/d(param) == mult_factor for loss = mult_factor * param
        assert param.grad.item() == pytest.approx(mult_factor)
    else:
        # If requires_closure=False, calling closure() should raise
        with pytest.raises(RuntimeError):
            pinninit_optimizer.closure()


def test_dummy_optimizer_accepts_args_and_kwargs() -> None:
    param = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
    # Provide dummy arg and kwarg
    dummy_args = (1, "test")
    dummy_kwargs = {"kwarg1": 0.1, "kwarg2": True}
    pinninit_optimizer = DummyPinninitOptimizer([param], *dummy_args, **dummy_kwargs)

    # Attach trainer and perform step
    trainer = DummyOptimizerTrainer(params=[param])
    pinninit_optimizer.attach_trainer(trainer)
    pinninit_optimizer.prepare_for_step()
    pinninit_optimizer.step()

    torch_optimizer = cast(DummyTorchOptimizer, pinninit_optimizer.optimizer)

    # Check that args were passed correctly
    assert torch_optimizer.args == dummy_args

    # Check that kwargs were passed correctly
    for k, v in dummy_kwargs.items():
        assert torch_optimizer.kwargs[k] == v
