from __future__ import annotations

import pytest
import torch
from torch import nn

from pinninit.optimizer.optimizer import OPTIMIZER_REGISTRY, OptimizerLabels

from .dummy_optimizer import DummyOptimizerTrainer


@pytest.mark.parametrize("label", list(OptimizerLabels))  # type: ignore[misc]
def test_optimizer_step(label: OptimizerLabels) -> None:
    # Dummy parameter and trainer
    param = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
    mult_factor = 2.0
    trainer = DummyOptimizerTrainer(params=[param], mult_factor=mult_factor)

    # Lookup the optimizer class using the enum member
    optimizer_class = OPTIMIZER_REGISTRY[label]
    optimizer = optimizer_class([param])
    assert isinstance(optimizer.optimizer, torch.optim.Optimizer)
    optimizer.attach_trainer(trainer)

    optimizer.prepare_for_step()
    optimizer.step()

    # Check that the gradient exists
    assert param.grad is not None
    assert param.grad.item() == pytest.approx(mult_factor)


@pytest.mark.parametrize("label", list(OptimizerLabels))  # type: ignore[misc]
def test_optimizer_closure(label: OptimizerLabels) -> None:
    # Dummy parameter and trainer
    param = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
    mult_factor = 3.0
    trainer = DummyOptimizerTrainer(params=[param], mult_factor=mult_factor)

    # Lookup the optimizer class using the enum member
    optimizer_class = OPTIMIZER_REGISTRY[label]
    optimizer = optimizer_class([param])
    assert isinstance(optimizer.optimizer, torch.optim.Optimizer)
    optimizer.attach_trainer(trainer)

    if optimizer.requires_closure:
        # Run closure()
        loss = optimizer.closure()
        assert isinstance(loss, float)
        assert param.grad is not None
        # d(loss)/d(param) == mult_factor for loss = mult_factor * param
        assert param.grad.item() == pytest.approx(mult_factor)
    else:
        # If requires_closure=False, calling closure() should raise
        with pytest.raises(RuntimeError):
            optimizer.closure()
