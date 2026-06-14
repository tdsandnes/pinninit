from __future__ import annotations

from typing import TYPE_CHECKING

from ..config.config import OptimizerConfig
from ..optimizer.optimizer_registry import OPTIMIZER_REGISTRY

if TYPE_CHECKING:
    import torch.nn as nn

    from ..optimizer.optimizer import Optimizer


def build_optimizer(
    config: OptimizerConfig,
    params: list[nn.Parameter] | list[dict[str, object]],
) -> Optimizer:
    """
    Build an Optimizer from an OptimizerConfig.

    Parameters
    ----------
    config:
        Optimizer config with label, learning rate, and extra hyperparameters.
    params:
        Parameters to optimise (model parameters plus any learnable constants).

    Returns
    -------
    Optimizer
    """
    optimizer_cls = OPTIMIZER_REGISTRY[config.optimizer]
    return optimizer_cls(params, lr=config.learning_rate, **config.hyperparameters)
